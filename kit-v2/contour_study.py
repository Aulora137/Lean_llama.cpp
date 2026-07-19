#!/usr/bin/env python3
"""contour_study.py — offline study of cross-token KV quantization on real K/Q dumps.

Answers three questions from real KCAL-format K and Q captures (see
src/leankv-calib.h; reader adapted from ~/LeanKV/scripts/analyze_k_calib.py and
the kcal_check.py probe):

  Q1 — matched-budget cross-token mixed precision (top-X% keys at 4-bit, rest
       2-bit) vs uniform 3-bit / 2-bit, held-out vs oracle significance.
  Q2 — heavy-hitter stability: do the keys that mattered in the first half of
       the context still matter for second-half queries? (top-16 Jaccard)
  Q3 — peak-vs-trough: clamp top-p% vs bottom-p% logits per row to the row
       median; which perturbs the softmax more? (re-derivation of Henry's
       lost experiment)

Quantizer: the existing TurboQuant reference pipeline from
/home/junc/LeanKV/prototype/turboquant (randomized Hadamard rotation +
per-vector L2 scaling + analytic Lloyd-Max levels for N(0,1/d)), no QJL.
Stored-size tiers per Henry's accounting: 2-bit=2.5 bpe, 3-bit=3.5, 4-bit=4.5.

Method notes (verified against this repo's graph code, not assumed):
  * Each dump holds ONE contiguous sequence per layer (prefill 512 + 216/237
    in two ubatches + 3 single-token decode steps). Records are concatenated
    along the token axis. rec1[0] != rec0[0] rules out BOS-restart.
  * Attention scales are the model's true scales:
      - gemma4/E2B : logits = q.k * 1.0   (hparams.f_attention_scale = 1.0,
                     QK-norm arch; llama-hparams.cpp:854)
      - gemma3-4b  : logits = q.k / sqrt(head_dim)  (f_attention_scale =
                     1/sqrt(n_embd_head_k_full); llama-hparams.cpp:841)
      - LFM2.5     : logits = q.k / sqrt(head_dim)  (build_lfm2.cpp:132)
    Using the naive 1/sqrt(d) on E2B would flatten every row ~16x and void
    the study for that arch.
  * gemma4/E2B uses sliding-window attention, window 512, on layers where
    il % 5 != 4 (GGUF sliding_window_pattern). KV-owning global layers:
    {4, 9, 14}. Mask semantics: key s visible to query t iff t - s < 512
    (llama.cpp: `pos - cell.pos >= n_swa` => masked). gemma3-4b has
    n_swa = 1024 > T = 731, so its mask never truncates here. LFM2 attention
    layers are full-causal.
  * E2B global layers {4,9,14} have head_dim 512; the rest 256 (per-layer
    dims come from the records themselves).
  * Split: CALIB = first T//2 positions, EVAL = the rest. Calib keys are the
    "old context" that gets quantized; eval-half keys stay F16 (the tiered
    recent window, cf. LeanKV TIERED_KV_CACHE). All plan decisions use calib
    statistics only (HELD-OUT); ORACLE re-ranks with eval-query mass.
  * Significance = mean softmax mass received per key, averaged over the
    rows *eligible* to see that key (avoids penalising late keys for having
    had fewer opportunities), then averaged over q-heads. Precision is
    assigned per key token (all kv-heads together); per-kv-head resolution
    is kept for Q2.
  * No query subsampling: every eval row of every layer is scored.

Run:  /home/junc/LeanKV/.venv/bin/python kit-v2/contour_study.py
Output: contour_study_report.md + summary on stdout.
"""

from __future__ import annotations

import struct
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/junc/LeanKV/prototype")
try:
    import torch  # noqa: E402
except ModuleNotFoundError:  # system python3 has no torch on the Ryzen box
    raise SystemExit(
        "contour_study.py needs torch (the LeanKV prototype quantizer is torch-based).\n"
        "Run with:  ~/LeanKV/.venv/bin/python3 kit-v2/contour_study.py"
    )

from turboquant.quantizer import TurboQuantizer  # noqa: E402

ROOT = Path("/home/junc/Lean_llama.cpp")
REPORT = ROOT / "contour_study_report.md"

FILE_MAGIC = 0x4C41434B  # 'KCAL'
REC_MAGIC = 0x52434B4C   # 'LKCR'
GGML_F32, GGML_F16 = 0, 1

SINK_N = 4
TOPX = (10, 20, 30, 50)
P_LIST = (1, 2, 5, 10)
TOP16 = 16
TOP8 = 8
BITS = (2, 3, 4)
BPE = {2: 2.5, 3: 3.5, 4: 4.5}  # stored bits per element per tier

MODELS = [
    # name, K dump, Q dump, scale mode, swa (window, global-layer predicate)
    dict(name="gemma4-E2B", kf="e2b_kq_k.bin", qf="e2b_kq_q.bin",
         scale="one", swa_window=512, swa_global=lambda il: il % 5 == 4),
    dict(name="LFM2.5-1.2B", kf="lfm2_kq_k.bin", qf="lfm2_kq_q.bin",
         scale="rsqrt_hd", swa_window=None, swa_global=None),
    dict(name="gemma3-4b", kf="gemma3_kq_k.bin", qf="gemma3_kq_q.bin",
         scale="rsqrt_hd", swa_window=1024, swa_global=lambda il: il % 6 == 5),
]


# ---------------------------------------------------------------- KCAL reader
def read_kcal_layers(path: Path):
    """Return {il: [T, n_head, head_dim] float32} concatenating records in
    file order along the token axis (verified single contiguous sequence)."""
    per_layer: dict[int, list[np.ndarray]] = defaultdict(list)
    with path.open("rb") as f:
        magic, version = struct.unpack("<II", f.read(8))
        assert magic == FILE_MAGIC and version == 1, (path, hex(magic), version)
        while True:
            hdr = f.read(4)
            if len(hdr) < 4:
                break
            (rec_magic,) = struct.unpack("<I", hdr)
            assert rec_magic == REC_MAGIC, hex(rec_magic)
            il, dtype, ndims = struct.unpack("<III", f.read(12))
            ne = struct.unpack("<4I", f.read(16))
            _nb = struct.unpack("<4I", f.read(16))
            (nbytes,) = struct.unpack("<I", f.read(4))
            raw = f.read(nbytes)
            dt = np.float32 if dtype == GGML_F32 else np.float16
            data = np.frombuffer(raw, dtype=dt).astype(np.float32)
            hd, nh, T = ne[0], ne[1], ne[2]
            per_layer[il].append(data.reshape(T, nh, hd))
    return {il: np.concatenate(mats, axis=0) for il, mats in per_layer.items()}


# ---------------------------------------------------------------- math utils
def masked_softmax_logsoftmax(L: np.ndarray, valid: np.ndarray):
    """Row-wise softmax + log-softmax over the last axis, restricted to valid
    columns. Invalid columns get p=0, logp=-inf. float64 output."""
    Lm = np.where(valid, L.astype(np.float64), -np.inf)
    m = Lm.max(axis=-1, keepdims=True)
    z = np.exp(Lm - m)
    s = z.sum(axis=-1, keepdims=True)
    p = z / s
    logp = Lm - (m + np.log(s))
    return p, logp


def kl_rows(P: np.ndarray, logP: np.ndarray, logQ: np.ndarray) -> np.ndarray:
    """KL(P||Q) per row (sum over last axis); P==0 columns contribute 0."""
    diff = np.subtract(logP, logQ, out=np.zeros_like(logP), where=P > 0)
    return (P * diff).sum(axis=-1)


def topk_membership(P2d: np.ndarray, k: int) -> np.ndarray:
    """Bool [N, T] marking the top-k columns per row of probability matrix."""
    idx = np.argpartition(-P2d, k, axis=-1)[:, :k]
    mem = np.zeros(P2d.shape, dtype=bool)
    np.put_along_axis(mem, idx, True, axis=-1)
    return mem


# ---------------------------------------------------------------- per-layer study
def study_layer(K: np.ndarray, Q: np.ndarray, il: int, scale_mode: str,
                swa_window, tq_cache: dict):
    T, nkv, hd = K.shape
    Tq, nqh, hdq = Q.shape
    assert T == Tq and hd == hdq, (K.shape, Q.shape)
    group = nqh // nkv
    scale = 1.0 if scale_mode == "one" else 1.0 / np.sqrt(hd)

    Tc = T // 2
    Ne = T - Tc
    tpos = np.arange(T)
    epos = tpos[Tc:]                      # eval query positions

    qh_to_kv = np.arange(nqh) // group
    K_byq = K[:, qh_to_kv, :]             # [T, nqh, hd]

    # ---- logits (model-true scale) ----
    L_calib = np.einsum("thd,shd->hts", Q[:Tc], K_byq[:Tc], optimize=True) * scale
    L_eval = np.einsum("thd,shd->hts", Q[Tc:], K_byq, optimize=True) * scale

    # ---- visibility masks ----
    vis_c = tpos[:Tc, None] >= tpos[None, :Tc]          # [Tc, Tc] causal
    vis_e = epos[:, None] >= tpos[None, :]              # [Ne, T]
    if swa_window is not None and swa_window < T:
        # sliding layer: key s visible iff t - s < window
        win_c = (tpos[:Tc, None] - tpos[None, :Tc]) < swa_window
        win_e = (epos[:, None] - tpos[None, :]) < swa_window
        vis_c &= win_c
        vis_e &= win_e

    P_c, _ = masked_softmax_logsoftmax(L_calib, vis_c[None])
    P_e, logP_e = masked_softmax_logsoftmax(L_eval, vis_e[None])

    # ---- significance (mean softmax mass over eligible rows) ----
    n_elig_c = vis_c.sum(axis=0).astype(np.float64)          # [Tc] rows able to see key s
    sig_ho_h = P_c.sum(axis=1) / n_elig_c                    # [nqh, Tc]
    sig_ho = sig_ho_h.mean(axis=0)                           # per-token, held-out

    n_elig_e = vis_e[:, :Tc].sum(axis=0).astype(np.float64)  # [Tc]
    sig_or_h = P_e[:, :, :Tc].sum(axis=1) / np.maximum(n_elig_e, 1)
    sig_or = sig_or_h.mean(axis=0)                           # per-token, oracle

    # ---- Q2: heavy-hitter stability (per kv-head) ----
    sig_ho_kv = sig_ho_h.reshape(nkv, group, Tc).mean(axis=1)      # [nkv, Tc]
    Pe_kv = P_e.reshape(nkv, group, Ne, T).mean(axis=1)            # [nkv, Ne, T]
    cons_mem = topk_membership(sig_ho_kv, TOP16)                   # [nkv, Tc]
    ev_calib = Pe_kv[:, :, :Tc].reshape(nkv * Ne, Tc)
    ev_mem = topk_membership(ev_calib, TOP16).reshape(nkv, Ne, Tc)
    inter = (ev_mem & cons_mem[:, None, :]).sum(axis=-1)           # [nkv, Ne]
    jac = inter / (2 * TOP16 - inter)
    q2 = dict(mean=float(jac.mean()), min=float(jac.min()),
              p10=float(np.percentile(jac, 10)))

    # ---- quantize calib keys with the TurboQuant reference pipeline ----
    Lq = {}
    relerr = {}
    Kc = K[:Tc]                                                    # [Tc, nkv, hd]
    x = torch.from_numpy(np.ascontiguousarray(Kc.transpose(1, 0, 2))).float()[None]
    for b in BITS:
        tq: TurboQuantizer = tq_cache[(hd, b)]
        with torch.no_grad():
            qkv = tq.quantize(x, layer_idx=il)
            khat = tq.dequantize(qkv, layer_idx=il, apply_inverse_rot=True)
        Khat = khat.squeeze(0).numpy().transpose(1, 0, 2)          # [Tc, nkv, hd]
        relerr[b] = float(np.linalg.norm(Khat - Kc) / (np.linalg.norm(Kc) + 1e-30))
        Khat_byq = Khat[:, qh_to_kv, :]
        Lq[b] = np.einsum("thd,shd->hts", Q[Tc:], Khat_byq, optimize=True) * scale

    # ---- Q1 plans ----
    n4 = {X: int(round(X / 100 * Tc)) for X in TOPX}
    order_ho = np.argsort(sig_ho)
    order_or = np.argsort(sig_or)
    sink = np.zeros(Tc, dtype=bool)
    sink[:SINK_N] = True

    def sel(order, n):
        m = np.zeros(Tc, dtype=bool)
        m[order[Tc - n:]] = True
        return m

    plans = {"u2": ("uniform", 2), "u3": ("uniform", 3), "u4": ("uniform", 4),
             "sink4": ("mask", sink.copy())}
    for X in TOPX:
        plans[f"ho{X}"] = ("mask", sel(order_ho, n4[X]))
        plans[f"ho{X}s"] = ("mask", sel(order_ho, n4[X]) | sink)
        plans[f"or{X}"] = ("mask", sel(order_or, n4[X]))

    q1 = {}
    P_e_flat = P_e.reshape(-1, T)
    top1_ref = P_e_flat.argmax(axis=-1)
    mem_ref = topk_membership(P_e_flat, TOP8)
    for pname, (kind, spec) in plans.items():
        Ld = L_eval.copy()
        if kind == "uniform":
            Ld[:, :, :Tc] = Lq[spec]
            n4b = Tc if spec == 4 else 0
            bpe = BPE[spec]
        else:
            is4 = spec
            Ld[:, :, :Tc] = np.where(is4[None, None, :], Lq[4], Lq[2])
            n4b = int(is4.sum())
            bpe = (4.5 * n4b + 2.5 * (Tc - n4b)) / Tc
        Pd, logPd = masked_softmax_logsoftmax(Ld, vis_e[None])
        kl = kl_rows(P_e, logP_e, logPd)                      # [nqh, Ne]
        Pd_flat = Pd.reshape(-1, T)
        top1 = float((Pd_flat.argmax(axis=-1) == top1_ref).mean())
        mem_q = topk_membership(Pd_flat, TOP8)
        inter8 = (mem_q & mem_ref).sum(axis=-1)
        jac8 = float((inter8 / (2 * TOP8 - inter8)).mean())
        q1[pname] = dict(kl=float(kl.mean()), top1=top1, jac8=jac8,
                         bpe=float(bpe), n4=n4b)

    # ---- Q3: peak vs trough clamping on F16 eval rows ----
    Lm = np.where(vis_e[None], L_eval.astype(np.float64), np.inf)
    S = np.sort(Lm, axis=-1)                                  # valid first, +inf tail
    Llen = vis_e.sum(axis=-1)                                 # [Ne]
    li = Llen[None, :, None]
    med = (np.take_along_axis(S, (Llen[None, :, None] - 1) // 2, axis=-1)
           + np.take_along_axis(S, Llen[None, :, None] // 2, axis=-1)) / 2  # [nqh,Ne,1]
    q3 = {}
    for p in P_LIST:
        n = np.maximum(1, np.round(p / 100 * Llen).astype(np.int64))[None, :, None]
        thr_top = np.take_along_axis(S, li - n, axis=-1)      # n-th largest
        thr_bot = np.take_along_axis(S, n - 1, axis=-1)       # n-th smallest
        Le64 = L_eval.astype(np.float64)
        Ltop = np.where(vis_e[None] & (Le64 >= thr_top), med, Le64)
        Lbot = np.where(vis_e[None] & (Le64 <= thr_bot), med, Le64)
        _, logPt = masked_softmax_logsoftmax(Ltop, vis_e[None])
        _, logPb = masked_softmax_logsoftmax(Lbot, vis_e[None])
        kt = float(kl_rows(P_e, logP_e, logPt).mean())
        kb = float(kl_rows(P_e, logP_e, logPb).mean())
        q3[p] = dict(kl_top=kt, kl_bot=kb,
                     ratio=(kb / kt) if kt > 0 else float("inf"))

    meta = dict(T=T, Tc=Tc, Ne=Ne, nqh=nqh, nkv=nkv, hd=hd, scale=scale,
                swa=swa_window if (swa_window is not None and swa_window < T) else None)
    return dict(meta=meta, q1=q1, q2=q2, q3=q3, relerr=relerr)


# ---------------------------------------------------------------- per-model driver
def run_model(cfg):
    t0 = time.time()
    Ks = read_kcal_layers(ROOT / cfg["kf"])
    Qs = read_kcal_layers(ROOT / cfg["qf"])
    layers = sorted(Ks)
    assert layers == sorted(Qs), "K/Q layer sets differ"

    # TurboQuant instances per (head_dim, bits); rotations indexed by layer id
    tq_cache = {}
    max_il = max(layers) + 1
    for il in layers:
        hd = Ks[il].shape[2]
        for b in BITS:
            if (hd, b) not in tq_cache:
                tq_cache[(hd, b)] = TurboQuantizer(
                    n_layers=max_il, head_dim=hd, bits=b, group_size=None,
                    rotation_strategy="randomized_hadamard", use_qjl=False,
                    seed=42, device="cpu")

    res = {}
    for il in layers:
        swa = None
        if cfg["swa_window"] is not None and not cfg["swa_global"](il):
            swa = cfg["swa_window"]
        res[il] = study_layer(Ks[il], Qs[il], il, cfg["scale"], swa, tq_cache)
        m = res[il]["meta"]
        print(f"  [{cfg['name']}] layer {il:2d} done "
              f"(hd={m['hd']}, swa={m['swa']}, {time.time()-t0:5.1f}s)", flush=True)
    return res


# ---------------------------------------------------------------- aggregation
def model_plan_table(res):
    """Mean over layers of per-layer q1 metrics -> {plan: dict}."""
    plans = next(iter(res.values()))["q1"].keys()
    out = {}
    for p in plans:
        kls = [res[il]["q1"][p]["kl"] for il in res]
        out[p] = dict(
            kl=float(np.mean(kls)),
            kl_med=float(np.median(kls)),
            top1=float(np.mean([res[il]["q1"][p]["top1"] for il in res])),
            jac8=float(np.mean([res[il]["q1"][p]["jac8"] for il in res])),
            bpe=float(np.mean([res[il]["q1"][p]["bpe"] for il in res])),
            bpe_max=float(np.max([res[il]["q1"][p]["bpe"] for il in res])),
        )
    return out


def evaluate_gates(all_res):
    verdicts = {}
    # Q1: any held-out plan with audited bpe < 3.5 beating u3 mean KL, on >= 2 archs
    ho_plans = [f"ho{X}" for X in TOPX] + [f"ho{X}s" for X in TOPX]
    per_plan_wins = {p: [] for p in ho_plans}
    per_model_win = {}
    for name, res in all_res.items():
        tab = model_plan_table(res)
        u3 = tab["u3"]["kl"]
        winners = [p for p in ho_plans
                   if tab[p]["bpe_max"] < 3.5 and tab[p]["kl"] < u3]
        per_model_win[name] = winners
        for p in winners:
            per_plan_wins[p].append(name)
    q1_go_plans = {p: ms for p, ms in per_plan_wins.items() if len(ms) >= 2}
    verdicts["q1"] = dict(go=bool(q1_go_plans), plans=q1_go_plans,
                          per_model=per_model_win)
    # Q2: mean Jaccard >= 0.5
    q2_means = {name: float(np.mean([res[il]["q2"]["mean"] for il in res]))
                for name, res in all_res.items()}
    verdicts["q2"] = dict(means=q2_means,
                          supportive={n: v >= 0.5 for n, v in q2_means.items()})
    # Q3: any layer with trough_dominance > 1.5 at every p
    q3_layers = {}
    for name, res in all_res.items():
        consistent = [il for il in res
                      if all(res[il]["q3"][p]["ratio"] > 1.5 for p in P_LIST)]
        q3_layers[name] = consistent
    verdicts["q3"] = dict(layers=q3_layers,
                          stands=any(v for v in q3_layers.values()))
    return verdicts


# ---------------------------------------------------------------- report
PLAN_ORDER = ["u2", "u3", "u4", "sink4",
              "ho10", "ho20", "ho30", "ho50",
              "ho10s", "ho20s", "ho30s", "ho50s",
              "or10", "or20", "or30", "or50"]

PLAN_DESC = {
    "u2": "uniform 2-bit", "u3": "uniform 3-bit", "u4": "uniform 4-bit",
    "sink4": "first-4 sinks @4b, rest 2-bit",
    "ho10": "held-out top-10% @4b", "ho20": "held-out top-20% @4b",
    "ho30": "held-out top-30% @4b", "ho50": "held-out top-50% @4b",
    "ho10s": "held-out top-10% +sinks", "ho20s": "held-out top-20% +sinks",
    "ho30s": "held-out top-30% +sinks", "ho50s": "held-out top-50% +sinks",
    "or10": "oracle top-10% @4b", "or20": "oracle top-20% @4b",
    "or30": "oracle top-30% @4b", "or50": "oracle top-50% @4b",
}


def fmt(x, prec=4):
    if x == 0:
        return "0"
    if abs(x) < 1e-3 or abs(x) >= 1e4:
        return f"{x:.2e}"
    return f"{x:.{prec}f}"


def write_report(all_res, verdicts):
    lines = []
    a = lines.append
    a("# Contour Study — cross-token KV quantization on real K/Q dumps")
    a("")
    a(f"Generated by `kit-v2/contour_study.py` on {time.strftime('%Y-%m-%d %H:%M')}. "
      "Real KCAL captures (post-RoPE K and Q) from three architectures; quantizer is "
      "the TurboQuant reference pipeline (`~/LeanKV/prototype/turboquant`): randomized "
      "Hadamard + per-vector L2 scale + analytic Lloyd-Max levels, no QJL. "
      "Stored-size tiers: 2-bit=2.5 bpe, 3-bit=3.5, 4-bit=4.5.")
    a("")
    a("## Method (what was actually computed)")
    a("")
    a("- Each dump is ONE contiguous sequence per layer (prefill 512 + 216/237 in two "
      "ubatches + 3 decode steps), concatenated along the token axis. "
      "T = 731 (gemma4-E2B, gemma3-4b), 752 (LFM2.5). No query subsampling.")
    a("- **Model-true attention scales** (from this repo's graph code, not the naive "
      "1/sqrt(d) for all): gemma4-E2B logits = q·k × **1.0** (`f_attention_scale = 1.0`, "
      "QK-norm arch, `llama-hparams.cpp:854`); gemma3-4b and LFM2.5 = q·k/√head_dim. "
      "Using 1/√256 on E2B would flatten every row ~16× and void that arch.")
    a("- **SWA honored**: gemma4-E2B sliding layers (il%5≠4, window 512; KV-owning "
      "global layers {4,9,14}) mask key s from query t when t−s ≥ 512. gemma3-4b "
      "n_swa=1024 > T, never truncates. E2B global layers have head_dim 512, others 256.")
    a("- Split: CALIB = first T//2 positions (365/376), EVAL = rest. Calib-half keys are "
      "quantized (the compressed old context); eval-half keys stay F16 (the hot recent "
      "window, as in the LeanKV tiered-cache design). Metrics on EVAL rows only; "
      "the budget audit `bpe = 4.5x + 2.5(1−x)` is over calib keys.")
    a("- Significance = mean softmax mass received per key over rows *eligible* to see "
      "it (HELD-OUT: calib rows only; ORACLE: eval rows), averaged over q-heads; "
      "precision assigned per key token across all kv-heads. Q2 keeps per-kv-head "
      "resolution. Q3 clamps are per q-head row; ties at the clamp threshold may clamp "
      "a few extra elements (negligible on f32 data).")
    a("- KL is KL(P_f16 || P_quant) in nats, per eval row, averaged over rows, heads, "
      "then layers. top-1/top-8 are over the full visible row (quantized old keys + "
      "F16 recent keys).")
    a("")

    for name, res in all_res.items():
        layers = sorted(res)
        m0 = res[layers[0]]["meta"]
        tab = model_plan_table(res)
        a(f"## {name}")
        a("")
        hds = sorted({res[il]['meta']['hd'] for il in layers})
        swa_l = [il for il in layers if res[il]["meta"]["swa"]]
        re2 = np.mean([res[il]["relerr"][2] for il in layers])
        re3 = np.mean([res[il]["relerr"][3] for il in layers])
        re4 = np.mean([res[il]["relerr"][4] for il in layers])
        a(f"T={m0['T']}, calib={m0['Tc']}, eval rows={m0['Ne']}, "
          f"n_head={m0['nqh']}, n_kv={m0['nkv']}, head_dim={hds}, "
          f"KV layers={len(layers)}"
          + (f", SWA(512) layers={swa_l}" if swa_l else ", full causal")
          + f". TQ reconstruction rel-err: 2b={re2:.3f} 3b={re3:.3f} 4b={re4:.3f}.")
        a("")
        a("### Q1 — matched-budget plans (mean over layers, EVAL rows)")
        a("")
        a("| plan | bpe | mean KL | med-layer KL | top-1 same | top-8 Jacc |")
        a("|---|---|---|---|---|---|")
        for p in PLAN_ORDER:
            t = tab[p]
            star = " **<u3**" if (t["bpe_max"] < 3.5 and t["kl"] < tab["u3"]["kl"]
                                  and p.startswith(("ho", "sink"))) else ""
            a(f"| {p} ({PLAN_DESC[p]}) | {t['bpe']:.2f} | {fmt(t['kl'])}{star} | "
              f"{fmt(t['kl_med'])} | {t['top1']*100:.1f}% | {t['jac8']:.3f} |")
        a("")
        a("### Per-layer detail (KL: u2 / u3 / ho30 / or30; Q2 Jaccard; Q3 trough ratio)")
        a("")
        a("| layer | swa | KL u2 | KL u3 | KL ho30 (3.10 bpe) | KL or30 | "
          "ho30<u3? | Q2 jac mean | Q2 jac min | Q3 ratio p=1/2/5/10 | trough>1.5 all p |")
        a("|---|---|---|---|---|---|---|---|---|---|---|")
        for il in layers:
            r = res[il]
            q1 = r["q1"]
            q3r = [r["q3"][p]["ratio"] for p in P_LIST]
            flag = "YES" if all(x > 1.5 for x in q3r) else ("gt1" if all(x > 1 for x in q3r) else "")
            win = "WIN" if q1["ho30"]["kl"] < q1["u3"]["kl"] else ""
            a(f"| {il} | {'512' if r['meta']['swa'] else '-'} | {fmt(q1['u2']['kl'])} | "
              f"{fmt(q1['u3']['kl'])} | {fmt(q1['ho30']['kl'])} | {fmt(q1['or30']['kl'])} | "
              f"{win} | {r['q2']['mean']:.3f} | {r['q2']['min']:.3f} | "
              f"{'/'.join(fmt(x,2) for x in q3r)} | {flag} |")
        a("")

    # cross-arch summary
    a("## Cross-arch summary")
    a("")
    a("| model | KL u2 (2.5) | KL u3 (3.5) | best held-out <3.5bpe | its KL | its bpe | "
      "oracle30 KL | Q2 jac mean | Q3 layers ratio>1.5 all p |")
    a("|---|---|---|---|---|---|---|---|---|")
    summary_rows = []
    for name, res in all_res.items():
        tab = model_plan_table(res)
        cand = [(tab[p]["kl"], p) for p in tab
                if p.startswith("ho") and tab[p]["bpe_max"] < 3.5]
        cand += [(tab["sink4"]["kl"], "sink4")]
        bk, bp = min(cand)
        q2m = float(np.mean([res[il]["q2"]["mean"] for il in res]))
        n15 = len(verdicts["q3"]["layers"][name])
        row = (f"| {name} | {fmt(tab['u2']['kl'])} | {fmt(tab['u3']['kl'])} | {bp} | "
               f"{fmt(bk)} | {tab[bp]['bpe']:.2f} | {fmt(tab['or30']['kl'])} | "
               f"{q2m:.3f} | {n15}/{len(res)} |")
        a(row)
        summary_rows.append(row)
    a("")

    # verdicts
    a("## GO / NO-GO")
    a("")
    v1 = verdicts["q1"]
    a(f"### Q1 — cross-token mixed precision: **{'GO' if v1['go'] else 'NO-GO'}**")
    a("")
    if v1["go"]:
        pl = "; ".join(f"`{p}` wins on {', '.join(ms)}" for p, ms in v1["plans"].items())
        a(f"Gate: a held-out mixed plan beats uniform 3-bit on mean KL at lower audited "
          f"bpe on >=2 archs. Met: {pl}.")
    else:
        a("Gate NOT met: no single held-out plan with audited bpe < 3.5 beats uniform "
          "3-bit mean KL on >= 2 architectures.")
    a(f"Per-arch winning held-out plans (bpe<3.5, KL<u3): " +
      "; ".join(f"{n}: {ws if ws else 'none'}" for n, ws in v1["per_model"].items()) + ".")
    a("")
    v2 = verdicts["q2"]
    sup = [n for n, s in v2["supportive"].items() if s]
    a(f"### Q2 — heavy-hitter stability: **{'SUPPORTIVE' if len(sup) == len(v2['means']) else ('MIXED' if sup else 'NOT SUPPORTIVE')}**")
    a("")
    a("Mean top-16 Jaccard (calib-consensus vs each eval query, per kv-head): " +
      "; ".join(f"{n}: {v:.3f} ({'>=0.5 supportive' if v >= 0.5 else '<0.5 not supportive'})"
                for n, v in v2["means"].items()) + ".")
    a("")
    v3 = verdicts["q3"]
    a(f"### Q3 — peak-vs-trough (Henry's lost experiment): "
      f"**{'FINDING STANDS' if v3['stands'] else 'DOES NOT REPLICATE'}**")
    a("")
    a("Layers with trough_dominance = KL_bottom/KL_top > 1.5 at every p in {1,2,5,10}: " +
      "; ".join(f"{n}: {ls if ls else 'none'}" for n, ls in v3["layers"].items()) + ".")
    a("")
    REPORT.write_text("\n".join(lines))
    return summary_rows


# ---------------------------------------------------------------- main
def main():
    t0 = time.time()
    all_res = {}
    for cfg in MODELS:
        print(f"== {cfg['name']} ==", flush=True)
        all_res[cfg["name"]] = run_model(cfg)
    verdicts = evaluate_gates(all_res)
    rows = write_report(all_res, verdicts)

    print("\n================ CROSS-ARCH SUMMARY ================")
    print("| model | KL u2 | KL u3 | best held-out <3.5bpe | its KL | its bpe | "
          "oracle30 KL | Q2 jac | Q3 layers>1.5 |")
    for r in rows:
        print(r)
    v1, v2, v3 = verdicts["q1"], verdicts["q2"], verdicts["q3"]
    print(f"\nQ1 mixed-precision : {'GO' if v1['go'] else 'NO-GO'}"
          + (f" (plans {list(v1['plans'])})" if v1["go"] else ""))
    print("   per-arch wins   : " + "; ".join(
        f"{n}: {ws if ws else 'none'}" for n, ws in v1["per_model"].items()))
    print(f"Q2 stability      : " + "; ".join(
        f"{n}={v:.3f}{'(supportive)' if v >= 0.5 else ''}" for n, v in v2["means"].items()))
    print(f"Q3 trough>peak    : {'FINDING STANDS' if v3['stands'] else 'DOES NOT REPLICATE'} "
          + "; ".join(f"{n}: {ls}" for n, ls in v3["layers"].items()))
    print(f"\nreport: {REPORT}")
    print(f"total runtime: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
