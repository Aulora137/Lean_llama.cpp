#!/usr/bin/env python3
"""vq_study.py — offline VECTOR QUANTIZATION (PQ / RVQ) study for the KV cache.

Question: at matched storage budgets, how much of the scalar->VQ rate-distortion
gap (~2.7 dB at 2-3 bits) does product / residual vector quantization of K
recover, measured on REAL attention (softmax KL vs F16) from KCAL K/Q dumps?

Reuses the contour_study.py scaffolding (KCAL reader, GQA mapping, causal/SWA
logit construction with model-true scales, calib/eval split, KL/top1/jaccard
metrics, TurboQuantizer scalar baseline). See that file for dump provenance.

Design (differences from contour_study):
  * Codebooks are fit on CALIB-half K vectors ONLY; the FULL K matrix used in
    eval attention (calib + eval positions) is then encoded/decoded with those
    frozen codebooks. Eval-half keys are therefore HELD OUT from the fit.
  * Scalar baselines are computed in BOTH regimes:
      - u{2,3}_half : contour-study-identical (only calib-half keys quantized,
        eval keys F16) — must reproduce the quoted contour numbers exactly
        (pipeline verification).
      - u{2,3}_full : full K matrix quantized — the apples-to-apples baseline
        for the VQ configs (same regime), used for gap_closure.
  * Space axis: "rot" = per-layer randomized-Hadamard rotation (the same
    matrices TurboQuantizer uses, seed 42+il) then per-32-element amax block
    scales (fp16), exactly the in-tree TQ2_0/TQ3_0 block scheme
    (ggml/src/ggml-tq.c: d = amax, normalize to [-1,1], store fp16 d).
    "raw" = same block-scale scheme on unrotated K (lets VQ exploit channel
    correlation that the rotation deliberately destroys).
  * Budget accounting (strict): effective bpe = code_bits + 0.5 (fp16 scale
    per 32 elems), identical to the TQ ladder (TQ2=2.5, TQ3=3.5). Codebooks
    are amortized across all cached tokens but their absolute per-layer size
    is reported (n_pos * K * m * 2 bytes). No per-token side info beyond
    codes + block scales.
  * k-means: k-means++ init, fixed seeds, >=25 Lloyd iterations (cap 40,
    tol 1e-5), empty clusters reseeded to farthest points. If a codebook has
    K >= N_samples it degenerates to "memorize all calib subvectors" (flagged
    DEGEN in the report). Calib sample counts here are tiny (365..3008 per
    codebook position, far below the 400k subsample cap -> no subsampling).

Run (staged, each stage < 10 min, all foreground):
  VENV=~/LeanKV/.venv/bin/python3
  $VENV kit-v2/vq_study.py --model e2b
  $VENV kit-v2/vq_study.py --model lfm2 --refit     # + codebook-generalization check
  $VENV kit-v2/vq_study.py --model gemma3 --layers 0:12   (then 12:23, 23:34)
  $VENV kit-v2/vq_study.py --aggregate
Output: vq_study_report.md + summary/verdicts on stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import contour_study as cs  # noqa: E402  (pulls in torch + TurboQuantizer)
import torch  # noqa: E402

torch.set_num_threads(8)          # llama-perplexity may share the box
torch.set_num_interop_threads(1)

ROOT = cs.ROOT
REPORT = ROOT / "vq_study_report.md"
PARTIAL_DIR = Path("/tmp/claude-1000/-home-junc-rikuri-rikurinode/"
                   "0a7b0d58-b4e2-4aa3-bb15-3e35bc56022f/scratchpad")
PARTIAL_DIR.mkdir(parents=True, exist_ok=True)

BLOCK = 32
TOP8 = 8
KM_MIN_ITERS = 25
KM_MAX_ITERS = 40
KM_TOL = 1e-5

SHORT = {"e2b": "gemma4-E2B", "lfm2": "LFM2.5-1.2B", "gemma3": "gemma3-4b"}
# contour_study cross-arch summary (must reproduce, 4 dp): (u2, u3) half-K KL
QUOTED = {"gemma4-E2B": (0.0948, 0.0274),
          "LFM2.5-1.2B": (0.1523, 0.0415),
          "gemma3-4b": (0.2065, 0.0458)}

# name, kind, params, code bits/dim
VQ_CONFIGS = [
    ("pq_m4_k256",  "pq",  dict(m=4, K=256),      2.0),
    ("pq_m4_k1024", "pq",  dict(m=4, K=1024),     2.5),
    ("pq_m2_k64",   "pq",  dict(m=2, K=64),       3.0),
    ("rvq_m4_2x16", "rvq", dict(m=4, Ks=(16, 16)), 2.0),
    ("rvq_m4_2x32", "rvq", dict(m=4, Ks=(32, 32)), 2.5),
]
SPACES = ("rot", "raw")
FLIP_ONLY = os.environ.get("VQ_FLIP_ONLY") == "1"
CODE_BITS = {n: cb for n, _, _, cb in VQ_CONFIGS}
EFF_BPE = {n: cb + 0.5 for n, cb in CODE_BITS.items()}
CFG25 = [n for n, cb in CODE_BITS.items() if cb == 2.0]   # 2.5 bpe effective
DESC = {
    "pq_m4_k256":  "PQ m=4, 256 cw",
    "pq_m4_k1024": "PQ m=4, 1024 cw",
    "pq_m2_k64":   "PQ m=2, 64 cw",
    "rvq_m4_2x16": "RVQ m=4, 2x16 cw",
    "rvq_m4_2x32": "RVQ m=4, 2x32 cw",
}


def stable_seed(*parts) -> int:
    return zlib.crc32("|".join(str(p) for p in parts).encode()) & 0x7FFFFFFF


# ------------------------------------------------------------------ k-means
def _d2_chunked_argmin(X: torch.Tensor, C: torch.Tensor, budget=int(6e7)):
    """codes, mind for batched [P,N,m] points vs [P,K,m] centroids."""
    P, N, m = X.shape
    K = C.shape[1]
    pc = max(1, budget // max(1, N * K))
    codes = torch.empty(P, N, dtype=torch.long)
    mind = torch.empty(P, N)
    for p0 in range(0, P, pc):
        p1 = min(P, p0 + pc)
        Xc, Cc = X[p0:p1], C[p0:p1]
        d2 = ((Xc * Xc).sum(-1, keepdim=True)
              - 2.0 * torch.bmm(Xc, Cc.transpose(1, 2))
              + (Cc * Cc).sum(-1).unsqueeze(1))
        mn, am = d2.min(-1)
        codes[p0:p1] = am
        mind[p0:p1] = mn
    return codes, mind


def kmeanspp_init(X: torch.Tensor, K: int, gen: torch.Generator) -> torch.Tensor:
    P, N, m = X.shape
    C = torch.empty(P, K, m)
    idx0 = torch.randint(N, (P,), generator=gen)
    C[:, 0] = X[torch.arange(P), idx0]
    d2 = ((X - C[:, 0:1]) ** 2).sum(-1)          # [P, N]
    for k in range(1, K):
        probs = d2.clamp_min(0)
        s = probs.sum(-1, keepdim=True)
        probs = torch.where(s > 0, probs, torch.ones_like(probs))
        idx = torch.multinomial(probs, 1, generator=gen).squeeze(-1)
        C[:, k] = X[torch.arange(P), idx]
        d2 = torch.minimum(d2, ((X - C[:, k:k + 1]) ** 2).sum(-1))
    return C


def fit_codebook(X: torch.Tensor, K: int, gen: torch.Generator):
    """Batched k-means over P independent codebook positions. X [P,N,m]."""
    P, N, m = X.shape
    if N <= K:  # degenerate: memorize every calib subvector, pad with dupes
        pad = X[:, :1].expand(P, K - N, m)
        return torch.cat([X, pad], dim=1).contiguous(), True, 0
    C = kmeanspp_init(X, K, gen)
    arangeP = torch.arange(P)
    iters = 0
    for it in range(KM_MAX_ITERS):
        iters = it + 1
        codes, mind = _d2_chunked_argmin(X, C)
        flat = (codes + arangeP.unsqueeze(1) * K).reshape(-1)
        sums = torch.zeros(P * K, m).index_add_(0, flat, X.reshape(-1, m))
        cnts = torch.bincount(flat, minlength=P * K).reshape(P, K)
        newC = sums.reshape(P, K, m) / cnts.unsqueeze(-1).clamp_min(1)
        empty = cnts == 0
        if bool(empty.any()):                      # reseed to farthest points
            nmax = min(int(empty.sum(1).max()), N)
            far = mind.topk(nmax, dim=1).indices   # [P, nmax]
            for p in torch.nonzero(empty.any(1)).flatten().tolist():
                es = torch.nonzero(empty[p]).flatten()
                take = min(len(es), nmax)
                newC[p, es[:take]] = X[p, far[p, :take]]
        shift = float((newC - C).abs().max())
        C = newC
        if iters >= KM_MIN_ITERS and shift < KM_TOL:
            break
    return C, False, iters


def gather_cw(C: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
    """C [P,K,m], codes [P,N] -> [P,N,m]."""
    m = C.shape[-1]
    return torch.gather(C, 1, codes.unsqueeze(-1).expand(-1, -1, m))


# ------------------------------------------------------- block-scale + spaces
def block_normalize(X: torch.Tensor, hd: int):
    """In-tree TQ scheme (ggml-tq.c): per-32 amax scale; normalize by fp32
    amax; the STORED scale is fp16 (used at decode). X [Tn, hd]."""
    nb = hd // BLOCK
    B = X.reshape(-1, nb, BLOCK)
    amax = B.abs().amax(-1)                               # [Tn, nb] fp32
    d16 = amax.to(torch.float16).to(torch.float32)        # stored fp16 scale
    Xn = B / amax.clamp_min(1e-10).unsqueeze(-1)
    return Xn.reshape(-1, hd), d16


def block_denormalize(Xhat: torch.Tensor, d16: torch.Tensor, hd: int):
    nb = hd // BLOCK
    return (Xhat.reshape(-1, nb, BLOCK) * d16.unsqueeze(-1)).reshape(-1, hd)


def vq_roundtrip(K_np: np.ndarray, Tc: int, R: torch.Tensor, space: str,
                 kind: str, prm: dict, seed_key: tuple, fit_on_eval=False):
    """Encode/decode the FULL K matrix with codebooks fit on one half.
    Returns (Khat [T,nkv,hd] np.f32, codebook_bytes, degenerate, n_fit)."""
    T, nkv, hd = K_np.shape
    m = prm["m"]
    P = hd // m
    with torch.inference_mode():
        X = torch.from_numpy(np.ascontiguousarray(K_np.reshape(T * nkv, hd)))
        if space == "rot":
            X = X @ R.T
        Xn, d16 = block_normalize(X, hd)
        Xp = Xn.reshape(-1, P, m).transpose(0, 1).contiguous()   # [P, Tn, m]
        rows = slice(Tc * nkv, None) if fit_on_eval else slice(0, Tc * nkv)
        Xfit = Xp[:, rows].contiguous()
        n_fit = Xfit.shape[1]
        gen = torch.Generator().manual_seed(stable_seed(*seed_key, "s1"))
        if kind == "pq":
            C, degen, _ = fit_codebook(Xfit, prm["K"], gen)
            codes, _ = _d2_chunked_argmin(Xp, C)
            Xhat_p = gather_cw(C, codes)
            cb_bytes = P * prm["K"] * m * 2
        else:  # rvq
            K1, K2 = prm["Ks"]
            C1, dg1, _ = fit_codebook(Xfit, K1, gen)
            c1f, _ = _d2_chunked_argmin(Xfit, C1)
            R1 = Xfit - gather_cw(C1, c1f)
            gen2 = torch.Generator().manual_seed(stable_seed(*seed_key, "s2"))
            C2, dg2, _ = fit_codebook(R1, K2, gen2)
            c1, _ = _d2_chunked_argmin(Xp, C1)
            X1 = gather_cw(C1, c1)
            c2, _ = _d2_chunked_argmin(Xp - X1, C2)
            Xhat_p = X1 + gather_cw(C2, c2)
            degen = dg1 or dg2
            cb_bytes = P * (K1 + K2) * m * 2
        Xhat = Xhat_p.transpose(0, 1).reshape(T * nkv, hd)
        Xhat = block_denormalize(Xhat, d16, hd)
        if space == "rot":
            Xhat = Xhat @ R
    return (Xhat.numpy().reshape(T, nkv, hd).astype(np.float32),
            cb_bytes, degen, n_fit)


def export_runtime_pq(short: str, out_path: Path):
    """Export frozen raw-space PQ m=4/K=256 codebooks for the CPU reference path.

    This deliberately exports only codebooks, not codes: the runtime still encodes
    each new K vector then immediately reconstructs it into an F16 cache.  It is a
    correctness bridge to a packed-code cache kernel, not a memory result.
    """
    cfg = next(c for c in cs.MODELS if c["name"] == SHORT[short])
    Ks = cs.read_kcal_layers(ROOT / cfg["kf"])
    layers = sorted(Ks)
    m, Kcw = 4, 256
    entries = []
    for il in layers:
        K_np = Ks[il]
        T, nkv, hd = K_np.shape
        assert hd % BLOCK == 0 and hd % m == 0
        X = torch.from_numpy(np.ascontiguousarray(K_np.reshape(T * nkv, hd)))
        Xn, _ = block_normalize(X, hd)
        P = hd // m
        Xp = Xn.reshape(-1, P, m).transpose(0, 1).contiguous()
        Xfit = Xp[:, : (T // 2) * nkv].contiguous()
        gen = torch.Generator().manual_seed(stable_seed(cfg["name"], il, "raw", "pq_m4_k256", "s1"))
        C, degen, _ = fit_codebook(Xfit, Kcw, gen)
        if degen:
            raise RuntimeError(f"refusing to export degenerate PQ codebook for layer {il}")
        entries.append((il, hd, C.numpy().astype("<f4", copy=False)))
        print(f"  exported PQ codebook layer {il:2d} hd={hd}", flush=True)

    # LPQ1 v1: magic, version, m, codewords, layer count; then layer, hd, P,
    # followed by P*K*m little-endian float32 centroids.
    with out_path.open("wb") as f:
        f.write(struct.pack("<5I", 0x3151504C, 1, m, Kcw, len(entries)))
        for il, hd, C in entries:
            f.write(struct.pack("<3I", il, hd, hd // m))
            f.write(C.tobytes(order="C"))
    print(f"runtime PQ codebooks: {out_path} ({out_path.stat().st_size / 1024:.1f} KiB)")


# ------------------------------------------------------------------ metrics
def snr_stats(K_np, Khat, Tc):
    num = (K_np.astype(np.float64) ** 2).sum(-1)
    den = ((Khat.astype(np.float64) - K_np.astype(np.float64)) ** 2).sum(-1)
    snr = np.clip(10.0 * np.log10((num + 1e-30) / (den + 1e-30)), -40.0, 99.0)
    return float(snr.mean()), float(snr[Tc:].mean())


def margin_flip_stats(L_ref, L_quant, vis_e):
    """Measure top-1 flips conditional on the FP16 top-1/top-2 logit margin.

    Attention KL averages over every position, while the hypothesized TQ2
    failure mode is a rare rank reversal at small FP16 margins.  The margin
    buckets below are per-layer quantiles so they remain meaningful across
    architectures with different attention scales.
    """
    mask = vis_e[None]
    Lr = np.where(mask, L_ref, -np.inf)
    Lq = np.where(mask, L_quant, -np.inf)
    top1 = Lr.argmax(axis=-1)
    # Mask the reference winner to find the reference runner-up, then measure
    # the same winner-vs-runner margin after quantization.
    runner_src = Lr.copy()
    np.put_along_axis(runner_src, top1[..., None], -np.inf, axis=-1)
    runner = runner_src.argmax(axis=-1)
    ref_win = np.take_along_axis(Lr, top1[..., None], axis=-1)[..., 0]
    ref_run = np.take_along_axis(Lr, runner[..., None], axis=-1)[..., 0]
    margin = (ref_win - ref_run).reshape(-1)
    qtop1 = Lq.argmax(axis=-1).reshape(-1)
    flips = qtop1 != top1.reshape(-1)
    q_win = np.take_along_axis(Lq, top1[..., None], axis=-1)[..., 0]
    q_run = np.take_along_axis(Lq, runner[..., None], axis=-1)[..., 0]
    margin_error = np.abs((q_win - q_run).reshape(-1) - margin)
    low10 = margin <= np.quantile(margin, 0.10)
    low25 = margin <= np.quantile(margin, 0.25)
    high75 = margin >= np.quantile(margin, 0.75)
    return dict(
        flip_all=float(flips.mean()),
        flip_low10=float(flips[low10].mean()),
        flip_low25=float(flips[low25].mean()),
        flip_high75=float(flips[high75].mean()),
        margin_error=float(margin_error.mean()),
    )


def eval_khat_full(Khat, Q, Tc, qh_to_kv, scale, vis_e, refs):
    """Attention metrics for a fully-quantized K matrix (torch bmm logits)."""
    P_e, logP_e, top1_ref, mem_ref, _L_ref = refs
    Kbyq = Khat[:, qh_to_kv, :]                       # [T, nqh, hd]
    with torch.inference_mode():
        Qe = torch.from_numpy(np.ascontiguousarray(Q[Tc:].transpose(1, 0, 2)))
        Kt = torch.from_numpy(np.ascontiguousarray(Kbyq.transpose(1, 2, 0)))
        L = torch.bmm(Qe, Kt).numpy() * scale         # [nqh, Ne, T]
    Pd, logPd = cs.masked_softmax_logsoftmax(L, vis_e[None])
    kl = float(cs.kl_rows(P_e, logP_e, logPd).mean())
    Pd_flat = Pd.reshape(-1, Pd.shape[-1])
    top1 = float((Pd_flat.argmax(axis=-1) == top1_ref).mean())
    mem_q = cs.topk_membership(Pd_flat, TOP8)
    inter = (mem_q & mem_ref).sum(axis=-1)
    jac8 = float((inter / (2 * TOP8 - inter)).mean())
    out = dict(kl=kl, top1=top1, jac8=jac8)
    out.update(margin_flip_stats(refs[4], L, vis_e))
    return out


# ------------------------------------------------------------- layer driver
def study_layer(K, Q, il, scale_mode, swa_window, tq2, tq3, R, do_refit, mname):
    T, nkv, hd = K.shape
    Tq, nqh, hdq = Q.shape
    assert T == Tq and hd == hdq
    group = nqh // nkv
    scale = 1.0 if scale_mode == "one" else 1.0 / np.sqrt(hd)
    Tc = T // 2
    tpos = np.arange(T)
    epos = tpos[Tc:]
    qh_to_kv = np.arange(nqh) // group
    K_byq = K[:, qh_to_kv, :]

    # reference logits + softmax: numpy path, contour-identical
    L_eval = np.einsum("thd,shd->hts", Q[Tc:], K_byq, optimize=True) * scale
    vis_e = epos[:, None] >= tpos[None, :]
    if swa_window is not None and swa_window < T:
        vis_e &= (epos[:, None] - tpos[None, :]) < swa_window
    P_e, logP_e = cs.masked_softmax_logsoftmax(L_eval, vis_e[None])
    P_e_flat = P_e.reshape(-1, T)
    top1_ref = P_e_flat.argmax(axis=-1)
    mem_ref = cs.topk_membership(P_e_flat, TOP8)
    refs = (P_e, logP_e, top1_ref, mem_ref, L_eval)

    out = {}
    scalar_cfgs = ((2, tq2),) if FLIP_ONLY else ((2, tq2), (3, tq3))
    for b, tq in scalar_cfgs:
        # ---- half-K verify: replicate contour_study.study_layer exactly ----
        x = torch.from_numpy(np.ascontiguousarray(K[:Tc].transpose(1, 0, 2))).float()[None]
        with torch.no_grad():
            qkv = tq.quantize(x, layer_idx=il)
            khat = tq.dequantize(qkv, layer_idx=il, apply_inverse_rot=True)
        Khat_c = khat.squeeze(0).numpy().transpose(1, 0, 2)
        Lq = np.einsum("thd,shd->hts", Q[Tc:], Khat_c[:, qh_to_kv, :],
                       optimize=True) * scale
        Ld = L_eval.copy()
        Ld[:, :, :Tc] = Lq
        Pd, logPd = cs.masked_softmax_logsoftmax(Ld, vis_e[None])
        out[f"u{b}_half"] = dict(kl=float(cs.kl_rows(P_e, logP_e, logPd).mean()))
        # ---- full-K scalar baseline (same regime as the VQ configs) ----
        xf = torch.from_numpy(np.ascontiguousarray(K.transpose(1, 0, 2))).float()[None]
        with torch.no_grad():
            qkvf = tq.quantize(xf, layer_idx=il)
            khf = tq.dequantize(qkvf, layer_idx=il, apply_inverse_rot=True)
        Khat_f = khf.squeeze(0).numpy().transpose(1, 0, 2).astype(np.float32)
        met = eval_khat_full(Khat_f, Q, Tc, qh_to_kv, scale, vis_e, refs)
        met["snr"], met["snr_eval"] = snr_stats(K, Khat_f, Tc)
        out[f"u{b}_full"] = met

    for space in SPACES:
        for name, kind, prm, _cb in VQ_CONFIGS:
            if FLIP_ONLY and name != "pq_m4_k256":
                continue
            for tag, fit_eval in ((f"{name}/{space}", False),) + (
                    ((f"refit:{name}/{space}", True),) if do_refit else ()):
                Khat, cb_bytes, degen, n_fit = vq_roundtrip(
                    K, Tc, R, space, kind, prm,
                    (mname, il, space, name), fit_on_eval=fit_eval)
                met = eval_khat_full(Khat, Q, Tc, qh_to_kv, scale, vis_e, refs)
                met["snr"], met["snr_eval"] = snr_stats(K, Khat, Tc)
                met.update(cb_bytes=int(cb_bytes), degen=bool(degen),
                           n_fit=int(n_fit))
                out[tag] = met

    meta = dict(T=T, Tc=Tc, Ne=T - Tc, nqh=nqh, nkv=nkv, hd=hd,
                swa=swa_window if (swa_window is not None and swa_window < T) else None)
    return out, meta


# ------------------------------------------------------------- model stage
def run_model_stage(short: str, layer_slice: slice | None, do_refit: bool):
    cfg = next(c for c in cs.MODELS if c["name"] == SHORT[short])
    t0 = time.time()
    Ks = cs.read_kcal_layers(ROOT / cfg["kf"])
    Qs = cs.read_kcal_layers(ROOT / cfg["qf"])
    layers = sorted(Ks)
    assert layers == sorted(Qs)
    sel = layers if layer_slice is None else layers[layer_slice]
    max_il = max(layers) + 1

    tq_cache = {}
    for il in layers:
        hd = Ks[il].shape[2]
        for b in (2, 3):
            if (hd, b) not in tq_cache:
                tq_cache[(hd, b)] = cs.TurboQuantizer(
                    n_layers=max_il, head_dim=hd, bits=b, group_size=None,
                    rotation_strategy="randomized_hadamard", use_qjl=False,
                    seed=42, device="cpu")

    path = PARTIAL_DIR / f"vq_partial_{short}.json"
    data = json.loads(path.read_text()) if path.exists() else {
        "model": cfg["name"], "layers": {}, "meta": {}}
    print(f"== {cfg['name']} == layers {sel} (of {layers}), refit={do_refit}",
          flush=True)
    for il in sel:
        hd = Ks[il].shape[2]
        swa = None
        if cfg["swa_window"] is not None and not cfg["swa_global"](il):
            swa = cfg["swa_window"]
        R = tq_cache[(hd, 2)].rotations[il].float()
        out, meta = study_layer(Ks[il], Qs[il], il, cfg["scale"], swa,
                                tq_cache[(hd, 2)], tq_cache[(hd, 3)], R,
                                do_refit, cfg["name"])
        # A focused flip rerun augments the completed full study rather than
        # discarding its other codec metrics.
        data["layers"].setdefault(str(il), {}).update(out)
        data["meta"][str(il)] = meta
        degs = sorted({t.split("/")[0].replace("refit:", "")
                       for t, v in out.items() if v.get("degen")})
        print(f"  layer {il:2d} done (hd={meta['hd']}, swa={meta['swa']}, "
              f"{time.time()-t0:6.1f}s)"
              + (f"  DEGEN: {','.join(degs)}" if degs else ""), flush=True)
        path.write_text(json.dumps(data))
    print(f"stage done in {time.time()-t0:.1f}s -> {path}", flush=True)


# ------------------------------------------------------------- aggregation
def agg_mean(res_layers: dict, key: str, field: str):
    vals = [res_layers[il][key][field] for il in res_layers if key in res_layers[il]]
    return float(np.mean(vals)) if vals else float("nan")


def load_all():
    out = {}
    for short, name in SHORT.items():
        p = PARTIAL_DIR / f"vq_partial_{short}.json"
        d = json.loads(p.read_text())
        out[name] = d
    return out


def aggregate():
    all_d = load_all()
    tabs = {}          # model -> {cfg_key: metrics dict}
    verify = {}
    for name, d in all_d.items():
        L = d["layers"]
        keys = set()
        for il in L:
            keys.update(L[il].keys())
        tab = {}
        for k in sorted(keys):
            e = dict(kl=agg_mean(L, k, "kl"))
            for f in ("top1", "jac8", "snr", "snr_eval", "flip_all",
                      "flip_low10", "flip_low25", "flip_high75",
                      "margin_error"):
                if any(f in L[il].get(k, {}) for il in L):
                    e[f] = agg_mean(L, k, f)
            cbs = [L[il][k]["cb_bytes"] for il in L
                   if k in L[il] and "cb_bytes" in L[il][k]]
            if cbs:
                e["cb_bytes_mean"] = float(np.mean(cbs))
                e["cb_bytes_set"] = sorted(set(cbs))
                e["degen_layers"] = sum(1 for il in L
                                        if L[il].get(k, {}).get("degen"))
                e["n_fit"] = int(min(L[il][k]["n_fit"] for il in L if k in L[il]))
            tab[k] = e
        tabs[name] = tab
        qu2, qu3 = QUOTED[name]
        v2, v3 = tab["u2_half"]["kl"], tab["u3_half"]["kl"]
        verify[name] = dict(u2=v2, u3=v3,
                            ok=(round(v2, 4) == qu2 and round(v3, 4) == qu3))

    # gap closure at 2.5 bpe effective (full-K regime, consistent baselines)
    gap = {}
    for name, tab in tabs.items():
        u2, u3 = tab["u2_full"]["kl"], tab["u3_full"]["kl"]
        g = {}
        for cfgname in CODE_BITS:
            for sp in SPACES:
                k = f"{cfgname}/{sp}"
                g[k] = (u2 - tab[k]["kl"]) / (u2 - u3)
        gap[name] = g

    go_configs = {}
    for cfgname in CFG25:
        for sp in SPACES:
            k = f"{cfgname}/{sp}"
            archs = [n for n in tabs if gap[n][k] >= 0.6]
            if len(archs) >= 2:
                go_configs[k] = archs
    go = bool(go_configs)

    stretch = {}   # arch -> [cfg keys at 2.0 code bits with KL <= u2_full]
    for name, tab in tabs.items():
        u2 = tab["u2_full"]["kl"]
        stretch[name] = [f"{c}/{sp}" for c in CFG25 for sp in SPACES
                         if tab[f"{c}/{sp}"]["kl"] <= u2]

    rot_vs_raw = {}  # cfgname -> archs where raw strictly beats rot on KL
    for cfgname in CODE_BITS:
        rot_vs_raw[cfgname] = [
            n for n, tab in tabs.items()
            if tab[f"{cfgname}/raw"]["kl"] < tab[f"{cfgname}/rot"]["kl"]]

    refit_model = next((n for n, tab in tabs.items()
                        if any(k.startswith("refit:") for k in tab)), None)
    refit = {}
    if refit_model:
        tab = tabs[refit_model]
        for cfgname in CODE_BITS:
            for sp in SPACES:
                k = f"{cfgname}/{sp}"
                rk = f"refit:{k}"
                if rk in tab:
                    refit[k] = dict(calib=tab[k]["kl"], evalfit=tab[rk]["kl"],
                                    delta=tab[k]["kl"] - tab[rk]["kl"],
                                    snr_eval_calib=tab[k]["snr_eval"],
                                    snr_eval_evalfit=tab[rk]["snr_eval"])

    return all_d, tabs, verify, gap, go, go_configs, stretch, rot_vs_raw, \
        refit_model, refit


# ------------------------------------------------------------------ report
def fmt(x, p=4):
    if isinstance(x, float) and (abs(x) < 1e-3 and x != 0):
        return f"{x:.2e}"
    return f"{x:.{p}f}" if isinstance(x, float) else str(x)


ROW_ORDER = (["u2_full"]
             + [f"{c}/{sp}" for c in ("pq_m4_k256", "rvq_m4_2x16") for sp in SPACES]
             + [f"{c}/{sp}" for c in ("pq_m4_k1024", "rvq_m4_2x32") for sp in SPACES]
             + ["u3_full"]
             + [f"pq_m2_k64/{sp}" for sp in SPACES])


def row_label(k):
    if k == "u2_full":
        return ("scalar TQ2 (uniform 2b)", "rot", 2.0, 2.5)
    if k == "u3_full":
        return ("scalar TQ3 (uniform 3b)", "rot", 3.0, 3.5)
    c, sp = k.split("/")
    return (DESC[c], sp, CODE_BITS[c], EFF_BPE[c])


def write_report(all_d, tabs, verify, gap, go, go_configs, stretch,
                 rot_vs_raw, refit_model, refit):
    lines = []
    a = lines.append
    a("# VQ Study — product/residual vector quantization of the KV cache "
      "(real K/Q dumps)")
    a("")
    a(f"Generated by `kit-v2/vq_study.py` on {time.strftime('%Y-%m-%d %H:%M')}. "
      "Same three KCAL captures, GQA mapping, model-true attention scales, SWA "
      "masks and calib/eval split as `contour_study_report.md`.")
    a("")
    a("## Method")
    a("")
    a("- **Fit/eval protocol**: codebooks are k-means-fit on CALIB-half K "
      "subvectors only; the **full** K matrix used in eval attention (calib + "
      "eval positions) is then encoded/decoded with those frozen codebooks, and "
      "all metrics are on EVAL-half queries. Eval-half keys are held out from "
      "every fit.")
    a("- **Scalar baselines, two regimes**: `u2_half/u3_half` quantize only the "
      "calib half (contour-study-identical; used ONLY to verify this pipeline "
      "reproduces the quoted contour numbers). `u2_full/u3_full` quantize the "
      "full K matrix — the like-for-like baseline for the VQ configs; "
      "**gap_closure uses the full-K baselines** so numerator and denominator "
      "share one regime.")
    a("- **Spaces**: `rot` = per-layer randomized-Hadamard (TurboQuant matrices, "
      "seed 42+il), then per-32-element amax block scales stored fp16 — exactly "
      "the in-tree TQ2_0/TQ3_0 block scheme (`ggml/src/ggml-tq.c`). `raw` = "
      "identical block-scale scheme on unrotated K (VQ may exploit channel "
      "correlation that the rotation destroys).")
    a("- **Budget accounting (strict)**: effective bpe = code_bits + 0.5 "
      "(fp16 scale / 32 elems), the same ladder as scalar TQ (TQ2=2.5, TQ3=3.5). "
      "No per-token side info beyond codes + block scales. Codebooks are "
      "amortized across all cached tokens; absolute per-layer size = "
      "n_pos x codewords x m x 2 bytes is reported per config.")
    a("- **k-means**: k-means++ init, fixed seeds, >=25 Lloyd iterations "
      f"(cap {KM_MAX_ITERS}, tol {KM_TOL}), empty clusters reseeded to farthest "
      "points. **Data limitation**: calib samples per codebook position are "
      "tiny — 365 (E2B, n_kv=1), 1460 (gemma3-4b), 3008 (LFM2.5) — far below "
      "the 400k subsample cap (no subsampling was needed). Configs where "
      "codewords >= samples are flagged DEGEN (codebook memorizes every calib "
      "subvector; eval keys get nearest-calib-subvector).")
    a("- **SNR** = mean over all T x n_kv key vectors of per-vector "
      "10 log10(||k||^2/||k-k_hat||^2) on the full encoded K (clipped to "
      "[-40, 99] dB).")
    a("")
    a("## Pipeline verification (half-K scalar vs quoted contour numbers)")
    a("")
    a("| model | u2 recomputed | u2 quoted | u3 recomputed | u3 quoted | match |")
    a("|---|---|---|---|---|---|")
    for name, v in verify.items():
        qu2, qu3 = QUOTED[name]
        a(f"| {name} | {v['u2']:.4f} | {qu2:.4f} | {v['u3']:.4f} | {qu3:.4f} | "
          f"{'PASS' if v['ok'] else '**FAIL**'} |")
    a("")

    for name, tab in tabs.items():
        d = all_d[name]
        metas = d["meta"]
        il0 = sorted(metas, key=int)[0]
        m0 = metas[il0]
        hds = sorted({metas[il]["hd"] for il in metas})
        a(f"## {name}")
        a("")
        a(f"T={m0['T']}, calib={m0['Tc']}, eval rows={m0['Ne']}, "
          f"n_head={m0['nqh']}, n_kv={m0['nkv']}, head_dim={hds}, "
          f"KV layers={len(metas)}. Calib subvectors per codebook position: "
          f"{m0['Tc'] * m0['nkv']}.")
        a("")
        a("| config | space | code b/d | eff bpe | mean KL | same-top-1 | "
          "top-8 Jacc | SNR dB | codebook/layer | gap_closure |")
        a("|---|---|---|---|---|---|---|---|---|---|")
        u2, u3 = tab["u2_full"]["kl"], tab["u3_full"]["kl"]
        for k in ROW_ORDER:
            lbl, sp, cb, eff = row_label(k)
            e = tab[k]
            gc = "" if k.startswith("u") else f"{gap[name][k]:+.3f}"
            if k in ("u2_full", "u3_full"):
                gc = "0.000" if k == "u2_full" else "1.000"
            cbs = ""
            if "cb_bytes_set" in e:
                cbs = "+".join(f"{b/1024:.0f}K" for b in e["cb_bytes_set"]) \
                    if len(e["cb_bytes_set"]) > 1 else f"{e['cb_bytes_set'][0]/1024:.0f} KiB"
                if e.get("degen_layers"):
                    cbs += f" (DEGEN {e['degen_layers']}/{len(metas)}L)"
            a(f"| {lbl} | {sp} | {cb:.1f} | {eff:.1f} | {fmt(e['kl'])} | "
              f"{e['top1']*100:.1f}% | {e['jac8']:.3f} | {e['snr']:.2f} | "
              f"{cbs or '—'} | {gc} |")
        a("")
        a(f"Half-K contour baselines for reference: u2={tab['u2_half']['kl']:.4f}, "
          f"u3={tab['u3_half']['kl']:.4f} (different regime — eval-half keys F16).")
        a("")

    a("## Cross-arch summary (full-K regime)")
    a("")
    a("| model | KL u2 (2.5) | KL u3 (3.5) | best 2.5-bpe VQ | its KL | "
      "gap_closure | best VQ SNR-u2 SNR | stretch (KL<=u2 @2.0 code b) |")
    a("|---|---|---|---|---|---|---|---|")
    for name, tab in tabs.items():
        u2, u3 = tab["u2_full"]["kl"], tab["u3_full"]["kl"]
        best = min((tab[f"{c}/{sp}"]["kl"], f"{c}/{sp}")
                   for c in CFG25 for sp in SPACES)
        bk, bcfg = best
        dsnr = tab[bcfg]["snr"] - tab["u2_full"]["snr"]
        a(f"| {name} | {fmt(u2)} | {fmt(u3)} | {bcfg} | {fmt(bk)} | "
              f"{gap[name][bcfg]:+.3f} | {dsnr:+.2f} dB | "
              f"{', '.join(stretch[name]) or 'none'} |")
    a("")

    a("## Margin-conditioned attention flips (full-K, open-loop)")
    a("")
    a("Reference margins are the FP16 winner-minus-runner-up logits per "
      "query/head/layer. `low 10%` and `low 25%` are per-layer margin "
      "quantiles; `high 75%` is the safest quarter. This directly tests "
      "whether VQ reduces the rank reversals that scalar TQ2 can compound "
      "through depth. It is still open-loop: quantized attention is not fed "
      "into later-layer K/Q generation.")
    a("")
    a("| model | codec | all flips | low 10% flips | low 25% flips | high 75% flips | mean margin error |")
    a("|---|---|---|---|---|---|---|")
    for name, tab in tabs.items():
        for k, label in (("u2_full", "scalar TQ2"),
                         ("pq_m4_k256/rot", "PQ m4/256 rot"),
                         ("pq_m4_k256/raw", "PQ m4/256 raw")):
            e = tab[k]
            a(f"| {name} | {label} | {e['flip_all']*100:.2f}% | "
              f"{e['flip_low10']*100:.2f}% | {e['flip_low25']*100:.2f}% | "
              f"{e['flip_high75']*100:.2f}% | {e['margin_error']:.4f} |")
    a("")

    a("## Gate verdicts")
    a("")
    a(f"### Kernel-work gate: **{'GO' if go else 'NO-GO'}**")
    a("")
    a("Gate: some VQ config at 2.5 bpe effective (code 2.0 + 0.5 scale) reaches "
      "gap_closure = (KL_u2 - KL_cfg)/(KL_u2 - KL_u3) >= 0.6 on >= 2 of 3 archs "
      "(full-K regime, consistent baselines).")
    if go:
        for k, archs in go_configs.items():
            a(f"- `{k}`: gap_closure >= 0.6 on {', '.join(archs)} "
              f"({'; '.join(f'{n}: {gap[n][k]:+.3f}' for n in tabs)})")
    else:
        a("- No 2.5-bpe VQ config reaches 0.6 on 2+ archs. Best per config:")
        for c in CFG25:
            for sp in SPACES:
                k = f"{c}/{sp}"
                a(f"  - `{k}`: " + "; ".join(f"{n} {gap[n][k]:+.3f}" for n in tabs))
    a("")
    n_stretch = sum(1 for n in stretch if stretch[n])
    a(f"### Stretch (2.0-code-bit VQ with KL <= scalar TQ2): "
      f"**{'MET on ' + str(n_stretch) + '/3 archs' if n_stretch else 'NOT MET'}**")
    a("")
    a("Note: under the strict accounting both sides are 2.5 bpe effective "
      "(+0.5 block scales each), so this is an equal-memory quality result; the "
      "'memory win' reading holds only if VQ block scales could later be "
      "coarsened or shared.")
    for name in tabs:
        a(f"- {name}: {', '.join(stretch[name]) if stretch[name] else 'none'}")
    a("")
    any_raw = {c: ms for c, ms in rot_vs_raw.items() if ms}
    a("### Rot vs raw (does exploiting channel correlation beat the rotated "
      "space?)")
    a("")
    if any_raw:
        for c, ms in any_raw.items():
            deltas = "; ".join(
                f"{n}: {tabs[n][f'{c}/rot']['kl']:.4f}->"
                f"{tabs[n][f'{c}/raw']['kl']:.4f}" for n in ms)
            a(f"- `{c}`: raw beats rot on {', '.join(ms)} ({deltas})")
    else:
        a("- Nowhere: the rotated space is at least as good for every config "
          "on every arch.")
    a("")
    if refit_model:
        a(f"### Codebook generalization ({refit_model}: refit on EVAL half)")
        a("")
        a("KL with calib-fit codebooks vs codebooks refit on the eval half "
          "(both encode the full K; metrics on eval queries). delta = "
          "calib-fit KL - eval-fit KL (positive = calib codebooks lose that "
          "much KL to distribution shift/overfitting).")
        a("")
        a("| config | calib-fit KL | eval-fit KL | delta | delta % | "
          "SNR_eval calib | SNR_eval evalfit |")
        a("|---|---|---|---|---|---|---|")
        for k, r in refit.items():
            pct = 100.0 * r["delta"] / r["calib"] if r["calib"] else 0.0
            a(f"| {k} | {fmt(r['calib'])} | {fmt(r['evalfit'])} | "
              f"{fmt(r['delta'])} | {pct:+.1f}% | {r['snr_eval_calib']:.2f} | "
              f"{r['snr_eval_evalfit']:.2f} |")
        a("")
    REPORT.write_text("\n".join(lines))


def print_summary(tabs, verify, gap, go, go_configs, stretch, rot_vs_raw,
                  refit_model, refit):
    print("\n================ VQ STUDY — CROSS-ARCH SUMMARY (full-K regime) "
          "================")
    print("| model | KL u2(2.5) | KL u3(3.5) | best 2.5-bpe VQ | its KL | "
          "gap_closure | verify half-K |")
    for name, tab in tabs.items():
        best = min((tab[f"{c}/{sp}"]["kl"], f"{c}/{sp}")
                   for c in CFG25 for sp in SPACES)
        bk, bcfg = best
        print(f"| {name} | {tab['u2_full']['kl']:.4f} | "
              f"{tab['u3_full']['kl']:.4f} | {bcfg} | {bk:.4f} | "
              f"{gap[name][bcfg]:+.3f} | "
              f"{'PASS' if verify[name]['ok'] else 'FAIL'} |")
    print("\nGap closure at 2.5 bpe effective (per config):")
    for c in CFG25:
        for sp in SPACES:
            k = f"{c}/{sp}"
            print(f"  {k:22s} " + "  ".join(
                f"{n}: {gap[n][k]:+.3f}" for n in tabs))
    print(f"\nGATE (kernel work)  : {'GO' if go else 'NO-GO'}"
          + (f" — {list(go_configs)}" if go else ""))
    n_stretch = sum(1 for n in stretch if stretch[n])
    print(f"STRETCH (2.0 code b): "
          + (f"MET on {n_stretch}/3 archs — "
             + "; ".join(f"{n}: {stretch[n]}" for n in tabs if stretch[n])
             if n_stretch else "NOT MET"))
    any_raw = {c: ms for c, ms in rot_vs_raw.items() if ms}
    print("ROT-vs-RAW          : "
          + (("raw wins somewhere: " + "; ".join(f"{c} on {ms}"
                                                 for c, ms in any_raw.items()))
             if any_raw else "rotated space never loses"))
    if refit_model:
        worst = max(refit.items(), key=lambda kv: kv[1]["delta"])
        print(f"REFIT check ({refit_model}): worst calib-fit loss "
              f"{worst[1]['delta']:+.4f} KL ({worst[0]}); "
              f"mean {np.mean([r['delta'] for r in refit.values()]):+.4f}")
    print(f"\nreport: {REPORT}")


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(SHORT))
    ap.add_argument("--layers", default=None,
                    help="index slice a:b into the sorted layer list")
    ap.add_argument("--refit", action="store_true",
                    help="also refit codebooks on the eval half (one model)")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--export-runtime-pq", metavar="PATH",
                    help="export raw PQ m=4/256 codebooks for the CPU runtime reference codec")
    args = ap.parse_args()
    if args.aggregate:
        res = aggregate()
        write_report(*res)
        print_summary(*res[1:])
        return
    if not args.model:
        ap.error("need --model or --aggregate")
    if args.export_runtime_pq:
        export_runtime_pq(args.model, Path(args.export_runtime_pq))
        return
    sl = None
    if args.layers:
        a_, b_ = args.layers.split(":")
        sl = slice(int(a_), int(b_))
    run_model_stage(args.model, sl, args.refit)


if __name__ == "__main__":
    main()
