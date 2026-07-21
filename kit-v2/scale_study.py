#!/usr/bin/env python3
"""scale_study.py — offline study of the BLOCK SCALE STATISTIC for TQ KV quant.

Question: the shipping codec (ggml/src/ggml-tq.c) uses per-32-element amax
(`d = max|block|`) with Lloyd-Max levels normalized so the outer levels are
+/-1.  amax is an extreme-order statistic and is therefore outlier-hijacked:
one large element inside a 32-element block stretches d and wastes the
(already tiny) 4/8/16-level codebook on empty range.  Does a ROBUST scale
statistic (absmean, as in BitNet b1.58; rms; or the per-block MSE optimum)
beat amax at 2-4 bits on real attention?

Everything else is held fixed against the shipping codec:
  * same in-tree Lloyd-Max level tables (TQ2_LEVELS/TQ3_LEVELS/TQ4_LEVELS,
    transcribed from ggml/src/ggml-tq.c),
  * same per-32-element block granularity (except the explicit per-vector rung),
  * same fp16 storage of d (indices chosen with the fp32 d, reconstruction with
    the fp16 d — exactly what quantize_row_tq*_0_ref does),
  * same randomized-Hadamard pre-rotation the runtime applies to the K cache
    (TurboQuant matrices, seed 42+il) — "rot" space.

Harness reuse (see kit-v2/vq_study.py, kit-v2/contour_study.py for provenance):
KCAL K+Q dumps, GQA mapping, model-true attention scales, SWA masks, calib/eval
split (calib = first T//2 positions), softmax-KL / same-top-1 / top-8 Jaccard on
EVAL-half queries with the FULL K matrix quantized ("u*_full" regime).

Scale schemes (Part A)
  amax      d = max|x|                                (shipping baseline)
  absmean   d = c * mean|x|          c swept on a grid
  rms       d = c * sqrt(mean x^2)   c swept on a grid
  mse_opt   d = argmin_d ||x - Q_d(x)||^2 per block (grid search)  -- upper
            bound on any fixed statistic at this granularity
  pervec    d = c * sqrt(mean x^2) over the WHOLE head_dim vector (one scale
            per vector; side info 16/head_dim bpe, not 0.5)
  intree3   the ACTUAL shipping 3-bit path: amax init -> least-squares optimal
            scale -> 2 coordinate-descent passes (rule 4: baseline must be the
            shipping codec, and TQ3_0 is NOT plain amax)

Ternary rung (Part B)
  levels {-1, 0, +1} at per-32 granularity, zero bucket |x| <= t*d.
  tern_absmean  d = c*mean|x|, threshold t*d, (c,t) swept
  tern_mse      exact per-block optimum: sort |x|, for each nonzero count m the
                optimal d is mean of the top-m magnitudes and
                SSE = sum x^2 - m*d^2; take the best m.  (Given levels
                {-d,0,+d} this is the exact joint (d, threshold) optimum.)
  Rate: log2(3) = 1.585 b/elem, packed 5 trits/byte = 1.6 b/elem, + 0.5 for the
  fp16 per-32 scale = 2.1 bpe effective (vs TQ2's 2.5).

Effective bpe accounting (rule 3/4): reported per row.
  per-32 fp16 scale = 0.5 bpe;  per-vector fp16 scale = 16/head_dim bpe.

Calibration discipline: the c / (c,t) constants are the only fitted quantities.
They are selected on CALIB-half key blocks by minimizing reconstruction SSE, and
the selected constant is then applied to the full K matrix; all reported KL /
top-1 / Jaccard are on EVAL-half queries, and `snr_eval` is over EVAL-half keys
only (rule 2).  The full grid is tabulated so the oracle-best-KL constant can be
compared against the calib-selected one.

Run:
  VENV=/home/junc/LeanKV/.venv/bin/python3
  $VENV kit-v2/scale_study.py --model e2b
  $VENV kit-v2/scale_study.py --model lfm2
  $VENV kit-v2/scale_study.py --model gemma3            # (or --layers 0:12 etc)
  $VENV kit-v2/scale_study.py --aggregate
Output: docs/leankv-scale-scheme-study-2026-07.md (Part A/B tables) + stdout.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import contour_study as cs  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(8)
torch.set_num_interop_threads(1)

ROOT = cs.ROOT
PARTIAL_DIR = Path("/tmp/claude-1000/-home-junc-rikuri-rikurinode/"
                   "0a7b0d58-b4e2-4aa3-bb15-3e35bc56022f/scratchpad")
PARTIAL_DIR.mkdir(parents=True, exist_ok=True)

BLOCK = 32
TOP8 = 8
SHORT = {"e2b": "gemma4-E2B", "lfm2": "LFM2.5-1.2B", "gemma3": "gemma3-4b"}

# ── in-tree Lloyd-Max levels, transcribed from ggml/src/ggml-tq.c ────────────
TQ_LEVELS = {
    2: np.array([-1.0000000, -0.2997714, +0.2997714, +1.0000000], dtype=np.float64),
    3: np.array([-1.0000000, -0.6245203, -0.3513239, -0.1138989,
                 +0.1138989, +0.3513239, +0.6245203, +1.0000000], dtype=np.float64),
    4: np.array([-1.0000000, -0.7573038, -0.5923403, -0.4599576,
                 -0.3450764, -0.2405254, -0.1421261, -0.0470277,
                 +0.0470277, +0.1421261, +0.2405254, +0.3450764,
                 +0.4599576, +0.5923403, +0.7573038, +1.0000000], dtype=np.float64),
}
TQ_BOUNDS = {b: ((L[:-1] + L[1:]) / 2.0) for b, L in TQ_LEVELS.items()}
# sanity: the midpoints must reproduce the in-tree *_BOUNDARIES tables
assert np.allclose(TQ_BOUNDS[2], [-0.6498857, 0.0, +0.6498857], atol=1e-6)
assert np.allclose(TQ_BOUNDS[3][0], -0.8122602, atol=1e-6)
assert np.allclose(TQ_BOUNDS[4][0], -0.8786519, atol=1e-6)

# ── scheme grids ────────────────────────────────────────────────────────────
# For 32 iid N(0,1) samples: E[amax] ~ 2.20 sigma, E[mean|x|] = 0.798 sigma, so
# amax corresponds to c ~ 2.76 (absmean) / ~2.20 (rms).  Grids bracket that.
ABSMEAN_GRID = tuple(round(1.2 + 0.2 * i, 2) for i in range(13))   # 1.2 .. 3.6
RMS_GRID     = tuple(round(0.8 + 0.2 * i, 2) for i in range(13))   # 0.8 .. 3.2
PERVEC_GRID  = tuple(round(0.8 + 0.2 * i, 2) for i in range(13))   # 0.8 .. 3.2
MSEOPT_TS    = np.linspace(0.12, 1.04, 93)      # d = t * amax candidates
TERN_C_GRID  = tuple(round(0.8 + 0.3 * i, 2) for i in range(8))    # 0.8 .. 2.9
TERN_T_GRID  = tuple(round(0.30 + 0.10 * i, 2) for i in range(8))  # 0.30 .. 1.00

# smaller grids for the TQ3 "init scheme + in-tree refinement" variants (each
# one re-runs the coordinate descent, so the sweep is deliberately narrow)
INIT_RMS_GRID     = (1.4, 1.8, 2.0, 2.2, 2.6)
INIT_ABSMEAN_GRID = (1.8, 2.2, 2.6, 3.0)

BITS = (2, 3, 4)
GRID_SCHEMES = {"absmean": ABSMEAN_GRID, "rms": RMS_GRID, "pervec": PERVEC_GRID}
FLAT_SCHEMES = ("amax", "mse_opt")

TERNARY_BPE = 1.6 + 0.5      # 5 trits/byte packing + fp16 per-32 scale


def cfg_keys(hd: int):
    """All config keys evaluated for a layer with this head_dim."""
    keys = []
    for b in BITS:
        for s in FLAT_SCHEMES:
            keys.append(f"{s}_b{b}")
        for s, grid in GRID_SCHEMES.items():
            for c in grid:
                keys.append(f"{s}@{c}_b{b}")
        if b == 3:
            keys.append("intree3_b3")
            keys.append("intree3init@mse_opt_b3")
            for c in INIT_RMS_GRID:
                keys.append(f"intree3init@rms:{c}_b3")
            for c in INIT_ABSMEAN_GRID:
                keys.append(f"intree3init@absmean:{c}_b3")
    keys.append("tern_mse")
    for c in TERN_C_GRID:
        for t in TERN_T_GRID:
            keys.append(f"tern_absmean@{c}/{t}")
    return keys


def eff_bpe(key: str, hd: int) -> float:
    if key.startswith("tern"):
        return TERNARY_BPE
    b = int(key.rsplit("_b", 1)[1])
    if key.startswith("pervec"):
        return b + 16.0 / hd
    return b + 0.5


# ── quantization primitives (torch, float32 data / float64 search) ──────────
def _fp16(t: torch.Tensor) -> torch.Tensor:
    return t.to(torch.float16).to(torch.float32)


def _apply_scale(B: torch.Tensor, d: torch.Tensor, bits: int) -> torch.Tensor:
    """B [.., n] blocks, d [..] fp32 scale -> reconstruction, in-tree semantics:
    indices picked with the fp32 d, reconstruction with the stored fp16 d."""
    L = torch.tensor(TQ_LEVELS[bits], dtype=torch.float32)
    Bnd = torch.tensor(TQ_BOUNDS[bits], dtype=torch.float32)
    d = d.clamp_min(0.0)
    ok = d > 1e-10
    id_ = torch.where(ok, 1.0 / d.clamp_min(1e-30), torch.zeros_like(d))
    xn = (B * id_.unsqueeze(-1)).clamp(-1.0, 1.0)
    idx = torch.bucketize(xn, Bnd)
    d16 = _fp16(d)
    out = L[idx] * d16.unsqueeze(-1)
    return torch.where(ok.unsqueeze(-1), out, torch.zeros_like(out))


def scale_of(B: torch.Tensor, scheme: str, c: float | None) -> torch.Tensor:
    """B [.., n] -> d [..]"""
    if scheme == "amax":
        return B.abs().amax(-1)
    if scheme == "absmean":
        return c * B.abs().mean(-1)
    if scheme in ("rms", "pervec"):
        return c * B.pow(2).mean(-1).sqrt()
    raise ValueError(scheme)


def scale_mse_opt(B: torch.Tensor, bits: int) -> torch.Tensor:
    """Per-block d minimizing ||x - Q_d(x)||^2, grid-searched as d = t * amax."""
    amax = B.abs().amax(-1)
    best_d, best_sse = None, None
    for t in MSEOPT_TS:
        d = amax * float(t)
        sse = (B - _apply_scale(B, d, bits)).pow(2).sum(-1)
        if best_d is None:
            best_d, best_sse = d, sse
        else:
            m = sse < best_sse
            best_d = torch.where(m, d, best_d)
            best_sse = torch.where(m, sse, best_sse)
    return best_d


def quant_blocks(B: torch.Tensor, bits: int, scheme: str, c: float | None):
    """B [N, nb, 32] (or [N, 1, hd] for pervec) -> reconstruction, same shape."""
    if scheme == "mse_opt":
        return _apply_scale(B, scale_mse_opt(B, bits), bits)
    return _apply_scale(B, scale_of(B, scheme, c), bits)


def quant_intree3(B: torch.Tensor, d0: torch.Tensor | None = None) -> torch.Tensor:
    """Exact replication of quantize_row_tq3_0_ref (ggml/src/ggml-tq.c):
    amax-normalized nearest-level init, least-squares optimal scale, then 2
    coordinate-descent passes over the 32 positions trying idx+-1.

    Note MSE(indices, d_opt) = sum x^2 - num^2/den with d_opt = num/den, so
    'does this move reduce MSE' is exactly 'does it increase num^2/den'.  That
    makes the descent vectorizable across blocks while remaining bit-exact in
    its accept/reject decisions."""
    L = torch.tensor(TQ_LEVELS[3], dtype=torch.float32)
    Bnd = torch.tensor(TQ_BOUNDS[3], dtype=torch.float32)
    n = B.shape[-1]
    amax = B.abs().amax(-1)
    ok = amax > 0
    # d0 is the INITIAL assignment scale.  None = shipping behaviour (amax);
    # anything else models `LEANKV_TQ_SCALE=<scheme>` feeding the same
    # least-squares + coordinate-descent refinement.
    init = amax if d0 is None else torch.where(d0 > 0, d0, amax)
    id_ = torch.where(ok, 1.0 / init.clamp_min(1e-30), torch.zeros_like(amax))
    xn = (B * id_.unsqueeze(-1)).clamp(-1.0, 1.0)
    idx = torch.bucketize(xn, Bnd)                              # [N, nb, 32]
    lev = L[idx]
    num = (B * lev).sum(-1)
    den = (lev * lev).sum(-1)
    obj = num.pow(2) / den.clamp_min(1e-30)                     # maximize

    for _ in range(2):
        improved = torch.zeros_like(num, dtype=torch.bool)
        for j in range(n):
            xj = B[..., j]
            lj = L[idx[..., j]]
            for delta in (-1, +1):
                cand = idx[..., j] + delta
                valid = (cand >= 0) & (cand <= 7)
                cl = L[cand.clamp(0, 7)]
                nnum = num - xj * lj + xj * cl
                nden = den - lj * lj + cl * cl
                nobj = nnum.pow(2) / nden.clamp_min(1e-30)
                take = valid & (nobj > obj)
                idx[..., j] = torch.where(take, cand.clamp(0, 7), idx[..., j])
                num = torch.where(take, nnum, num)
                den = torch.where(take, nden, den)
                obj = torch.where(take, nobj, obj)
                lj = torch.where(take, cl, lj)
                improved |= take
        if not bool(improved.any()):
            break

    d = num / den.clamp_min(1e-30)
    d16 = _fp16(d)
    out = L[idx] * d16.unsqueeze(-1)
    return torch.where(ok.unsqueeze(-1), out, torch.zeros_like(out))


def quant_ternary(B: torch.Tensor, mode: str, c: float | None, t: float | None):
    """levels {-1,0,+1} * d, zero bucket |x| <= t*d."""
    if mode == "absmean":
        d = c * B.abs().mean(-1)
        d16 = _fp16(d)
        a = B.abs()
        nz = a > (t * d).unsqueeze(-1)
        return torch.where(nz, torch.sign(B) * d16.unsqueeze(-1),
                           torch.zeros_like(B))
    # exact per-block optimum over (d, threshold) jointly
    a = B.abs()
    srt, _ = torch.sort(a, dim=-1, descending=True)
    cs_ = torch.cumsum(srt, dim=-1)                     # [.., n], m=1..n
    n = B.shape[-1]
    ms = torch.arange(1, n + 1, dtype=torch.float32)
    dm = cs_ / ms                                       # optimal d for count m
    sse = -ms * dm.pow(2)                               # sum x^2 is constant
    zero_sse = torch.zeros_like(sse[..., :1])           # m = 0
    allsse = torch.cat([zero_sse, sse], dim=-1)
    mbest = allsse.argmin(-1)                           # [..]
    dbest = torch.where(mbest > 0,
                        torch.gather(dm, -1, (mbest - 1).clamp_min(0).unsqueeze(-1)).squeeze(-1),
                        torch.zeros_like(cs_[..., 0]))
    thr = torch.where(mbest > 0,
                      torch.gather(srt, -1, (mbest - 1).clamp_min(0).unsqueeze(-1)).squeeze(-1),
                      torch.full_like(dbest, float("inf")))
    d16 = _fp16(dbest)
    nz = a >= thr.unsqueeze(-1)
    nz &= (mbest > 0).unsqueeze(-1)
    return torch.where(nz, torch.sign(B) * d16.unsqueeze(-1), torch.zeros_like(B))


def reconstruct(Xr: torch.Tensor, hd: int, key: str) -> torch.Tensor:
    """Xr [N, hd] rotated K rows -> reconstruction [N, hd]."""
    if key == "tern_mse":
        B = Xr.reshape(-1, hd // BLOCK, BLOCK)
        return quant_ternary(B, "mse", None, None).reshape(-1, hd)
    if key.startswith("tern_absmean@"):
        c, t = key.split("@")[1].split("/")
        B = Xr.reshape(-1, hd // BLOCK, BLOCK)
        return quant_ternary(B, "absmean", float(c), float(t)).reshape(-1, hd)
    head, bs = key.rsplit("_b", 1)
    bits = int(bs)
    if head == "intree3":
        B = Xr.reshape(-1, hd // BLOCK, BLOCK)
        return quant_intree3(B).reshape(-1, hd)
    if head.startswith("intree3init@"):
        # what `LEANKV_TQ_SCALE=<scheme>` actually does at tq3_0: swap the
        # initial assignment scale, keep the in-tree LS + coordinate descent.
        spec = head.split("@", 1)[1]
        B = Xr.reshape(-1, hd // BLOCK, BLOCK)
        if spec == "mse_opt":
            d0 = scale_mse_opt(B, 3)
        else:
            s, c = spec.split(":")
            d0 = scale_of(B, s, float(c))
        return quant_intree3(B, d0).reshape(-1, hd)
    if "@" in head:
        scheme, c = head.split("@")
        c = float(c)
    else:
        scheme, c = head, None
    if scheme == "pervec":
        B = Xr.reshape(-1, 1, hd)
        return quant_blocks(B, bits, "pervec", c).reshape(-1, hd)
    B = Xr.reshape(-1, hd // BLOCK, BLOCK)
    return quant_blocks(B, bits, scheme, c).reshape(-1, hd)


# ── metrics ─────────────────────────────────────────────────────────────────
def snr_stats(K_np, Khat, Tc):
    num = (K_np.astype(np.float64) ** 2).sum(-1)
    den = ((Khat.astype(np.float64) - K_np.astype(np.float64)) ** 2).sum(-1)
    snr = np.clip(10.0 * np.log10((num + 1e-30) / (den + 1e-30)), -40.0, 99.0)
    return float(snr.mean()), float(snr[Tc:].mean())


def eval_khat(Khat, Q, Tc, qh_to_kv, scale, vis_e, refs):
    P_e, logP_e, top1_ref, mem_ref = refs
    Kbyq = Khat[:, qh_to_kv, :]
    with torch.inference_mode():
        Qe = torch.from_numpy(np.ascontiguousarray(Q[Tc:].transpose(1, 0, 2)))
        Kt = torch.from_numpy(np.ascontiguousarray(Kbyq.transpose(1, 2, 0)))
        L = torch.bmm(Qe, Kt).numpy() * scale
    Pd, logPd = cs.masked_softmax_logsoftmax(L, vis_e[None])
    kl = float(cs.kl_rows(P_e, logP_e, logPd).mean())
    Pd_flat = Pd.reshape(-1, Pd.shape[-1])
    top1 = float((Pd_flat.argmax(axis=-1) == top1_ref).mean())
    mem_q = cs.topk_membership(Pd_flat, TOP8)
    inter = (mem_q & mem_ref).sum(axis=-1)
    jac8 = float((inter / (2 * TOP8 - inter)).mean())
    return dict(kl=kl, top1=top1, jac8=jac8)


# ── per-layer driver ────────────────────────────────────────────────────────
def study_layer(K, Q, il, scale_mode, swa_window, R):
    T, nkv, hd = K.shape
    _, nqh, _ = Q.shape
    group = nqh // nkv
    scale = 1.0 if scale_mode == "one" else 1.0 / np.sqrt(hd)
    Tc = T // 2
    tpos = np.arange(T)
    epos = tpos[Tc:]
    qh_to_kv = np.arange(nqh) // group
    K_byq = K[:, qh_to_kv, :]

    L_eval = np.einsum("thd,shd->hts", Q[Tc:], K_byq, optimize=True) * scale
    vis_e = epos[:, None] >= tpos[None, :]
    if swa_window is not None and swa_window < T:
        vis_e &= (epos[:, None] - tpos[None, :]) < swa_window
    P_e, logP_e = cs.masked_softmax_logsoftmax(L_eval, vis_e[None])
    P_e_flat = P_e.reshape(-1, T)
    refs = (P_e, logP_e, P_e_flat.argmax(axis=-1), cs.topk_membership(P_e_flat, TOP8))

    with torch.inference_mode():
        X = torch.from_numpy(np.ascontiguousarray(K.reshape(T * nkv, hd)))
        Xr = X @ R.T
        Xr_c = Xr[: Tc * nkv]

    out = {}
    for key in cfg_keys(hd):
        with torch.inference_mode():
            Xhat_r = reconstruct(Xr, hd, key)
            # calib-half reconstruction SSE: the ONLY quantity used to pick the
            # fitted constants c / (c,t).  Eval-half keys never inform the choice.
            sse_c = float((Xhat_r[: Tc * nkv] - Xr_c).pow(2).sum())
            Xhat = (Xhat_r @ R).numpy().reshape(T, nkv, hd).astype(np.float32)
        met = eval_khat(Xhat, Q, Tc, qh_to_kv, scale, vis_e, refs)
        met["snr"], met["snr_eval"] = snr_stats(K, Xhat, Tc)
        met["sse_calib"] = sse_c
        out[key] = met

    meta = dict(T=T, Tc=Tc, Ne=T - Tc, nqh=nqh, nkv=nkv, hd=hd,
                n_blocks=T * nkv * (hd // BLOCK), n_vectors=T * nkv,
                n_elems=T * nkv * hd,
                swa=swa_window if (swa_window is not None and swa_window < T) else None)
    return out, meta


def run_model_stage(short: str, layer_slice: slice | None):
    cfg = next(c for c in cs.MODELS if c["name"] == SHORT[short])
    t0 = time.time()
    Ks = cs.read_kcal_layers(ROOT / cfg["kf"])
    Qs = cs.read_kcal_layers(ROOT / cfg["qf"])
    layers = sorted(Ks)
    sel = layers if layer_slice is None else layers[layer_slice]
    max_il = max(layers) + 1

    rot = {}
    for il in layers:
        hd = Ks[il].shape[2]
        if hd not in rot:
            rot[hd] = cs.TurboQuantizer(
                n_layers=max_il, head_dim=hd, bits=2, group_size=None,
                rotation_strategy="randomized_hadamard", use_qjl=False,
                seed=42, device="cpu").rotations

    path = PARTIAL_DIR / f"scale_partial_{short}.json"
    data = json.loads(path.read_text()) if path.exists() else {
        "model": cfg["name"], "layers": {}, "meta": {}}
    print(f"== {cfg['name']} == layers {sel}", flush=True)
    for il in sel:
        hd = Ks[il].shape[2]
        swa = None
        if cfg["swa_window"] is not None and not cfg["swa_global"](il):
            swa = cfg["swa_window"]
        R = rot[hd][il].float()
        out, meta = study_layer(Ks[il], Qs[il], il, cfg["scale"], swa, R)
        data["layers"].setdefault(str(il), {}).update(out)
        data["meta"][str(il)] = meta
        print(f"  layer {il:2d} done (hd={hd}, swa={meta['swa']}, "
              f"{time.time()-t0:6.1f}s)", flush=True)
        path.write_text(json.dumps(data))
    print(f"stage done in {time.time()-t0:.1f}s -> {path}", flush=True)


# ── aggregation ─────────────────────────────────────────────────────────────
def agg(L, key, field):
    vals = [L[il][key][field] for il in L if key in L[il]]
    return float(np.mean(vals)) if vals else float("nan")


def agg_sum(L, key, field):
    vals = [L[il][key][field] for il in L if key in L[il]]
    return float(np.sum(vals)) if vals else float("nan")


def build_tab(d):
    L = d["layers"]
    keys = set()
    for il in L:
        keys.update(L[il].keys())
    tab = {}
    for k in sorted(keys):
        tab[k] = dict(kl=agg(L, k, "kl"), top1=agg(L, k, "top1"),
                      jac8=agg(L, k, "jac8"), snr=agg(L, k, "snr"),
                      snr_eval=agg(L, k, "snr_eval"),
                      sse_calib=agg_sum(L, k, "sse_calib"),
                      n_layers=sum(1 for il in L if k in L[il]))
    return tab


def select_const(tab, scheme, bits, grid):
    """Calib-SSE-selected constant (the honest pick) and KL-oracle constant."""
    keys = [f"{scheme}@{c}_b{bits}" for c in grid if f"{scheme}@{c}_b{bits}" in tab]
    if not keys:
        return None, None
    best_sse = min(keys, key=lambda k: tab[k]["sse_calib"])
    best_kl = min(keys, key=lambda k: tab[k]["kl"])
    return best_sse, best_kl


def select_tern(tab):
    keys = [k for k in tab if k.startswith("tern_absmean@")]
    if not keys:
        return None, None
    return (min(keys, key=lambda k: tab[k]["sse_calib"]),
            min(keys, key=lambda k: tab[k]["kl"]))


def load_all():
    out = {}
    for short, name in SHORT.items():
        p = PARTIAL_DIR / f"scale_partial_{short}.json"
        if p.exists():
            out[name] = json.loads(p.read_text())
    return out


def aggregate():
    all_d = load_all()
    tabs = {n: build_tab(d) for n, d in all_d.items()}
    picks = {}
    for n, tab in tabs.items():
        p = {}
        for b in BITS:
            for s, grid in GRID_SCHEMES.items():
                p[(s, b)] = select_const(tab, s, b, grid)
        p[("tern_absmean", 0)] = select_tern(tab)
        for s, grid in (("rms", INIT_RMS_GRID), ("absmean", INIT_ABSMEAN_GRID)):
            ks = [f"intree3init@{s}:{c}_b3" for c in grid
                  if f"intree3init@{s}:{c}_b3" in tab]
            p[(f"intree3init_{s}", 3)] = (
                (min(ks, key=lambda k: tab[k]["sse_calib"]),
                 min(ks, key=lambda k: tab[k]["kl"])) if ks else (None, None))
        picks[n] = p
    return all_d, tabs, picks


# ── report ──────────────────────────────────────────────────────────────────
DOC = ROOT / "docs" / "leankv-scale-scheme-study-2026-07.md"


def label(key):
    if key == "tern_mse":
        return "ternary mse_opt (exact)"
    if key.startswith("tern_absmean@"):
        c, t = key.split("@")[1].split("/")
        return f"ternary absmean c={c}, t={t}"
    head, b = key.rsplit("_b", 1)
    if head == "intree3":
        return "**in-tree TQ3_0** (amax init -> LS scale -> CD)"
    if head.startswith("intree3init@"):
        return ("in-tree TQ3_0 refine, init="
                + head.split("@", 1)[1].replace(":", " c="))
    if "@" in head:
        s, c = head.split("@")
        return f"{s} c={c}"
    return {"amax": "**amax (shipping)**", "mse_opt": "mse_opt (per-block optimum)"}[head]


def write_report(all_d, tabs, picks, extra=""):
    a = []
    w = a.append
    w("# KV-cache scale-scheme study — is `d = amax` the wrong block scale?")
    w("")
    w(f"Generated by `kit-v2/scale_study.py` on {time.strftime('%Y-%m-%d %H:%M')}. "
      "Obeys the five methodology rules in `docs/leankv-vq-study-2026-07.md`.")
    w("")
    w("## Method")
    w("")
    w("Everything except the block scale statistic is held identical to the "
      "shipping codec (`ggml/src/ggml-tq.c`): the same Lloyd-Max level tables "
      "`TQ2/3/4_LEVELS`, the same per-32-element blocks, the same fp16 storage "
      "of `d` (indices chosen with the fp32 `d`, reconstruction with the fp16 "
      "`d`), and the same randomized-Hadamard pre-rotation the runtime applies "
      "to K (TurboQuant matrices, seed 42+il).")
    w("")
    w("Data: the three KCAL K/Q captures used by `contour_study.py` / "
      "`vq_study.py`; GQA mapping, model-true attention scales, SWA masks and "
      "the calib/eval split (calib = first `T//2` positions) are unchanged. The "
      "FULL K matrix is quantized (the `u*_full` regime); all KL / same-top-1 / "
      "top-8 Jaccard numbers are over EVAL-half queries, and `SNR eval` is over "
      "EVAL-half key vectors only (rule 2).")
    w("")
    w("The constants `c` (absmean / rms / per-vector) and `(c, t)` (ternary) are "
      "the only fitted quantities. They are chosen by minimizing CALIB-half "
      "reconstruction SSE and then applied to the whole matrix; the KL-oracle "
      "choice is tabulated beside them so the fitting cost is visible.")
    w("")
    w("Effective bpe = code bits + scale side info: per-32 fp16 scale = 0.5 bpe, "
      "per-vector fp16 scale = 16/head_dim bpe (0.0625 at head_dim 256, 0.25 at "
      "head_dim 64). Ternary = 1.6 b/elem (5 trits per byte) + 0.5 = 2.1 bpe.")
    w("")

    for name, tab in tabs.items():
        metas = all_d[name]["meta"]
        il0 = sorted(metas, key=int)[0]
        m0 = metas[il0]
        hds = sorted({metas[il]["hd"] for il in metas})
        nblk = sum(metas[il]["n_blocks"] for il in metas)
        nvec = sum(metas[il]["n_vectors"] for il in metas)
        nel = sum(metas[il]["n_elems"] for il in metas)
        w(f"## {name}")
        w("")
        w(f"T={m0['T']}, calib={m0['Tc']}, eval rows={m0['Ne']}, n_head="
          f"{m0['nqh']}, n_kv={m0['nkv']}, head_dim={hds}, KV layers="
          f"{len(metas)}. Sample counts across those layers: **{nel:,} key "
          f"elements**, **{nblk:,} 32-element blocks**, **{nvec:,} key "
          f"vectors**; eval-query rows scored per layer = "
          f"{m0['Ne']} x {m0['nqh']}.")
        w("")
        for b in BITS:
            w(f"### {b}-bit")
            w("")
            w("| scheme | eff bpe | mean KL | same-top-1 | top-8 Jacc | "
              "SNR all dB | SNR eval dB | vs amax KL |")
            w("|---|---|---|---|---|---|---|---|")
            base = tab[f"amax_b{b}"]["kl"]
            rows = [f"amax_b{b}"]
            for s in ("absmean", "rms", "pervec"):
                sse_k, kl_k = picks[name][(s, b)]
                if sse_k:
                    rows.append(sse_k)
                    if kl_k != sse_k:
                        rows.append(kl_k + "  (KL-oracle)")
            rows.append(f"mse_opt_b{b}")
            if b == 3:
                rows.append("intree3_b3")
                rows.append("intree3init@mse_opt_b3")
                for s in ("rms", "absmean"):
                    k2, _ = picks[name][(f"intree3init_{s}", 3)]
                    if k2:
                        rows.append(k2)
            hd0 = hds[0]
            for r in rows:
                oracle = r.endswith("(KL-oracle)")
                k = r.split("  ")[0]
                e = tab[k]
                lbl = label(k) + (" *(KL-oracle c)*" if oracle else "")
                dl = 100.0 * (e["kl"] - base) / base
                w(f"| {lbl} | {eff_bpe(k, hd0):.4f} | {e['kl']:.4f} | "
                  f"{e['top1']*100:.2f}% | {e['jac8']:.3f} | {e['snr']:.2f} | "
                  f"{e['snr_eval']:.2f} | {dl:+.1f}% |")
            w("")
        # full grids
        w("<details><summary>full constant sweeps (this model)</summary>")
        w("")
        w("| scheme | bits | c | eff bpe | mean KL | SNR eval dB | calib SSE |")
        w("|---|---|---|---|---|---|---|")
        for b in BITS:
            for s, grid in GRID_SCHEMES.items():
                for c in grid:
                    k = f"{s}@{c}_b{b}"
                    if k not in tab:
                        continue
                    e = tab[k]
                    w(f"| {s} | {b} | {c} | {eff_bpe(k, hds[0]):.4f} | "
                      f"{e['kl']:.4f} | {e['snr_eval']:.2f} | {e['sse_calib']:.4g} |")
        w("")
        w("</details>")
        w("")

        # Part B
        w(f"### Ternary rung ({name})")
        w("")
        sse_k, kl_k = picks[name][("tern_absmean", 0)]
        w("| scheme | eff bpe | mean KL | same-top-1 | top-8 Jacc | SNR all dB | "
          "SNR eval dB |")
        w("|---|---|---|---|---|---|---|")
        for k in [x for x in (sse_k, kl_k if kl_k != sse_k else None, "tern_mse")
                  if x]:
            e = tab[k]
            w(f"| {label(k)} | {TERNARY_BPE:.2f} | {e['kl']:.4f} | "
              f"{e['top1']*100:.2f}% | {e['jac8']:.3f} | {e['snr']:.2f} | "
              f"{e['snr_eval']:.2f} |")
        e2 = tab["amax_b2"]
        w(f"| TQ2 amax (reference) | 2.50 | {e2['kl']:.4f} | "
          f"{e2['top1']*100:.2f}% | {e2['jac8']:.3f} | {e2['snr']:.2f} | "
          f"{e2['snr_eval']:.2f} |")
        bm = min((f"absmean@{c}_b2" for c in ABSMEAN_GRID
                  if f"absmean@{c}_b2" in tab),
                 key=lambda k: tab[k]["sse_calib"], default=None)
        if bm:
            e3 = tab[bm]
            w(f"| TQ2 {label(bm)} (reference) | 2.50 | {e3['kl']:.4f} | "
              f"{e3['top1']*100:.2f}% | {e3['jac8']:.3f} | {e3['snr']:.2f} | "
              f"{e3['snr_eval']:.2f} |")
        w("")
        # Does ternary's zero bucket buy ranking quality that MSE alone would
        # not predict?  Calibrate log(KL) against SNR on the mse_opt rung at
        # 2/3/4 bits (same levels family, same block granularity, scale chosen
        # the same way), then ask where ternary lands relative to that line.
        xs = np.array([tab[f"mse_opt_b{b}"]["snr_eval"] for b in BITS])
        ys = np.log(np.array([tab[f"mse_opt_b{b}"]["kl"] for b in BITS]))
        slope, icpt = np.polyfit(xs, ys, 1)
        w("")
        w("**Does ternary beat its own SNR deficit on ranking?** Calibrating "
          "`log KL` against held-out SNR on the mse_opt rung at 2/3/4 bits "
          f"(fit: log KL = {slope:+.4f}*SNR {icpt:+.4f}, i.e. "
          f"{-8.686*slope:.2f} dB per e-fold of KL) and reading the ternary "
          "points off that line:")
        w("")
        w("| ternary variant | SNR eval dB | vs TQ2 mse_opt SNR | KL predicted "
          "from SNR | KL actual | actual/predicted |")
        w("|---|---|---|---|---|---|")
        ref_snr = tab["mse_opt_b2"]["snr_eval"]
        for k in [x for x in (sse_k, "tern_mse") if x]:
            e = tab[k]
            pred = float(np.exp(slope * e["snr_eval"] + icpt))
            w(f"| {label(k)} | {e['snr_eval']:.2f} | "
              f"{e['snr_eval']-ref_snr:+.2f} dB | {pred:.4f} | {e['kl']:.4f} | "
              f"{e['kl']/pred:.3f}x |")
        w("")
        w("(`actual/predicted` < 1 would mean the zero bucket preserves "
          "attention ranking better than its reconstruction error implies; "
          "> 1 means it is worse than even its SNR says.)")
        w("")
        w("<details><summary>full ternary (c, t) sweep</summary>")
        w("")
        w("| c | t | mean KL | SNR eval dB | calib SSE |")
        w("|---|---|---|---|---|")
        for c in TERN_C_GRID:
            for t in TERN_T_GRID:
                k = f"tern_absmean@{c}/{t}"
                if k in tab:
                    e = tab[k]
                    w(f"| {c} | {t} | {e['kl']:.4f} | {e['snr_eval']:.2f} | "
                      f"{e['sse_calib']:.4g} |")
        w("")
        w("</details>")
        w("")

    # cross-model
    w("## Cross-model Part A summary (calib-SSE-selected constants)")
    w("")
    w("Baseline is the **shipping codec** (rule 4): `amax` at 2 and 4 bits, and "
      "the full in-tree TQ3_0 path (amax init -> least-squares scale -> "
      "coordinate descent) at 3 bits.")
    w("")
    w("| bits | model | shipping KL | best alternative | its KL | KL delta | "
      "shipping SNR eval | best SNR eval |")
    w("|---|---|---|---|---|---|---|---|")
    for b in BITS:
        ship = "intree3_b3" if b == 3 else f"amax_b{b}"
        for name, tab in tabs.items():
            cands = [f"amax_b{b}", f"mse_opt_b{b}"]
            for s in ("absmean", "rms", "pervec"):
                k, _ = picks[name][(s, b)]
                if k:
                    cands.append(k)
            if b == 3:
                cands += ["intree3_b3", "intree3init@mse_opt_b3"]
                for s in ("rms", "absmean"):
                    k2, _ = picks[name][(f"intree3init_{s}", 3)]
                    if k2:
                        cands.append(k2)
            best = min((c for c in cands if c != ship), key=lambda k: tab[k]["kl"])
            base = tab[ship]
            e = tab[best]
            w(f"| {b} | {name} | {base['kl']:.4f} ({label(ship)}) | "
              f"{label(best)} | {e['kl']:.4f} | "
              f"{100*(e['kl']-base['kl'])/base['kl']:+.1f}% | "
              f"{base['snr_eval']:.2f} | {e['snr_eval']:.2f} |")
    w("")
    if extra:
        w(extra)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text("\n".join(a))
    print(f"report -> {DOC}")


def print_summary(tabs, picks):
    for b in BITS:
        print(f"\n===== {b}-bit =====")
        for name, tab in tabs.items():
            base = tab[f"amax_b{b}"]
            line = [f"{name:14s} amax KL {base['kl']:.4f} "
                    f"(snr_e {base['snr_eval']:.2f})"]
            for s in ("absmean", "rms", "pervec"):
                k, ko = picks[name][(s, b)]
                if k:
                    line.append(f"{k.split('_b')[0]:14s} {tab[k]['kl']:.4f}")
            line.append(f"mse_opt {tab[f'mse_opt_b{b}']['kl']:.4f}")
            if b == 3 and "intree3_b3" in tab:
                line.append(f"intree3 {tab['intree3_b3']['kl']:.4f}")
            print("  " + " | ".join(line))
    print("\n===== ternary =====")
    for name, tab in tabs.items():
        k, ko = picks[name][("tern_absmean", 0)]
        print(f"  {name:14s} tern_mse {tab['tern_mse']['kl']:.4f} "
              f"(snr_e {tab['tern_mse']['snr_eval']:.2f}) | best absmean {k} "
              f"{tab[k]['kl']:.4f} | TQ2 amax {tab['amax_b2']['kl']:.4f} "
              f"(snr_e {tab['amax_b2']['snr_eval']:.2f})")


def part_c_section() -> str:
    """Part C table, parsed from kld_scale_status.txt (run_scale_closedloop.sh)."""
    st = ROOT / "kld_scale_status.txt"
    if not st.exists():
        return ""
    import re
    rows = []
    for line in st.read_text().splitlines():
        m = re.match(r"^(\S+): Mean\s+KLD:\s+([\d.]+) .. ([\d.]+) \| "
                     r"Same top p:\s+([\d.]+) .. ([\d.]+) % \| KV: (.*?) \d\d:", line)
        if m:
            rows.append(dict(tag=m.group(1), kld=float(m.group(2)),
                             kld_err=float(m.group(3)), top=float(m.group(4)),
                             top_err=float(m.group(5)), kv=m.group(6).strip()))
    if not rows:
        return ""
    by = {r["tag"]: r for r in rows}
    a = ["## Part C — closed-loop validation (llama-perplexity KLD)", "",
         "Real runtime, real KV cache, error compounding through layers — the "
         "measurement that overturned the VQ study's offline verdict (rule 1). "
         "Canonical `wiki.test.raw`, `-c 2048`, KLD against the F16 bases "
         "already on disk (`base_f16_e2b.kld`, `base_f16_lfm2.kld`). Every arm "
         "re-runs its OWN amax control on the same binary, so the comparison is "
         "within-build.", "",
         "| arm | scale | Mean KLD | Same top-1 | KV size | KLD vs amax |",
         "|---|---|---|---|---|---|"]
    seen = set()
    for r in rows:
        base_tag = r["tag"].rsplit("-", 1)[0] + "-amax"
        rel = ""
        if base_tag in by and r["tag"] != base_tag:
            b = by[base_tag]["kld"]
            rel = f"{100*(r['kld']-b)/b:+.1f}%"
        arm, sc = r["tag"].rsplit("-", 1)
        a.append(f"| {arm} | {sc} | {r['kld']:.6f} ± {r['kld_err']:.6f} | "
                 f"{r['top']:.3f} ± {r['top_err']:.3f} % | {r['kv']} | {rel} |")
        seen.add(r["tag"])
    a.append("")
    return "\n".join(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(SHORT))
    ap.add_argument("--layers", default=None)
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()
    if args.aggregate:
        all_d, tabs, picks = aggregate()
        write_report(all_d, tabs, picks, extra=part_c_section())
        print_summary(tabs, picks)
        return
    if not args.model:
        ap.error("need --model or --aggregate")
    sl = None
    if args.layers:
        a_, b_ = args.layers.split(":")
        sl = slice(int(a_), int(b_))
    run_model_stage(args.model, sl)


if __name__ == "__main__":
    main()
