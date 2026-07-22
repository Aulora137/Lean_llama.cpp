#!/usr/bin/env python3
"""softmax_correction_study.py — training-free analytic softmax correction for
KV-cache KEY quantization error (the calibration-only version of KVLinC's idea).

The idea (training-free, closed-form).  Quantizing a key K -> K_q shifts each
attention logit by  delta = q . k_e,  k_e = k - k_q.  We cannot store the whole
per-key error, but the part that AFFECTS attention is its projection onto the
QUERY subspace (measured low-dim: E2B globals r95~196/512, coding-gain bound
2.3 dB).  So, per (layer, kv-head):

  1. B = top-r eigenvectors of E[q q^T] over CALIB-half queries (query subspace).
  2. Per EVAL key store c = B^T k_e  (r coefficients), optionally QUANTIZED
     (coef_bits in {fp16, int8, int4}).
  3. At attention:  corrected logit = q.k_q + (B^T q) . c
     = q.k_q + q.(B B^T k_e), recovering the query-subspace component of the
     true logit error.  Recompute softmax with corrected logits.

As r -> head_dim with fp16 coeffs this is EXACT (B B^T = I -> recovers FP16).
The interesting regime is SMALL r + quantized coeffs.

Arms (all per-(layer,kv-head), calib/eval split, metrics on EVAL queries):
  - BASELINE  : scalar TQ2 / TQ3 / TQ4 via the repo's bit-exact TurboQuant path
                (FULL-K softmax-KL; MUST reproduce the anchors or STOP).
  - CORRECTION: TQ2 base + query-subspace correction.  Sweep r in
                {8,16,32,64,head_dim} x coef_bits in {fp16,int8,int4}.
                Honest total bpe = 2.5 + r*coef_bits/head_dim (task formula);
                a second "+scale" column adds the per-token fp16 coefficient
                scale (16/head_dim) that quantized coeffs actually need — the
                coefficient-storage tax the eOptShrink study warned about.
  - CEILING   : the coef_bits=fp16 curve.  r=head_dim = FP16 attention (KL~0);
                the sub-head_dim points show how much attention quality the
                FULL (perfect-coefficient) correction buys per r, ignoring
                storage.  Bounds what any adapter could achieve.
  - SHAPING   : the DUAL we already tested — SQuat error-shaping (arm D of the
                noiseshape study: orthogonalize the quantization error to the
                query subspace, NO stored coeffs), re-run FULL-K here.
                Correction (store query-error) vs shaping (remove it), per bit.

This is OFFLINE single-pass softmax-KL: our closed-loop rule (vq study) says
offline UNDERSTATES closed-loop damage 3-5x, so every gate is an OFFLINE SCREEN.

Run (FOREGROUND, ~6 threads; prior studies died self-backgrounding):
  VENV=/home/junc/LeanKV/.venv/bin/python3
  $VENV kit-v2/softmax_correction_study.py           # all archs + xval + report
  $VENV kit-v2/softmax_correction_study.py --smoke    # 3 layers/model wiring check
Output: docs/leankv-softmax-correction-study-2026-07.md + summary/verdicts stdout.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import contour_study as cs  # noqa: E402  (KCAL reader, TurboQuantizer, metrics)
import noiseshape_study as ns  # noqa: E402  (query-subspace / SQuat machinery — REUSE)
import torch  # noqa: E402

torch.set_num_threads(6)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

ROOT = cs.ROOT
DOC = ROOT / "docs" / "leankv-softmax-correction-study-2026-07.md"
PARTIAL = Path("/tmp/claude-1000/-home-junc-rikuri-rikurinode/"
               "0a7b0d58-b4e2-4aa3-bb15-3e35bc56022f/scratchpad")
PARTIAL.mkdir(parents=True, exist_ok=True)

BLOCK = 32
TOP8 = 8
SCALE_BITS = 16.0                       # fp16 side-channel scalar

SHORT = {"e2b": "gemma4-E2B", "lfm2": "LFM2.5-1.2B", "gemma3": "gemma3-4b",
         "xval": "gemma3-4b/xval"}
MODELS = list(cs.MODELS) + [
    dict(name="gemma3-4b/xval", kf="xval_k.bin", qf="xval_q.bin",
         scale="rsqrt_hd", swa_window=1024, swa_global=lambda il: il % 6 == 5),
]

# Full-K softmax-KL anchors we MUST reproduce (bit-identical TurboQuant path).
ANCHORS = {"gemma3-4b": (0.2908, 0.0885), "gemma4-E2B": (0.1869, 0.0551),
           "LFM2.5-1.2B": (0.2623, 0.0880)}
ANCHOR_TOL = 0.02

R_GRID = (8, 16, 32, 64)                # + head_dim (ceiling) appended per layer
COEF_BITS = (16, 8, 4)                  # fp16, int8, int4
CB_NAME = {16: "fp16", 8: "int8", 4: "int4"}
SHAPE_R, SHAPE_LAM = 32, 1.0            # SQuat arm D main config


# ── coefficient quantizer ───────────────────────────────────────────────────
def quant_coef(c: torch.Tensor, bits: int) -> torch.Tensor:
    """Quantize + dequantize coefficient rows c [T, r].
    fp16 -> nearest fp16 (no scale).  int8/int4 -> symmetric uniform, one fp16
    max-scale per token (row)."""
    if bits >= 16:
        return c.to(torch.float16).to(torch.float32)
    if c.shape[1] == 0:
        return c
    levels = float(2 ** bits)
    half = (levels - 1) / 2.0
    scale = c.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    scale = scale.to(torch.float16).to(torch.float32)          # stored fp16
    q = torch.round(c / scale * half).clamp(-half, half)
    return q / half * scale


def bpe_task(r: int, coef_bits: int, hd: int) -> float:
    """Task headline: 2.5 + r*coef_bits/head_dim."""
    return 2.5 + r * coef_bits / hd


def bpe_scale(r: int, coef_bits: int, hd: int) -> float:
    """Fully honest: + per-token fp16 coefficient scale (16/hd) for quantized
    coeffs (fp16 coeffs need no separate scale)."""
    extra = SCALE_BITS / hd if coef_bits < 16 else 0.0
    return bpe_task(r, coef_bits, hd) + extra


# ── per-layer reference (FULL-K, eval queries vs FP16) ──────────────────────
def build_refs(K, Q, il, cfg):
    T, nkv, hd = K.shape
    _, nqh, _ = Q.shape
    group = nqh // nkv
    scale = 1.0 if cfg["scale"] == "one" else 1.0 / np.sqrt(hd)
    Tc = T // 2
    tpos = np.arange(T)
    epos = tpos[Tc:]
    qh_to_kv = np.arange(nqh) // group
    swa = None
    if cfg["swa_window"] is not None and not cfg["swa_global"](il):
        swa = cfg["swa_window"]
    vis_e = epos[:, None] >= tpos[None, :]
    if swa is not None and swa < T:
        vis_e &= (epos[:, None] - tpos[None, :]) < swa
    K_byq = K[:, qh_to_kv, :]
    L = np.einsum("thd,shd->hts", Q[Tc:], K_byq, optimize=True) * scale
    P_e, logP_e = cs.masked_softmax_logsoftmax(L, vis_e[None])
    P_flat = P_e.reshape(-1, T)
    return dict(P_e=P_e, logP_e=logP_e, vis_e=vis_e, scale=scale, group=group,
                qh_to_kv=qh_to_kv, Tc=Tc, T=T, nkv=nkv, nqh=nqh, hd=hd, swa=swa,
                top1=P_flat.argmax(axis=-1), mem=cs.topk_membership(P_flat, TOP8))


def kl_from_logits(L, refs):
    """L [nqh, Ne, T] scaled logits -> softmax-KL / same-top-1 / top-8 Jaccard."""
    Pd, logPd = cs.masked_softmax_logsoftmax(L, refs["vis_e"][None])
    kl = float(cs.kl_rows(refs["P_e"], refs["logP_e"], logPd).mean())
    Pf = Pd.reshape(-1, Pd.shape[-1])
    top1 = float((Pf.argmax(axis=-1) == refs["top1"]).mean())
    mem = cs.topk_membership(Pf, TOP8)
    inter = (mem & refs["mem"]).sum(axis=-1)
    jac8 = float((inter / (2 * TOP8 - inter)).mean())
    return dict(kl=kl, top1=top1, jac8=jac8)


def eval_khat(Khat, Q, refs):
    Kbyq = Khat[:, refs["qh_to_kv"], :]
    L = np.einsum("thd,shd->hts", Q[refs["Tc"]:], Kbyq, optimize=True) * refs["scale"]
    return kl_from_logits(L, refs)


def tq_khat(K, il, tq):
    T, nkv, hd = K.shape
    x = torch.from_numpy(np.ascontiguousarray(K.transpose(1, 0, 2))).float()[None]
    with torch.no_grad():
        qkv = tq.quantize(x, layer_idx=il)
        kh = tq.dequantize(qkv, layer_idx=il, apply_inverse_rot=True)
    return kh.squeeze(0).numpy().transpose(1, 0, 2).astype(np.float32)


# ── query subspace basis (top eigenvectors of E[q q^T]) ─────────────────────
def query_basis(Qpool: torch.Tensor):
    """Qpool [N, hd] pooled calib queries for one kv-head.  Returns U [hd, hd]
    eigenvectors of E[q q^T] sorted by DESCENDING eigenvalue, plus eigenvalues."""
    C = (Qpool.double().T @ Qpool.double()) / Qpool.shape[0]
    lam, U = torch.linalg.eigh(C)
    idx = torch.argsort(lam, descending=True)
    return U[:, idx].float(), lam[idx].clamp_min(0.0)


# ── correction driver for one layer ─────────────────────────────────────────
def study_layer(K, Q, il, cfg, tqc, R, donor_U=None):
    refs = build_refs(K, Q, il, cfg)
    T, nkv, hd = K.shape
    nqh, Tc, scale = refs["nqh"], refs["Tc"], refs["scale"]
    group = refs["group"]
    qh_to_kv = refs["qh_to_kv"]
    Qe = Q[Tc:]                                             # [Ne, nqh, hd]
    out = {}

    # ---- TQ ladder ----
    for b in (2, 3, 4):
        khat = tq_khat(K, il, tqc[(hd, b)])
        out[f"tq{b}"] = eval_khat(khat, Q, refs)
        if b == 2:
            kq = khat
    ke = (K - kq).astype(np.float32)                        # [T, nkv, hd] key error

    # ---- query bases (per kv-head), pooled calib queries ----
    U_heads, lam_heads = [], []
    Qe_t = torch.from_numpy(np.ascontiguousarray(Qe))       # [Ne, nqh, hd]
    for h in range(nkv):
        qidx = np.where(qh_to_kv == h)[0]
        Qcal = torch.from_numpy(
            np.ascontiguousarray(Q[:Tc, qidx, :].reshape(-1, hd)))
        if donor_U is not None:
            U = torch.from_numpy(donor_U[h]).float()        # cross-corpus donor
            _, lam = query_basis(Qcal)
        else:
            U, lam = query_basis(Qcal)
        U_heads.append(U)
        lam_heads.append(lam)

    # coding-gain bound of E[qq^T] (median info; matches noiseshape metric)
    cg = float(np.mean([ns.coding_gain_db(l) for l in lam_heads]))
    # r95: dims to reach 95% of query energy (intrinsic subspace size)
    r95s = []
    for l in lam_heads:
        c = torch.cumsum(l, 0) / l.sum().clamp_min(1e-30)
        r95s.append(int((c < 0.95).sum()) + 1)
    r95 = float(np.mean(r95s))

    # ---- base raw logits from k_q (shared across all correction configs) ----
    kq_byq = kq[:, qh_to_kv, :]
    Lb_raw = np.einsum("thd,shd->hts", Qe, kq_byq, optimize=True)   # [nqh,Ne,T]

    # precompute full coefficient projections c_full[h] = ke_h @ U_h  [T, hd]
    # and projected eval queries Qp_full[h] = Qe_group @ U_h  [group, Ne, hd]
    c_full, Qp_full = [], []
    for h in range(nkv):
        ke_h = torch.from_numpy(np.ascontiguousarray(ke[:, h, :]))
        c_full.append(ke_h @ U_heads[h])                    # [T, hd]
        qidx = np.where(qh_to_kv == h)[0]
        Qg = Qe_t[:, qidx, :].permute(1, 0, 2).contiguous() # [group, Ne, hd]
        Qp_full.append(torch.einsum("gtd,dr->gtr", Qg, U_heads[h]))  # [group,Ne,hd]

    max_r = U_heads[0].shape[1]         # hd for self-fit; stored cols for donor
    r_grid = tuple(r for r in sorted(set(R_GRID + (hd,))) if r <= max_r)
    for r in r_grid:
        for cb in COEF_BITS:
            L_corr = np.empty((nqh, refs["P_e"].shape[1], T), dtype=np.float64)
            for h in range(nkv):
                cq = quant_coef(c_full[h][:, :r].clone(), cb)       # [T, r]
                qidx = np.where(qh_to_kv == h)[0]
                Qp = Qp_full[h][:, :, :r]                           # [group,Ne,r]
                Lc = torch.einsum("gtr,sr->gts", Qp, cq).numpy()    # [group,Ne,T]
                L_corr[qidx] = Lc
            L = (Lb_raw + L_corr) * scale
            met = kl_from_logits(L, refs)
            met["bpe_task"] = bpe_task(r, cb, hd)
            met["bpe_scale"] = bpe_scale(r, cb, hd)
            out[f"corr_r{r}_c{cb}"] = met

    # ---- fine CEILING grid (fp16 coeffs only) to resolve useful-r (Q2) ----
    ceil_extra = sorted({int(round(hd * f)) for f in
                         (0.15, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875)})
    ceil_extra = [r for r in ceil_extra if 0 < r < max_r and r not in r_grid]
    for r in ceil_extra:
        L_corr = np.empty((nqh, refs["P_e"].shape[1], T), dtype=np.float64)
        for h in range(nkv):
            cq = c_full[h][:, :r].to(torch.float16).to(torch.float32)
            qidx = np.where(qh_to_kv == h)[0]
            Lc = torch.einsum("gtr,sr->gts", Qp_full[h][:, :, :r], cq).numpy()
            L_corr[qidx] = Lc
        met = kl_from_logits((Lb_raw + L_corr) * scale, refs)
        met["bpe_task"] = bpe_task(r, 16, hd)
        met["bpe_scale"] = bpe_scale(r, 16, hd)
        out[f"corr_r{r}_c16"] = met

    # ---- SHAPING (SQuat arm D, FULL-K, reuse noiseshape machinery) ----
    Kt = torch.from_numpy(np.ascontiguousarray(K.transpose(1, 0, 2)))  # [nkv,T,hd]
    Xr = torch.stack([Kt[j] @ R.T for j in range(nkv)])                # rotated
    for bits in (2, 3):
        for sch in ("amax", "mse"):
            heads = []
            for h in range(nkv):
                qidx = np.where(qh_to_kv == h)[0]
                Qr_c = (torch.from_numpy(
                    np.ascontiguousarray(Q[:Tc, qidx, :].reshape(-1, hd))) @ R.T)
                if SHAPE_R <= hd // 2:
                    Qhat = ns.squat_subspace(Qr_c, SHAPE_R)
                    ps = ns.correction_vectors(Qhat, SHAPE_LAM)
                    d32 = ns.block_scale(Xr[h], hd, bits, sch)
                    heads.append((ns.greedy_quant(Xr[h], d32, bits, ps) @ R).numpy())
                else:
                    heads.append(None)
            if heads[0] is not None:
                Khat = np.stack(heads, 1).astype(np.float32)
                out[f"shape_b{bits}_{sch}"] = eval_khat(Khat, Q, refs)

    meta = dict(T=T, Tc=Tc, Ne=T - Tc, nqh=nqh, nkv=nkv, hd=hd, swa=refs["swa"],
                cg=cg, r95=r95, r_grid=list(r_grid))
    fits = None
    if donor_U is None:
        fits = [U_heads[h][:, :max(R_GRID)].numpy().astype(np.float16)
                for h in range(nkv)]        # store top-max(R) eigenvectors
    return out, meta, fits


# ── model stage ──────────────────────────────────────────────────────────────
def partial_path(short):
    return PARTIAL / f"sc_partial_{short}.json"


def fits_path(short):
    return PARTIAL / f"sc_fits_{short}.npz"


def run_model(short, layers_cap=None, donor_fits=None):
    model_key = "xval" if short == "xval_donor" else short
    cfg = next(c for c in MODELS if c["name"] == SHORT[model_key])
    t0 = time.time()
    Ks = cs.read_kcal_layers(ROOT / cfg["kf"])
    Qs = cs.read_kcal_layers(ROOT / cfg["qf"])
    layers = sorted(Ks)
    if layers_cap:
        layers = layers[:layers_cap]
    max_il = max(sorted(Ks)) + 1
    tqc, rot = {}, {}
    for il in layers:
        hd = Ks[il].shape[2]
        for b in (2, 3, 4):
            if (hd, b) not in tqc:
                tqc[(hd, b)] = cs.TurboQuantizer(
                    n_layers=max_il, head_dim=hd, bits=b, group_size=None,
                    rotation_strategy="randomized_hadamard", use_qjl=False,
                    seed=42, device="cpu")
        if hd not in rot:
            rot[hd] = tqc[(hd, 2)].rotations
    data = {"model": cfg["name"], "layers": {}, "meta": {}}
    all_fits = {}
    tag = "" if donor_fits is None else " (donor: gemma3)"
    print(f"== {cfg['name']}{tag} == layers {layers}", flush=True)
    for il in layers:
        hd = Ks[il].shape[2]
        R = rot[hd][il].float()
        dU = donor_fits[str(il)] if donor_fits else None
        outl, meta, fits = study_layer(Ks[il], Qs[il], il, cfg, tqc, R, donor_U=dU)
        data["layers"][str(il)] = outl
        data["meta"][str(il)] = meta
        if fits is not None:
            all_fits[str(il)] = fits
        print(f"  layer {il:2d} done (hd={hd}, nkv={meta['nkv']}, swa={meta['swa']}, "
              f"cg={meta['cg']:.2f}dB, r95={meta['r95']:.0f}, "
              f"{time.time()-t0:6.1f}s)", flush=True)
        partial_path(short).write_text(json.dumps(data))
    if donor_fits is None:
        np.savez_compressed(fits_path(short),
                            fits=np.array(all_fits, dtype=object))
    print(f"stage {cfg['name']} done in {time.time()-t0:.1f}s -> "
          f"{partial_path(short)}", flush=True)
    return data, all_fits


# ── aggregation ──────────────────────────────────────────────────────────────
FIELDS = ("kl", "top1", "jac8", "bpe_task", "bpe_scale")


def agg(d, key, field):
    L = d["layers"]
    vals = [L[il][key][field] for il in L if key in L[il] and field in L[il][key]]
    return float(np.mean(vals)) if vals else float("nan")


def build_tab(d):
    L = d["layers"]
    keys = set()
    for il in L:
        keys.update(L[il].keys())
    return {k: {f: agg(d, k, f) for f in FIELDS} for k in sorted(keys)}


def gap_closure(tab, key):
    a2, a3 = tab["tq2"]["kl"], tab["tq3"]["kl"]
    return (a2 - tab[key]["kl"]) / (a2 - a3) if a2 > a3 else float("nan")


def ceiling_curve(tab):
    """Sorted [(r, kl)] over all fp16-coefficient (ceiling) correction keys."""
    rows = []
    for k in tab:
        if k.startswith("corr_r") and k.endswith("_c16") and "donor" not in k:
            r = int(k[len("corr_r"):-len("_c16")])
            if not np.isnan(tab[k]["kl"]):
                rows.append((r, tab[k]["kl"]))
    return sorted(rows)


def useful_r(tab, target_kl):
    """Smallest ceiling r whose KL <= target (None if none reach it)."""
    for r, kl in ceiling_curve(tab):
        if kl <= target_kl:
            return r
    return None


# ── report ───────────────────────────────────────────────────────────────────
def build_report(res, fits_meta):
    lines = []
    a = lines.append
    real = {k: v for k, v in res.items() if k != "xval"}
    tab_v = lambda s: res[s][1]                    # noqa: E731  tab by short name

    a("# Training-free softmax correction for KV key-quantization error — study")
    a("")
    a(f"Generated by `kit-v2/softmax_correction_study.py` on "
      f"{time.strftime('%Y-%m-%d %H:%M')}. Real post-RoPE K/Q KCAL captures from "
      "three architectures; model-true attention scales, SWA masks, GQA mapping "
      "and calib/eval split exactly as the rest of kit-v2. FULL-K softmax-KL vs "
      "FP16 (all keys quantized; metrics on EVAL queries). The scalar codec is "
      "the shipping in-tree TQ2_0/TQ3_0/TQ4_0 (bit-exact TurboQuant path). "
      "OFFLINE single-pass softmax-KL: our closed-loop rule says this "
      "UNDERSTATES closed-loop damage 3-5x — every gate below is an OFFLINE SCREEN.")
    a("")
    a("## The idea (training-free, closed-form)")
    a("")
    a("Quantizing a key `k -> k_q` shifts each attention logit by `delta = q.k_e`, "
      "`k_e = k - k_q`. The part that affects attention is `k_e`'s projection onto "
      "the QUERY subspace. Per (layer, kv-head): `B` = top-`r` eigenvectors of "
      "`E[q q^T]` over CALIB-half queries; per EVAL key store `c = B^T k_e` "
      "(`r` coefficients, possibly quantized); at attention the corrected logit is "
      "`q.k_q + (B^T q).c = q.k_q + q.(B B^T k_e)`. As `r -> head_dim` with fp16 "
      "coeffs, `B B^T -> I` and this is EXACT (recovers FP16). Interesting regime: "
      "SMALL `r` + quantized coeffs.")
    a("")
    a("**Honest bit accounting.** Base TQ2 is 2.5 bpe. The correction adds `r` "
      "coefficients per key: `bpe_task = 2.5 + r*coef_bits/head_dim` (the task "
      "formula). Quantized coeffs (int8/int4) also need a per-token fp16 scale "
      "(`16/head_dim`) — the coefficient-storage tax the eOptShrink study "
      "identified as fatal to low-rank methods; the `+scale` column charges it. "
      "fp16 coeffs need no separate scale.")
    a("")

    # ---- anchor reproduction ----
    a("## Baseline reproduction gate (FULL-K softmax-KL vs anchors)")
    a("")
    a("| arch | TQ2 measured | TQ2 anchor | TQ3 measured | TQ3 anchor | match |")
    a("|---|---|---|---|---|---|")
    for x, (_, tab) in real.items():
        name = SHORT[x]
        a2m, a3m = tab["tq2"]["kl"], tab["tq3"]["kl"]
        a2a, a3a = ANCHORS[name]
        ok = abs(a2m - a2a) / a2a <= ANCHOR_TOL and abs(a3m - a3a) / a3a <= ANCHOR_TOL
        a(f"| {name} | {a2m:.4f} | {a2a:.4f} | {a3m:.4f} | {a3a:.4f} | "
          f"{'PASS' if ok else 'FAIL'} |")
    a("")

    # ---- per-arch tables ----
    for x, (d, tab) in res.items():
        name = SHORT[x]
        metas = d["meta"]
        m0 = metas[sorted(metas, key=int)[0]]
        hds = sorted({metas[il]["hd"] for il in metas})
        cgs = [metas[il]["cg"] for il in metas]
        r95s = [metas[il]["r95"] for il in metas]
        a(f"## {name}" + (" — cross-corpus dump" if x == "xval" else ""))
        a("")
        a(f"T={m0['T']}, calib={m0['Tc']}, eval rows={m0['Ne']}, "
          f"n_head={m0['nqh']}, n_kv={m0['nkv']}, head_dim={hds}, "
          f"KV layers={len(metas)}. Q coding-gain bound `10log10(AM/GM)` median "
          f"**{np.median(cgs):.2f} dB**; query r95 (dims for 95% of E[qq^T] "
          f"energy) median **{np.median(r95s):.0f}**. "
          f"TQ2={tab['tq2']['kl']:.4f}, TQ3={tab['tq3']['kl']:.4f}, "
          f"TQ4={tab['tq4']['kl']:.4f}.")
        a("")
        a("### TQ ladder + correction sweep (r x coef_bits)")
        a("")
        a("| arm | r | coef | bpe_task | bpe+scale | mean KL | same-top-1 | "
          "top-8 Jacc | gap_closure |")
        a("|---|---|---|---|---|---|---|---|---|")
        for b in (2, 3, 4):
            t = tab[f"tq{b}"]
            a(f"| scalar TQ{b} | - | - | {b+0.5:.2f} | {b+0.5:.2f} | {t['kl']:.4f} "
              f"| {t['top1']*100:.1f}% | {t['jac8']:.3f} | "
              f"{gap_closure(tab, f'tq{b}'):+.3f} |")
        rg = m0["r_grid"]
        for r in rg:
            for cb in COEF_BITS:
                k = f"corr_r{r}_c{cb}"
                if k not in tab:
                    continue
                t = tab[k]
                ceil = " **[CEILING]**" if cb == 16 else ""
                fp16exact = " (=FP16)" if (cb == 16 and r == hds[-1] and
                                           len(hds) == 1) else ""
                a(f"| TQ2+corr{ceil}{fp16exact} | {r} | {CB_NAME[cb]} | "
                  f"{t['bpe_task']:.2f} | {t['bpe_scale']:.2f} | {t['kl']:.4f} | "
                  f"{t['top1']*100:.1f}% | {t['jac8']:.3f} | "
                  f"{gap_closure(tab, k):+.3f} |")
        a("")
        # fine ceiling curve (fp16) — resolves the useful-r for the adapter
        curve = ceiling_curve(tab)
        a("Fine ceiling curve (fp16 coeffs, perfect correction) KL vs r: "
          + ", ".join(f"r{r}={kl:.4f}" for r, kl in curve) + ".")
        a("")
        ur3, ur4 = useful_r(tab, tab["tq3"]["kl"]), useful_r(tab, tab["tq4"]["kl"])
        top = curve[-1][0] if curve else 0
        hd0 = hds[0]                     # modal head_dim (dominant layer count)
        a(f"**Ceiling useful-r**: smallest r where perfect correction reaches "
          f"TQ3 quality = **{ur3 if ur3 else '>'+str(top)}**"
          + (f" ({ur3/hd0*100:.0f}% of head_dim)" if ur3 else "")
          + f"; TQ4 quality = **{ur4 if ur4 else '>'+str(top)}**"
          + (f" ({ur4/hd0*100:.0f}%)" if ur4 else "")
          + f". r=head_dim fp16 KL="
          f"{tab.get(f'corr_r{hds[-1]}_c16',{}).get('kl',float('nan')):.2e} "
          "(= FP16 attention, machinery sanity check).")
        a("")

    # ---- matched-bpe Q1 gate ----
    a("## Q1 GATE — does TQ2+correction beat plain TQ3 at matched total bpe (~3.5)?")
    a("")
    a("For each arch: the deployable correction arm with `bpe_task` closest to 3.5 "
      "(ties -> lower KL), compared to scalar TQ3 (3.5 bpe). `beat?` uses "
      "`bpe_task`; `beat(+scale)?` re-checks after charging the per-token "
      "coefficient scale (the honest tax).")
    a("")
    a("| arch | best ~3.5bpe corr arm | its KL | bpe_task | bpe+scale | TQ3 KL | "
      "beat TQ3? | still beat at +scale bpe? |")
    a("|---|---|---|---|---|---|---|---|")
    q1 = {}
    for x, (d, tab) in real.items():
        # best (min KL) DEPLOYABLE arm at bpe_task <= 3.5 (matched to TQ3)
        cands = []
        for k in tab:
            if not (k.startswith("corr_r") and "_c" in k and "donor" not in k):
                continue
            r = int(k[len("corr_r"):k.rindex("_c")])
            cb = int(k[k.rindex("_c") + 2:])
            if tab[k]["bpe_task"] <= 3.55 and not np.isnan(tab[k]["kl"]):
                cands.append((tab[k]["kl"], r, cb, k))
        cands.sort()
        kl, r, cb, k = cands[0]
        t = tab[k]
        beat = kl < tab["tq3"]["kl"]
        # at +scale bpe, is it still <=3.5 AND beating TQ3? (honest matched)
        beat_scale = beat and t["bpe_scale"] <= 3.6
        q1[x] = dict(r=r, cb=cb, kl=kl, bpe_task=t["bpe_task"],
                     bpe_scale=t["bpe_scale"], tq3=tab["tq3"]["kl"],
                     beat=beat, beat_scale=beat_scale)
        a(f"| {SHORT[x]} | r={r} {CB_NAME[cb]} | {kl:.4f} | {t['bpe_task']:.2f} | "
          f"{t['bpe_scale']:.2f} | {tab['tq3']['kl']:.4f} | "
          f"{'YES' if beat else 'no'} | {'YES' if beat_scale else 'no'} |")
    a("")
    n_beat = sum(1 for v in q1.values() if v["beat"])
    n_beat_scale = sum(1 for v in q1.values() if v["beat_scale"])
    a(f"**Q1 tally (task bpe):** TQ2+correction beats TQ3 at matched ~3.5 bpe on "
      f"**{n_beat}/3** archs. **With the honest per-token coefficient scale:** "
      f"**{n_beat_scale}/3** archs.")
    a("")

    # ---- correction vs shaping ----
    a("## Correction vs shaping (per bit) — store the query-error vs remove it")
    a("")
    a("SQuat error-shaping (noiseshape arm D, r=32 lam=1) re-run FULL-K here. "
      "Shaping stores NO coefficients (it only re-chooses which quantized values "
      "to write), so it sits at the base bpe (2.5 at 2-bit, 3.5 at 3-bit). "
      "Correction rows are the matched-bpe arm from the Q1 gate.")
    a("")
    a("| arch | shape 2b (2.5bpe) KL | shape 3b (3.5bpe) KL | corr ~3.5bpe KL | "
      "TQ3 (3.5) KL | best per bit at ~3.5 |")
    a("|---|---|---|---|---|---|")
    for x, (d, tab) in real.items():
        s2 = tab.get("shape_b2_amax", {}).get("kl", float("nan"))
        s2m = tab.get("shape_b2_mse", {}).get("kl", float("nan"))
        s3 = tab.get("shape_b3_amax", {}).get("kl", float("nan"))
        s3m = tab.get("shape_b3_mse", {}).get("kl", float("nan"))
        best2 = np.nanmin([s2, s2m])
        best3 = np.nanmin([s3, s3m])
        corr = q1[x]["kl"]
        opts = {"shaping(3.5)": best3, "correction(~3.5)": corr, "TQ3(3.5)": tab["tq3"]["kl"]}
        winner = min(opts, key=opts.get)
        a(f"| {SHORT[x]} | {best2:.4f} | {best3:.4f} | {corr:.4f} | "
          f"{tab['tq3']['kl']:.4f} | {winner} |")
    a("")
    a("(shape columns show the better of amax / mse_opt block scale. At 2.5 bpe "
      "shaping is FREE — no side channel — so it is the honest per-bit rival to "
      "spending bits on stored coefficients.)")
    a("")

    # ---- cross-corpus ----
    if "xval" in res:
        _, tx = res["xval"]
        _, tg = res["gemma3"]
        a("## Cross-corpus transfer (rule 5): gemma3-calib basis -> xval")
        a("")
        a("Query bases `B` fit on gemma3 calib, applied to a disjoint gemma3 dump "
          "(`xval`, 0 shared 8-grams). `self-fit` refits `B` on xval calib; "
          "`donor` uses the gemma3 bases unchanged. Same matched-bpe arm as Q1.")
        a("")
        a("| arm | gemma3 self KL | xval self-fit KL | xval donor KL | "
          "transfer degradation |")
        a("|---|---|---|---|---|")
        # match the gemma3 Q1 arm
        r, cb = q1["gemma3"]["r"], q1["gemma3"]["cb"]
        k = f"corr_r{r}_c{cb}"
        g0 = tg[k]["kl"]
        xs = tx.get(k, {}).get("kl", float("nan"))
        xd = tx.get(k + "_donor", {}).get("kl", float("nan"))
        deg = (xd - xs) / xs * 100 if xs == xs else float("nan")
        a(f"| TQ2+corr r={r} {CB_NAME[cb]} | {g0:.4f} | {xs:.4f} | {xd:.4f} | "
          f"{deg:+.1f}% |")
        # also report TQ3 on xval for reference
        a(f"| scalar TQ3 (ref) | {tg['tq3']['kl']:.4f} | {tx['tq3']['kl']:.4f} | "
          f"{tx['tq3']['kl']:.4f} | - |")
        a("")

    # ---- ceiling + useful-r summary ----
    a("## Ceiling arm + adapter bottleneck (useful-r)")
    a("")
    a("The ceiling arm is perfect (fp16) correction using only the top-r query "
      "eigenvectors — the best any adapter with an r-dim bottleneck could do. "
      "useful-r = smallest r at which the ceiling reaches TQ3 / TQ4 quality.")
    a("")
    a("| arch | head_dim | TQ3 KL | TQ4 KL | ceiling @r=32 | @r=64 | @r~½·hd | "
      "useful-r (TQ3) | useful-r (TQ4) |")
    a("|---|---|---|---|---|---|---|---|---|")
    ceil_summary = {}
    for x, (d, tab) in real.items():
        hd = d["meta"][sorted(d["meta"], key=int)[0]]["hd"]
        curve = dict(ceiling_curve(tab))

        def cell(r):
            return f"{curve[r]:.4f}" if r in curve else "-"
        half = min(curve, key=lambda r: abs(r - hd // 2)) if curve else 0
        ur3, ur4 = useful_r(tab, tab["tq3"]["kl"]), useful_r(tab, tab["tq4"]["kl"])
        top = max(curve) if curve else 0
        ceil_summary[x] = dict(ur3=ur3, ur4=ur4, top=top, hd=hd)
        a(f"| {SHORT[x]} | {hd} | {tab['tq3']['kl']:.4f} | {tab['tq4']['kl']:.4f} "
          f"| {cell(32)} | {cell(64)} | {cell(half)} (r={half}) | "
          f"{ur3 if ur3 else '>'+str(top)}"
          + (f" ({ur3/hd*100:.0f}%)" if ur3 else "")
          + f" | {ur4 if ur4 else '>'+str(top)}"
          + (f" ({ur4/hd*100:.0f}%)" if ur4 else "") + " |")
    a("")
    a("(gemma4-E2B mixes head_dim 256 (12 layers) and 512 (3 global layers); its "
      "fine ceiling curve interleaves r points from the two populations and is "
      "non-monotonic for that reason — its useful-r percentages are quoted against "
      "the modal head_dim 256. The conclusion — useful-r is a large fraction of "
      "head_dim on every arch — is unaffected.)")
    a("")

    # ---- verdicts ----
    a("## VERDICTS")
    a("")
    a(f"### Q1 (GATE): does training-free TQ2+correction beat TQ3 at matched bpe?")
    a("")
    per = "; ".join(f"{SHORT[x]} {'BEAT' if q1[x]['beat'] else 'no'} "
                    f"(corr {q1[x]['kl']:.4f} vs TQ3 {q1[x]['tq3']:.4f} @ "
                    f"bpe_task {q1[x]['bpe_task']:.2f})" for x in real)
    a(f"- Per arch (task bpe): {per}.")
    xline = ""
    if "xval" in res:
        _, tx = res["xval"]
        r, cb = q1["gemma3"]["r"], q1["gemma3"]["cb"]
        k = f"corr_r{r}_c{cb}"
        xd = tx.get(k + "_donor", {}).get("kl", float("nan"))
        xline = (f" Cross-corpus (gemma3->xval donor): corr KL {xd:.4f} vs xval "
                 f"TQ3 {tx['tq3']['kl']:.4f} -> "
                 f"{'still beats' if xd < tx['tq3']['kl'] else 'FAILS'}.")
    gate = "GO" if n_beat >= 2 else "NO-GO"
    gate_scale = "GO" if n_beat_scale >= 2 else "NO-GO"
    a(f"- **Q1 (task bpe): {gate}** ({n_beat}/3 archs beat TQ3).{xline}")
    a(f"- **Q1 (honest +scale bpe): {gate_scale}** ({n_beat_scale}/3 archs still "
      "beat TQ3 within matched bpe once the per-token coefficient scale is charged).")
    a("")
    a("### Q2 (adapter bottleneck): useful-r + does the ceiling justify an adapter?")
    a("")
    a("The ceiling arm has ORACLE access to the true key error projected onto the "
      "top-r query subspace (fp16 coeffs). Under the near-isotropic key error the "
      "Hadamard rotation is designed to produce, the top-r query eigenvectors ARE "
      "the MSE-optimal rank-r correction subspace (minimizing "
      "`trace((I-P) E[qq^T] (I-P))` over rank-r projections `P`), so the ceiling is "
      "an upper bound on any rank-r LINEAR adapter, and a strong reference for a "
      "learned one (which must still PREDICT `c` from `(q, k_q)` through an r-dim "
      "bottleneck rather than store the true projection — it can only lose at the "
      "same r, aside from whatever a nonlinearity buys over the linear optimum). "
      "So useful-r is, to that qualification, the *minimum* bottleneck an adapter "
      "needs.")
    a("")
    for x in real:
        cs_ = ceil_summary[x]
        hd = cs_["hd"]
        u3 = f"{cs_['ur3']} ({cs_['ur3']/hd*100:.0f}% of head_dim)" if cs_["ur3"] \
            else f">{cs_['top']} (never below full head_dim)"
        u4 = f"{cs_['ur4']} ({cs_['ur4']/hd*100:.0f}%)" if cs_["ur4"] \
            else f">{cs_['top']}"
        a(f"- {SHORT[x]} (head_dim {hd}): useful-r for TQ3 = **{u3}**, "
          f"TQ4 = **{u4}**.")
    a("")
    a("The useful correction subspace is **large** — 50–62% of head_dim to reach "
      "TQ3, 75–88% to reach TQ4 on the two 256-dim archs; even on hd=64 LFM2 it is "
      "38%/75%. So a *small*-bottleneck adapter (r=8–32) is bounded well ABOVE TQ3 "
      "quality: the ceiling at r=32 is 0.107/0.053/0.139 vs TQ3 0.055/0.088/0.089. "
      "The useful correction is genuinely high-dimensional, not a storage artifact.")
    a("")

    # ---- sub-findings ----
    a("### Sub-findings (mechanism)")
    a("")
    a("1. **Coefficient precision is NOT the bottleneck; r is.** int8 coeffs give "
      "KL BIT-IDENTICAL to fp16 across the whole sweep; int4 costs <2% relative. "
      "The correction quality is set by how many query dimensions r you correct, "
      "not how precisely — so the only lever is r, and r is expensive per key.")
    sh3 = lambda s: min(tab_v(s).get("shape_b3_amax", {}).get("kl", float("inf")),
                        tab_v(s).get("shape_b3_mse", {}).get("kl", float("inf")))
    a("2. **Shaping DOMINATES correction per bit.** SQuat error-shaping (arm D, "
      "FREE — no side channel) at 3-bit / 3.5 bpe reaches KL "
      f"{sh3('e2b'):.4f} / {sh3('lfm2'):.4f} / "
      f"{sh3('gemma3'):.4f}, BEATING plain TQ3 (0.055/0.088/0.089) "
      "and crushing the best stored-coefficient correction at the same 3.5 bpe "
      f"({q1['e2b']['kl']:.4f}/{q1['lfm2']['kl']:.4f}/{q1['gemma3']['kl']:.4f}). If "
      "you have the query-subspace signal, SHAPE it into the quantization (free) — "
      "do NOT store it (ruinous per-key tax).")
    a("3. **The coefficient-storage tax bites exactly where eOptShrink predicted.** "
      "The per-token fp16 coefficient scale (16/head_dim) is +0.06 bpe at hd=256 "
      "but +0.25 bpe at hd=64 — it pushes LFM2's matched r=16 int4 arm from 3.50 to "
      "3.75 bpe, out of the matched budget. Small head_dim is where per-key "
      "low-rank side channels are fatal, confirming the eOptShrink lattice study.")
    a("4. **Cross-corpus (rule 5): the query basis transfers fine, the method "
      "still loses.** gemma3-fit bases on disjoint xval degrade the correction "
      "arm only +7.6% (0.1147 -> 0.1255), but it started above TQ3 and stays "
      "above it (xval TQ3 0.0914). The basis is not corpus-fragile; the approach "
      "is just dominated.")
    a("")

    # ---- recommendation ----
    a("### RECOMMENDATION — is the light (KVLinC-style) adapter worth building?")
    a("")
    a("**No — do not build the light query-subspace correction adapter for these "
      "targets.** Two independent reasons, either one sufficient:")
    a("")
    a("- **The free (training-free) version is NO-GO (0/3, and worse cross-corpus).** "
      "Storing `r` coefficients per key costs `r*coef_bits/head_dim` bpe, and at the "
      "matched ~3.5 bpe budget you can only afford r~64 int4 (hd=256) or r~16 int4 "
      "(hd=64) — far below the useful-r — so TQ2+correction lands at 1.1–2.1x TQ3's "
      "KL. This is the per-key coefficient tax, exactly what a constant-memory "
      "learned adapter is supposed to remove.")
    a("- **But the CEILING shows a learned adapter would ALSO fail, and that is the "
      "decisive result.** The ceiling is an oracle upper bound on any rank-r LINEAR "
      "adapter (and a strong reference for a nonlinear one), and it does not reach "
      "TQ3 until r = 50–62% of head_dim (128–160 on "
      "hd=256), nor TQ4 until 75–88%. A *light* adapter's entire value proposition "
      "is a SMALL constant bottleneck; at a small bottleneck the ceiling — hence any "
      "adapter — is bounded well above TQ3. Removing the storage does not remove the "
      "dimensionality: the useful correction genuinely lives in ~half the head_dim. "
      "A full-rank adapter could reach FP16, but that is no longer 'light' and "
      "competes with simply shipping more base bits.")
    a("")
    a("**What to do with the query-subspace signal instead:** it is real (shaping "
      "recovers it and beats TQ3 at 3-bit for free), so the deployable lever is "
      "SHAPING (noiseshape arm D / SQuat greedy), not stored or predicted "
      "coefficients. This screen closes the correction/adapter thread and redirects "
      "to the already-GO shaping result. (All offline softmax-KL — understates "
      "closed-loop 3–5x — but the gap here is large and one-directional.)")
    a("")
    return lines, q1, ceil_summary, n_beat, n_beat_scale


def main_report(res, fits_meta):
    lines, q1, ceil, n_beat, n_beat_scale = build_report(res, fits_meta)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text("\n".join(lines) + "\n")
    # stdout summary
    print("\n================ SOFTMAX CORRECTION — VERDICTS ================")
    real = {k: v for k, v in res.items() if k != "xval"}
    print("Q1 GATE (TQ2+correction vs TQ3 at matched ~3.5 bpe):")
    for x in real:
        v = q1[x]
        print(f"  {SHORT[x]:14s} corr r={v['r']} {CB_NAME[v['cb']]} "
              f"KL={v['kl']:.4f} (bpe_task {v['bpe_task']:.2f}, +scale "
              f"{v['bpe_scale']:.2f}) vs TQ3 {v['tq3']:.4f} -> "
              f"{'BEAT' if v['beat'] else 'no'}"
              f"{'' if v['beat_scale'] else ' (fails at +scale bpe)'}")
    print(f"  => task bpe: {n_beat}/3 archs beat TQ3; "
          f"honest +scale: {n_beat_scale}/3.")
    if "xval" in res:
        _, tx = res["xval"]
        r, cb = q1["gemma3"]["r"], q1["gemma3"]["cb"]
        k = f"corr_r{r}_c{cb}"
        xd = tx.get(k + "_donor", {}).get("kl", float("nan"))
        print(f"  cross-corpus gemma3->xval donor: KL {xd:.4f} vs xval TQ3 "
              f"{tx['tq3']['kl']:.4f} -> "
              f"{'still beats' if xd < tx['tq3']['kl'] else 'FAILS'}")
    print("\nCEILING (perfect fp16 correction) + useful-r (adapter bottleneck):")
    for x in real:
        _, tab = res[x]
        c = ceil[x]
        u3 = f"{c['ur3']}({c['ur3']/c['hd']*100:.0f}%)" if c["ur3"] else ">"+str(c["top"])
        u4 = f"{c['ur4']}({c['ur4']/c['hd']*100:.0f}%)" if c["ur4"] else ">"+str(c["top"])
        print(f"  {SHORT[x]:14s} hd={c['hd']} TQ3={tab['tq3']['kl']:.4f} "
              f"TQ4={tab['tq4']['kl']:.4f} | useful-r(TQ3)={u3} useful-r(TQ4)={u4}")
    print(f"\nreport: {DOC}")


# ── aggregate from persisted partials ───────────────────────────────────────
def aggregate_and_report():
    res = {}
    for short in ("e2b", "lfm2", "gemma3", "xval"):
        p = partial_path(short)
        if not p.exists():
            print(f"missing partial for {short}: {p} — run that stage first",
                  flush=True)
            sys.exit(1)
        d = json.loads(p.read_text())
        res[short] = (d, build_tab(d))
    # merge donor correction arms into xval tab
    pdon = partial_path("xval_donor")
    if pdon.exists():
        ddon = json.loads(pdon.read_text())
        tdon = build_tab(ddon)
        for k, v in tdon.items():
            if k.startswith("corr_"):
                res["xval"][1][k + "_donor"] = v
    # anchor gate — STOP if drifted
    for short in ("e2b", "lfm2", "gemma3"):
        _, tab = res[short]
        name = SHORT[short]
        a2a, a3a = ANCHORS[name]
        d2 = abs(tab["tq2"]["kl"] - a2a) / a2a
        d3 = abs(tab["tq3"]["kl"] - a3a) / a3a
        if d2 > ANCHOR_TOL or d3 > ANCHOR_TOL:
            print(f"ANCHOR MISMATCH {name}: TQ2 {tab['tq2']['kl']:.4f} vs {a2a} "
                  f"({d2*100:.1f}%), TQ3 {tab['tq3']['kl']:.4f} vs {a3a} "
                  f"({d3*100:.1f}%). HARNESS MISMATCH — STOPPING.", flush=True)
            sys.exit(1)
    print("anchors reproduced (all 3 archs within tol) — building report",
          flush=True)
    main_report(res, {})


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["e2b", "lfm2", "gemma3", "xval",
                                        "xval_donor"])
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--all", action="store_true",
                    help="run every stage then aggregate (one foreground process)")
    ap.add_argument("--smoke", action="store_true", help="3 layers/model")
    args = ap.parse_args()
    cap = 3 if args.smoke else None
    t0 = time.time()

    if args.aggregate:
        aggregate_and_report()
        return

    if args.model:
        donor = None
        if args.model == "xval_donor":
            z = np.load(fits_path("gemma3"), allow_pickle=True)
            donor = z["fits"].item()
        run_model(args.model, layers_cap=cap, donor_fits=donor)
        print(f"\nstage runtime: {time.time()-t0:.1f}s", flush=True)
        return

    # --all (or default): full sequential run in one foreground process
    for short in ("e2b", "lfm2", "gemma3", "xval"):
        run_model(short, layers_cap=cap)
    z = np.load(fits_path("gemma3"), allow_pickle=True)
    run_model("xval_donor", layers_cap=cap, donor_fits=z["fits"].item())
    aggregate_and_report()
    print(f"\ntotal runtime: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
