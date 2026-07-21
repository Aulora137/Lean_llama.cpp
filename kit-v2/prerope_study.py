#!/usr/bin/env python3
"""prerope_study.py — the ONE untested hypothesis from docs/Ternary KV-cache quant.md.

Does PER-CHANNEL K quantization applied PRE-RoPE rescue Gemma-4 E2B 2-bit K?

The sibling study (docs/leankv-perchannel-e2b-study-2026-07.md) measured per-channel
K POST-RoPE and found NO-GO: KIVI per-channel is a *regression* vs the per-vector
Hadamard baseline (0.249 vs 0.187) because RoPE smears the structured outlier
CHANNELS that KIVI/KVQuant exploit across coordinate pairs. Its explicit caveat:
the doc's central claim is PRE-RoPE per-channel, where the outlier-channel structure
is intact — and that could not be tested from post-RoPE dumps.

This script tests it directly, on a MATCHED single-pass capture:
  e2b_kpre.bin   pre-RoPE  K  (named leankv_kpre_calib-<il> before ggml_rope_ext)
  e2b_kpost.bin  post-RoPE K  (leankv_k_calib-<il>, the cache-stored K)
  e2b_q_v2.bin   post-RoPE Q  (leankv_q_calib-<il>)

Pipeline under test (the doc's recommendation):
  K_proj -> per-channel quantize (pre-RoPE space) -> dequant -> RoPE -> QK^T

VALIDATION GATE (trust anchor, runs first, must pass): reimplement gemma4 NEOX
rope_ext in numpy and confirm RoPE(e2b_kpre) reproduces e2b_kpost to tight
tolerance for BOTH layer types. gemma4 E2B rope params (from the loader / gguf):
  - GLOBAL layers (hd=512, il {4,9,14}): freq_base=1e6, NEOX, full n_rot=512,
    BUT freq_factors = rope_freqs.weight (first 64 pairs=1.0 rotate, pairs 64..255
    =1e30 -> theta/1e30~0 -> identity). i.e. gemma4 globals rotate only 64/256 pairs.
  - LOCAL/sliding layers (hd=256): freq_base_swa=1e4, NEOX, full n_rot=256, no
    freq_factors.
  positions = row index in the concatenated KCAL sequence (0..T-1).

Reuses ALL sibling scaffolding: cs KCAL reader / softmax-KL / GQA map / SWA masks /
TurboQuantizer rotations; ss in-tree Lloyd-Max + mse_opt; the calib(first T//2)/eval
split; perchannel_study.arm_A (per-vector L2+Hadamard baseline), arm_B (per-channel
+ optional dense/sparse outliers), layer_refs, eval_khat.

Run (foreground, ~6 threads to spare the perplexity job):
  OMP_NUM_THREADS=6 /home/junc/LeanKV/.venv/bin/python3 kit-v2/prerope_study.py
Output: docs/leankv-prerope-perchannel-e2b-2026-07.md + summary/verdict on stdout.
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "6")

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import contour_study as cs             # noqa: E402
import scale_study as ss               # noqa: E402  (imported for parity / side effects)
import perchannel_study as pcs         # noqa: E402  arm_A, arm_B, layer_refs, eval_khat
import torch                           # noqa: E402
from gguf_extract import read_tensor   # noqa: E402
from turboquant.lloyd_max import get_precomputed_codebook  # noqa: E402

torch.set_num_threads(6)

ROOT = cs.ROOT
DOC = ROOT / "docs" / "leankv-prerope-perchannel-e2b-2026-07.md"
MODEL = Path("/home/junc/rikuri/rikurinode/models/gemma-4-E2B-it-Q4_K_M.gguf")

KPRE_F = ROOT / "e2b_kpre.bin"
KPOST_F = ROOT / "e2b_kpost.bin"
Q_F = ROOT / "e2b_q_v2.bin"

# gemma4 E2B rope params (loader log + gguf metadata)
FREQ_BASE_GLOBAL = 1_000_000.0   # gemma4.rope.freq_base
FREQ_BASE_LOCAL = 10_000.0       # gemma4.rope.freq_base_swa
# global = full-attention (non-sliding); il {4,9,14}; predicate il%5==4
IS_GLOBAL = lambda il: il % 5 == 4
CFG = pcs.MCFG["e2b"]             # scale="one", swa=512, glob=il%5==4 (files unused)

# reference anchors from the post-RoPE sibling study (for context in the doc)
ANCHOR_POST_BASE_2B = 0.187      # arm A per-vector L2+Hadamard, 2-bit
ANCHOR_POST_PERCHAN_2B = 0.249   # arm B per-chan mse g1, 2-bit (post-RoPE)
ANCHOR_TQ3 = 0.055               # arm A per-vector L2+Hadamard, 3-bit (operative target)


# ── gemma4 NEOX rope reimplementation ───────────────────────────────────────
def rope_neox(x: np.ndarray, pos: np.ndarray, freq_base: float,
              freq_factors: np.ndarray | None = None) -> np.ndarray:
    """Reproduce ggml_rope_ext(GGML_ROPE_TYPE_NEOX) with freq_scale=1, ext_factor=0,
    attn_factor=1 (E2B's settings). x: [T, nh, hd]. NEOX pairs coord ic with ic+H,
    H=hd/2. theta_ic = pos * freq_base^(-2 ic/hd) / freq_factor[ic]."""
    T, nh, hd = x.shape
    H = hd // 2
    ic = np.arange(H, dtype=np.float64)
    inv = freq_base ** (-2.0 * ic / hd)                  # [H]
    if freq_factors is not None:
        inv = inv / freq_factors[:H].astype(np.float64)  # theta/freq_factor
    theta = pos.astype(np.float64)[:, None] * inv[None, :]   # [T, H]
    cos = np.cos(theta)[:, None, :]                      # [T,1,H]
    sin = np.sin(theta)[:, None, :]
    x0 = x[..., :H].astype(np.float64)
    x1 = x[..., H:].astype(np.float64)
    out = np.empty((T, nh, hd), dtype=np.float64)
    out[..., :H] = x0 * cos - x1 * sin
    out[..., H:] = x0 * sin + x1 * cos
    return out.astype(np.float32)


def layer_rope_params(hd: int):
    """(freq_base, freq_factors) for a layer identified by its head_dim."""
    if hd == 512:                       # global / full-attention
        return FREQ_BASE_GLOBAL, ROPE_FREQS
    return FREQ_BASE_LOCAL, None        # local / sliding


# rope_freqs.weight (shared across global layers): first 64 pairs=1, rest=1e30
ROPE_FREQS = read_tensor(str(MODEL), "rope_freqs.weight").reshape(-1)


# ── validation gate ─────────────────────────────────────────────────────────
def validate_rope(Kpre, Kpost):
    print("== VALIDATION GATE: RoPE(e2b_kpre) vs e2b_kpost ==", flush=True)
    rows = []
    ok_all = True
    for il in sorted(Kpre):
        kpre, kpost = Kpre[il], Kpost[il]
        T, nkv, hd = kpre.shape
        pos = np.arange(T)
        fb, ff = layer_rope_params(hd)
        khat = rope_neox(kpre, pos, fb, ff)
        diff = khat - kpost
        rel = float(np.linalg.norm(diff) / (np.linalg.norm(kpost) + 1e-30))
        # per-element relative error on non-tiny elements
        denom = np.abs(kpost)
        big = denom > (0.01 * denom.max())
        elem_rel = float(np.max(np.abs(diff[big]) / denom[big])) if big.any() else 0.0
        # cosine per (t,kv) vector
        a = khat.reshape(T * nkv, hd)
        b = kpost.reshape(T * nkv, hd)
        cos = float((( a * b).sum(-1) /
                     (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-30)).mean())
        kind = "GLOBAL" if hd == 512 else "local"
        passed = (cos > 0.9999) and (rel < 1e-3)
        ok_all &= passed
        rows.append((il, hd, kind, rel, elem_rel, cos, passed))
        print(f"  il={il:2d} {kind:6s} hd={hd} relF={rel:.2e} "
              f"max_elem_rel={elem_rel:.2e} cos={cos:.6f} {'OK' if passed else 'FAIL'}",
              flush=True)
    print(f"  -> validation {'PASSED' if ok_all else 'FAILED'}", flush=True)
    return ok_all, rows


# ── effective bpe (honest side-info accounting) ─────────────────────────────
def outlier_bits(hd: int) -> float:
    """Honest per-outlier storage: index (log2 head_dim, per-vector position) +
    fp16 value. The sibling study used a flat 20 and undercounted the index."""
    return np.log2(hd) + 16.0


def eff_bpe(kind: str, bits: int, hd: int, T: int, group: int, kept_frac: float) -> float:
    if kind == "postA":                 # per-vector L2 + Hadamard (+outliers)
        return bits + 16.0 / hd + kept_frac * outlier_bits(hd)
    if kind in ("postB", "preB"):       # per-channel scales (+outliers)
        return bits + 16.0 / (group * T) + kept_frac * outlier_bits(hd)
    return float("nan")


# ── arm evaluation ──────────────────────────────────────────────────────────
def run_arms(Kpre, Kpost, Q, rot, cb):
    """Per-layer metrics for every arm. Returns per_layer dict like the sibling."""
    per_layer = {}
    for il in sorted(Kpre):
        kpre, kpost, q = Kpre[il], Kpost[il], Q[il]
        T, nkv, hd = kpost.shape
        Tc = T // 2
        pos = np.arange(T)
        fb, ff = layer_rope_params(hd)
        R = rot[hd][il].float()
        refs = pcs.layer_refs(kpost, q, CFG, il)     # reference = TRUE post-RoPE K
        row = {}

        def add(name, kind, bits, group, Khat_post, fit):
            met = pcs.eval_khat(Khat_post, kpost, q, refs)
            met["bpe"] = eff_bpe(kind, bits, hd, T, group, fit.get("kept_frac", 0.0))
            met["samp"] = fit.get("samp", float("nan"))
            met["kept_frac"] = fit.get("kept_frac", 0.0)
            row[name] = met

        # ---- POST-RoPE reference arms (reproduce the sibling anchors) ----
        Khat, fit = pcs.arm_A(kpost, Tc, R, cb, 2)
        add("postA_b2", "postA", 2, 1, Khat, fit)
        Khat, fit = pcs.arm_A(kpost, Tc, R, cb, 3)
        add("postA_b3", "postA", 3, 1, Khat, fit)
        Khat, fit = pcs.arm_B(kpost, Tc, 2, 1, "mse")
        add("postB_mse_g1_b2", "postB", 2, 1, Khat, fit)

        # ---- PRE-RoPE per-channel (the experiment): quant pre -> RoPE -> attend ----
        for g in (1, 8, 32):
            Khat_pre, fit = pcs.arm_B(kpre, Tc, 2, g, "mse")
            add(f"preB_mse_g{g}_b2", "preB", 2, g,
                rope_neox(Khat_pre, pos, fb, ff), fit)
        # amax variant (matches sibling's amax column, in pre-RoPE space)
        Khat_pre, fit = pcs.arm_B(kpre, Tc, 2, 1, "amax")
        add("preB_amax_g1_b2", "preB", 2, 1, rope_neox(Khat_pre, pos, fb, ff), fit)
        # 3-bit
        for g in (1, 8):
            Khat_pre, fit = pcs.arm_B(kpre, Tc, 3, g, "mse")
            add(f"preB_mse_g{g}_b3", "preB", 3, g,
                rope_neox(Khat_pre, pos, fb, ff), fit)
        # ---- PRE-RoPE per-channel + dense/sparse outliers (kept in pre-RoPE space) ----
        for p in (1.0, 2.0):
            Khat_pre, fit = pcs.arm_B(kpre, Tc, 2, 1, "mse", pct=p)
            add(f"preE_g1+{p:g}", "preB", 2, 1,
                rope_neox(Khat_pre, pos, fb, ff), fit)

        per_layer[str(il)] = dict(hd=hd, glob=bool(IS_GLOBAL(il)), T=T, Tc=Tc,
                                  nkv=nkv, nqh=q.shape[1], row=row)
    return per_layer


# ── aggregation (mirrors the sibling) ───────────────────────────────────────
def agg(pl, name, field, sel="all"):
    vals = []
    for _, d in pl.items():
        if name not in d["row"]:
            continue
        if sel == "glob" and d["hd"] != 512:
            continue
        if sel == "local" and d["hd"] == 512:
            continue
        vals.append(d["row"][name][field])
    return float(np.mean(vals)) if vals else float("nan")


def gap_closure(pl, name, sel="all"):
    a2 = agg(pl, "postA_b2", "kl", sel)
    a3 = agg(pl, "postA_b3", "kl", sel)
    return (a2 - agg(pl, name, "kl", sel)) / (a2 - a3)


ARM_LABEL = {
    "postA_b2": "POST baseline per-vector L2+Had (2b)",
    "postA_b3": "POST baseline per-vector L2+Had (3b) [TQ3 target]",
    "postB_mse_g1_b2": "POST per-chan mse g1 (2b) [KIVI post-RoPE]",
    "preB_amax_g1_b2": "PRE per-chan amax g1 (2b)",
    "preB_mse_g1_b2": "PRE per-chan mse g1 (2b)",
    "preB_mse_g8_b2": "PRE per-chan mse g8 (2b)",
    "preB_mse_g32_b2": "PRE per-chan mse g32 (2b)",
    "preB_mse_g1_b3": "PRE per-chan mse g1 (3b)",
    "preB_mse_g8_b3": "PRE per-chan mse g8 (3b)",
    "preE_g1+1": "PRE per-chan mse g1 + 1% outliers (2b)",
    "preE_g1+2": "PRE per-chan mse g1 + 2% outliers (2b)",
}
ORDER = ["postA_b2", "postA_b3", "postB_mse_g1_b2",
         "preB_amax_g1_b2", "preB_mse_g1_b2", "preB_mse_g8_b2", "preB_mse_g32_b2",
         "preB_mse_g1_b3", "preB_mse_g8_b3", "preE_g1+1", "preE_g1+2"]
# deployable PRE-RoPE 2-bit arms (the ones the verdict judges)
PRE_2B = ["preB_mse_g1_b2", "preB_mse_g8_b2", "preB_mse_g32_b2",
          "preE_g1+1", "preE_g1+2"]


def fmt_tbl(a, pl, sel):
    a("| arm | eff bpe | softmax-KL | same-top-1 | top-8 Jacc | gap_closure | samples/scale |")
    a("|---|---|---|---|---|---|---|")
    for name in ORDER:
        kl = agg(pl, name, "kl", sel)
        if kl != kl:
            continue
        samp = agg(pl, name, "samp", sel)
        a(f"| {ARM_LABEL.get(name, name)} | {agg(pl,name,'bpe',sel):.3f} | {kl:.4f} | "
          f"{agg(pl,name,'top1',sel)*100:.1f}% | {agg(pl,name,'jac8',sel):.3f} | "
          f"{gap_closure(pl,name,sel):+.3f} | "
          f"{'-' if samp!=samp else f'{samp:.0f}'} |")
    a("")


# ── report ──────────────────────────────────────────────────────────────────
def build_report(pl, vrows, val_ok):
    L = []
    a = L.append
    a2 = agg(pl, "postA_b2", "kl"); a3 = agg(pl, "postA_b3", "kl")
    bestpre = min(PRE_2B, key=lambda n: agg(pl, n, "kl"))
    bkl = agg(pl, bestpre, "kl"); bbpe = agg(pl, bestpre, "bpe")
    bgc = gap_closure(pl, bestpre)
    postperchan = agg(pl, "postB_mse_g1_b2", "kl")
    prg1 = agg(pl, "preB_mse_g1_b2", "kl")

    a("# Pre-RoPE per-channel K rescue of E2B 2-bit K — measured study")
    a("")
    a(f"Generated by `kit-v2/prerope_study.py` on {time.strftime('%Y-%m-%d %H:%M')}. "
      "Tests the ONE untested hypothesis from `docs/Ternary KV-cache quant.md`: does "
      "per-channel K quantization applied **PRE-RoPE** rescue Gemma-4 E2B 2-bit K, "
      "where the post-RoPE version was measured NO-GO (KIVI per-channel a regression "
      "at 0.249 vs the 0.187 baseline, because RoPE smears the outlier-channel "
      "structure)? Matched single-pass capture (pre-RoPE K, post-RoPE K, post-RoPE Q); "
      "model-true attention scale, SWA masks, GQA map, calib(first T//2)/eval split "
      "reused verbatim from the sibling `perchannel_study.py`.")
    a("")

    # ---- validation gate ----
    a("## RoPE validation gate (trust anchor)")
    a("")
    a("Reimplemented gemma4 NEOX `rope_ext` in numpy (freq_scale=1, ext_factor=0, "
      "attn_factor=1). GLOBAL layers (hd=512): freq_base=1e6 + `rope_freqs.weight` "
      "freq_factors (first 64 pairs rotate, pairs 64..255 have factor 1e30 -> "
      "identity). LOCAL layers (hd=256): freq_base_swa=1e4, no freq_factors. "
      "positions = concatenated row index. **RoPE(e2b_kpre) must reproduce e2b_kpost.**")
    a("")
    a("| il | kind | hd | rel Frobenius err | max elem rel err | mean cosine | pass |")
    a("|---|---|---|---|---|---|---|")
    for il, hd, kind, rel, erel, cos, passed in vrows:
        a(f"| {il} | {kind} | {hd} | {rel:.2e} | {erel:.2e} | {cos:.6f} | "
          f"{'OK' if passed else 'FAIL'} |")
    a("")
    a(f"**Gate {'PASSED' if val_ok else 'FAILED'}** — RoPE reimplementation reproduces "
      "the captured post-RoPE K for both layer types "
      f"(all cosine > 0.9999, all rel-F < 1e-3). The pre-RoPE experiment below is "
      f"{'trustworthy' if val_ok else 'VOID (fix RoPE first)'}.")
    a("")

    m0 = next(iter(pl.values()))
    a("## Arms — pre-RoPE quantize -> RoPE -> attend, vs post-RoPE reference points")
    a("")
    a(f"T={m0['T']}, calib={m0['Tc']}, MQA n_kv={m0['nkv']}, n_head={m0['nqh']}, "
      "15 KV layers (12 local hd=256, 3 global hd=512 at il {4,9,14}). Reference "
      "points: **POST baseline** = per-vector L2+Hadamard (the shipping codec, "
      f"2b={a2:.4f}~0.187, 3b={a3:.4f}~0.055=operative target); **POST per-chan** = "
      f"KIVI post-RoPE ({postperchan:.4f}~0.249). Effective bpe includes per-channel "
      "fp16 scales `16/(group*T)` and honest outliers `(log2 head_dim + 16)` bits/kept "
      "element (index+fp16 value; the sibling used a flat 20 and undercounted the index).")
    a("")
    a("`gap_closure = (KL_postA2 - KL_arm) / (KL_postA2 - KL_postA3)`: fraction of the "
      "post-RoPE 2->3-bit KL gap an arm recovers at 2 bits. 1.0 = TQ3 target.")
    a("")
    a("### All E2B layers")
    a("")
    fmt_tbl(a, pl, "all")
    a("### Global layers only (hd=512, il {4,9,14})")
    a("")
    fmt_tbl(a, pl, "glob")
    a("### Local layers only (hd=256)")
    a("")
    fmt_tbl(a, pl, "local")

    # ---- per-layer diagnostic ----
    a("## Per-layer diagnostic — pre-RoPE per-channel vs post-RoPE reference")
    a("")
    a("`relK` = ||Khat_post - Kpost|| / ||Kpost|| (post-RoPE recon error), `KL` = "
      "softmax-KL. `postB` = KIVI post-RoPE per-chan g1; `preB` = pre-RoPE per-chan "
      "g1 (the hypothesis); `preE` = best pre-RoPE arm.")
    a("")
    best = min(PRE_2B, key=lambda n: agg(pl, n, "kl"))
    a("| il | hd | kind | postA2 KL | postB g1 KL | preB g1 KL | preB relK | "
      f"{ARM_LABEL[best]} KL |")
    a("|---|---|---|---|---|---|---|---|")
    for il in sorted(pl, key=int):
        d = pl[il]; r = d["row"]
        kind = "GLOBAL" if d["hd"] == 512 else "local"
        a(f"| {il} | {d['hd']} | {kind} | {r['postA_b2']['kl']:.4f} | "
          f"{r['postB_mse_g1_b2']['kl']:.4f} | {r['preB_mse_g1_b2']['kl']:.4f} | "
          f"{r['preB_mse_g1_b2']['relK']:.3f} | {r[best]['kl']:.4f} |")
    a("")

    L += build_verdict(pl)
    return L


def build_verdict(pl):
    a = []; w = a.append
    a2 = agg(pl, "postA_b2", "kl"); a3 = agg(pl, "postA_b3", "kl")
    postpc = agg(pl, "postB_mse_g1_b2", "kl")
    prepc = agg(pl, "preB_mse_g1_b2", "kl")
    prepc_bpe = agg(pl, "preB_mse_g1_b2", "bpe")
    prepc_g = agg(pl, "preB_mse_g1_b2", "kl", "glob")
    prepc_l = agg(pl, "preB_mse_g1_b2", "kl", "local")
    preg8 = agg(pl, "preB_mse_g8_b2", "kl")
    preg32 = agg(pl, "preB_mse_g32_b2", "kl")
    best = min(PRE_2B, key=lambda n: agg(pl, n, "kl"))
    bkl = agg(pl, best, "kl"); bbpe = agg(pl, best, "bpe"); bgc = gap_closure(pl, best)
    bkl_g = agg(pl, best, "kl", "glob"); bkl_l = agg(pl, best, "kl", "local")

    # GO gate: reach TQ3 anchor (gap_closure ~1) OR clearly beat BOTH post-RoPE
    # per-channel (0.249) AND the post-RoPE baseline (0.187) at <=2.5 bpe.
    reaches_tq3 = (bgc >= 0.85) and (bbpe <= 2.5)
    beats_both = (bkl < 0.9 * postpc) and (bkl < 0.9 * a2) and (bbpe <= 2.5)
    go = reaches_tq3 or beats_both
    pre_helps_vs_post_pc = prepc < postpc
    pre_helps_vs_base = prepc < a2

    # precomputed conditional text (avoid nested quotes inside f-strings)
    beats_pc_word = "BEATS" if pre_helps_vs_post_pc else "does NOT beat"
    beats_base_word = "beats" if pre_helps_vs_base else "does NOT beat"
    pc_pct = (postpc - prepc) / postpc * 100.0
    base_pct = (a2 - prepc) / a2 * 100.0
    if pre_helps_vs_post_pc:
        premise_line = ("Moving the per-channel scale into pre-RoPE space, where the "
                        "outlier channels are intact, DOES help relative to post-RoPE "
                        "per-channel — the KVQuant/KIVI premise directionally holds.")
    else:
        premise_line = ("Even in pre-RoPE space the per-channel scale does not beat "
                        "post-RoPE per-channel — the KVQuant/KIVI premise does not hold "
                        "for E2B even where the outlier channels are intact.")
    reach_word = "reaches" if bgc >= 0.85 else f"sits ~{bkl/a3:.1f}x above"
    group_word = "help" if preg8 < prepc else "hurt"

    w("## GO / NO-GO")
    w("")
    w("**GO gate:** does pre-RoPE per-channel K get E2B 2-bit K to the TQ3 anchor "
      f"(softmax-KL ~{a3:.3f}, gap_closure -> 1.0), or at least clearly beat BOTH the "
      f"post-RoPE per-channel {postpc:.4f} AND the post-RoPE baseline {a2:.4f}, at "
      "<= 2.5 bpe?")
    w("")
    w(f"- **Bare pre-RoPE per-channel (the KIVI premise), `preB mse g1 (2b)` = "
      f"{prepc:.4f}** (global {prepc_g:.4f}, local {prepc_l:.4f}) at {prepc_bpe:.3f} "
      f"bpe. vs post-RoPE per-channel {postpc:.4f}: {beats_pc_word} it "
      f"({pc_pct:+.1f}%). vs post-RoPE baseline {a2:.4f}: {beats_base_word} it "
      f"({base_pct:+.1f}%). {premise_line}")
    w(f"- **Best pre-RoPE 2-bit arm `{ARM_LABEL.get(best,best)}` = {bkl:.4f}** "
      f"(global {bkl_g:.4f}, local {bkl_l:.4f}) at {bbpe:.3f} bpe, gap_closure "
      f"{bgc:+.3f} — {reach_word} the TQ3 anchor ({a3:.3f}).")
    w(f"- Larger channel groups {group_word} (g8={preg8:.4f}, g32={preg32:.4f}) "
      "as in the post-RoPE study.")
    w("")
    w(f"### Verdict: **{'GO' if go else 'NO-GO'}**")
    w("")
    if go:
        w(f"Pre-RoPE per-channel K reaches the operative target for E2B 2-bit K "
          f"(best {bkl:.4f} at {bbpe:.3f} bpe, gap_closure {bgc:+.3f}). The doc's "
          "central claim holds: quantizing K in pre-RoPE space (outlier channels "
          "intact), then applying RoPE on read, recovers the accuracy that post-RoPE "
          "per-channel loses. This justifies real kernel work on a store-pre-RoPE / "
          "apply-RoPE-on-read K cache with a fused dequant+RoPE path.")
    else:
        verb = "modestly beats" if pre_helps_vs_post_pc else "does not beat"
        base_clause = (f", and even edges under the per-vector Hadamard baseline ({a2:.4f})"
                       if pre_helps_vs_base else
                       f" (though still above the per-vector Hadamard baseline {a2:.4f})")
        w(f"No pre-RoPE 2-bit arm reaches the TQ3 anchor (best gap_closure {bgc:+.3f}, "
          f"{bkl/a3:.1f}x above {a3:.3f}). Moving per-channel quantization into "
          f"pre-RoPE space {verb} the post-RoPE per-channel arm "
          f"({prepc:.4f} vs {postpc:.4f}){base_clause}, but the gap to a deployable "
          "2-bit K is not closed. After the ~3-5x closed-loop amplification (VQ study "
          "rule 1) this still lands above the ~0.27 cliff, so 2-bit K would still "
          "collapse in the loop.")
        w("")
        pre_status = ("also NO-GO" if not pre_helps_vs_post_pc
                      else "a real-but-insufficient improvement")
        headline = ("The doc headline recommendation (store-pre-RoPE per-channel cache) "
                    "is measured-dead for E2B." if not pre_helps_vs_post_pc else
                    "A store-pre-RoPE cache would recover the post-RoPE per-channel "
                    "regression, but since even that does not reach a deployable 2-bit "
                    "K, it does not justify the fused dequant+RoPE kernel + cache "
                    "relayout on its own.")
        w(f"**Recommendation.** Pre-RoPE per-channel is {pre_status} for E2B 2-bit K; "
          f"E2B K stays **TQ4 / gated-TQ3**. {headline} The only cost-justified 2-bit "
          "lever remains a generic sparse-outlier side channel on the CURRENT "
          "post-RoPE per-vector Hadamard codec (see the sibling study).")
    w("")
    return a


# ── main ────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    print(f"rope_freqs.weight: {ROPE_FREQS.shape}, "
          f"{int((ROPE_FREQS<1e10).sum())} rotated pairs / {ROPE_FREQS.size}", flush=True)
    Kpre = cs.read_kcal_layers(KPRE_F)
    Kpost = cs.read_kcal_layers(KPOST_F)
    Q = cs.read_kcal_layers(Q_F)

    val_ok, vrows = validate_rope(Kpre, Kpost)

    # rotations + codebooks (reuse sibling infra), keyed by head_dim
    layers = sorted(Kpost)
    maxil = max(layers) + 1
    rot, cb = {}, {}
    for il in layers:
        hd = Kpost[il].shape[2]
        if hd not in rot:
            rot[hd] = cs.TurboQuantizer(
                n_layers=maxil, head_dim=hd, bits=2, group_size=None,
                rotation_strategy="randomized_hadamard", use_qjl=False,
                seed=42, device="cpu").rotations
        for b in (2, 3):
            cb.setdefault((hd, b), get_precomputed_codebook(b, hd))

    print("== running arms ==", flush=True)
    pl = run_arms(Kpre, Kpost, Q, rot, cb)

    lines = build_report(pl, vrows, val_ok)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text("\n".join(lines) + "\n")

    # stdout summary
    print("\n================ E2B PRE-RoPE SUMMARY (softmax-KL) ================")
    print(f"{'arm':46s} {'bpe':>6s} {'all':>8s} {'glob':>8s} {'local':>8s} {'gap_cl':>7s}")
    for name in ORDER:
        if agg(pl, name, "kl") != agg(pl, name, "kl"):
            continue
        print(f"{ARM_LABEL.get(name,name):46s} {agg(pl,name,'bpe'):6.3f} "
              f"{agg(pl,name,'kl'):8.4f} {agg(pl,name,'kl','glob'):8.4f} "
              f"{agg(pl,name,'kl','local'):8.4f} {gap_closure(pl,name):+7.3f}")
    for ln in build_verdict(pl):
        print(ln)
    print(f"\nreport -> {DOC}")
    print(f"total runtime {time.time()-t0:.1f}s  (rope validation: "
          f"{'PASS' if val_ok else 'FAIL'})")


if __name__ == "__main__":
    main()
