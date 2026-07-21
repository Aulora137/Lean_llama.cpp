#!/usr/bin/env python3
"""
kv_policy.py — the adaptive KV-quantization PROBER (2026-07-21)

A geometry -> method decision function. Reads a GGUF, extracts KV-cache geometry
WITHOUT running the model, classifies the architecture family, and emits a
KV-quant plan grounded in the MEASURED LeanKV menu (docs/leankv-*.md). Every
branch cites the doc it comes from; on architectures we never measured it says so
instead of faking confidence.

    python3 kit-v2/kv_policy.py <model.gguf> [--target-bpw N] [--emit-plan plan.types]
    python3 kit-v2/kv_policy.py --validate [--models-dir DIR]   # self-check vs the doc

PROBE (step 1)  — geometry only, one GGUF header read, no inference:
    arch, n_layer, n_embd, n_head, n_head_kv (scalar or per-layer array; 0 = a
    conv/SSM/non-attention layer), per-layer head_dim (gemma4: key_length 512 on
    global / key_length_swa 256 on local, read from sliding_window_pattern),
    shared_kv_layers, rope freq_base(_swa), KV-owning layers (via blk.N.attn_k
    tensor presence). Derives: q_dim, rank_bounded[il], mqa_ratio,
    kv_bypass_fraction, arch_family.

POLICY (step 2) — the decision function encodes the measured menu. See MENU below
    and docs/leankv-adaptive-menu-2026-07.md.

VALIDATE (step 3) — --validate reproduces every measured model's shipping config
    from the production ladder (docs/leankv-tq-production-ladder-2026-07.md).

Reuses kit-v2/gguf_extract.py's minimal GGUF reader (_R).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import struct
import sys
from dataclasses import dataclass, field, asdict
from typing import Optional

# --- REUSE the minimal GGUF reader from the sibling study tool -----------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf_extract import _R, _SCALAR_FMT, GGUF_STRING, GGUF_ARRAY  # noqa: E402


# ---------------------------------------------------------------------------
# GGUF header reader: metadata KV dict + tensor names (reuses _R primitives)
# ---------------------------------------------------------------------------

def _read_value(r: _R, vt: int):
    if vt in _SCALAR_FMT:
        fmt, sz = _SCALAR_FMT[vt]
        return struct.unpack(fmt, r.take(sz))[0]
    if vt == GGUF_STRING:
        return r.gstr()
    if vt == GGUF_ARRAY:
        et = r.u32()
        cnt = r.u64()
        if et == GGUF_STRING:
            return [r.gstr() for _ in range(cnt)]
        fmt, sz = _SCALAR_FMT[et]
        return [struct.unpack(fmt, r.take(sz))[0] for _ in range(cnt)]
    raise ValueError(f"unknown gguf value type {vt}")


def read_header(path: str) -> tuple[dict, list[str]]:
    """Return (metadata_dict, tensor_names). Reads only the header region (bounded
    64 MiB is plenty for metadata + tensor infos even at 128k vocab)."""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        buf = f.read(min(size, 64 << 20))
    r = _R(buf)
    assert r.take(4) == b"GGUF", "not a GGUF file"
    r.u32()                       # version
    n_tensors = r.u64()
    n_kv = r.u64()
    meta: dict = {}
    for _ in range(n_kv):
        key = r.gstr()
        vt = r.u32()
        meta[key] = _read_value(r, vt)
    names: list[str] = []
    for _ in range(n_tensors):
        name = r.gstr()
        ndim = r.u32()
        for _ in range(ndim):
            r.u64()
        r.u32()                   # ggml type
        r.u64()                   # offset
        names.append(name)
    return meta, names


# ---------------------------------------------------------------------------
# PROBE — geometry extraction
# ---------------------------------------------------------------------------

# arches this prober has grounding for (measured in the LeanKV docs)
KNOWN_ARCHES = {"gemma4", "gemma3", "lfm2", "lfm2moe", "qwen35"}
# arches with no autoregressive KV cache to quantize (encoders etc.)
NO_KV_ARCHES = {"bert", "nomic-bert", "jina-bert-v2"}


@dataclass
class Geom:
    path: str
    name: str
    arch: str
    n_layer: int
    n_embd: int
    n_head: int
    q_dim: int
    n_head_kv_attn: int                 # kv heads on an attention layer
    mqa_ratio: float
    kv_owning_layers: list[int]         # layers that compute their own K (attn_k tensor)
    n_kv_owning: int
    head_dim_local: int
    head_dim_global: Optional[int]      # None if no local/global split
    global_layers: list[int]            # gemma4 full-attention layers (rank-bounded on wide models)
    rank_bounded_layers: list[int]
    rank_bounded_any: bool
    rank_bounded_all_owners: bool
    shared_kv_layers: int               # gemma4 cross-layer reuse count (0 if none)
    kv_bypass_fraction: float           # fraction of layers with NO own KV tensor
    rope_freq_base: Optional[float]
    rope_freq_base_swa: Optional[float]
    has_shortconv: bool
    has_ssm: bool
    family: str


def _mv(meta: dict, arch: str, *suffixes, default=None):
    for s in suffixes:
        k = f"{arch}.{s}"
        if k in meta:
            return meta[k]
    return default


def probe(path: str) -> Geom:
    meta, names = read_header(path)
    arch = meta.get("general.architecture", "?")
    name = meta.get("general.name", os.path.basename(path))

    n_layer = int(_mv(meta, arch, "block_count", default=0))
    n_embd = int(_mv(meta, arch, "embedding_length", default=0))
    n_head = int(_mv(meta, arch, "attention.head_count", default=0))

    # per-layer head_count_kv may be a scalar or an array (0 = conv/SSM layer)
    nhkv_raw = _mv(meta, arch, "attention.head_count_kv", default=1)
    if isinstance(nhkv_raw, list):
        nhkv_per_layer = [int(x) for x in nhkv_raw]
    else:
        nhkv_per_layer = [int(nhkv_raw)] * max(n_layer, 1)

    # KV-owning layers: the robust, arch-agnostic signal is the presence of a
    # per-layer K projection tensor (blk.N.attn_k.weight). Shared/conv/SSM layers
    # lack it. (On gemma4 EVERY layer has attn_k; the sharing is runtime — handled
    # separately via shared_kv_layers.)
    owns = set()
    for nm in names:
        m = re.match(r"blk\.(\d+)\.attn_k(?:\.|_norm)", nm)
        if m:
            owns.add(int(m.group(1)))
    # gemma4-style cross-layer KV sharing: attn_k weights persist on EVERY layer,
    # but only the first (n_layer - shared_kv_layers) layers actually own a KV
    # cache. Engine truth: has_kv(il) = il < n_layer_kv_from_start, where
    # n_layer_kv_from_start = n_layer - shared_kv_layers (src/llama-hparams.cpp,
    # src/llama.cpp llama_kv_cache_init pushes nullptr k_l/v_l for shared layers).
    # LFM2/Qwen omit attn_k on conv/SSM layers, so the tensor signal is already
    # correct there; this filter only fires for the shared-KV Gemma family.
    shared_kv_layers = int(_mv(meta, arch, "attention.shared_kv_layers", default=0) or 0)
    if shared_kv_layers > 0 and n_layer > shared_kv_layers:
        n_owner = n_layer - shared_kv_layers
        owns = {il for il in owns if il < n_owner}
    kv_owning = sorted(owns)
    n_kv_owning = len(kv_owning)

    n_head_kv_attn = 1
    for il in kv_owning:
        if il < len(nhkv_per_layer) and nhkv_per_layer[il] > 0:
            n_head_kv_attn = nhkv_per_layer[il]
            break
    else:
        # fall back to any positive entry / scalar
        pos = [x for x in nhkv_per_layer if x > 0]
        n_head_kv_attn = pos[0] if pos else 1

    q_dim = n_embd // n_head if n_head else 0
    mqa_ratio = n_head / max(n_head_kv_attn, 1)

    # head_dim: gemma-style key_length (global) / key_length_swa (local); else
    # explicit key_length; else n_embd/n_head.
    key_len = _mv(meta, arch, "attention.key_length")
    key_len_swa = _mv(meta, arch, "attention.key_length_swa")
    head_dim_default = q_dim
    if key_len is not None:
        head_dim_local = int(key_len_swa) if key_len_swa is not None else int(key_len)
        head_dim_global = int(key_len) if key_len_swa is not None else None
    else:
        head_dim_local = head_dim_default
        head_dim_global = None

    # global (full-attention) layers on gemma4: sliding_window_pattern[il] == False
    swa_pattern = _mv(meta, arch, "attention.sliding_window_pattern")
    global_layers: list[int] = []
    if isinstance(swa_pattern, list) and head_dim_global is not None:
        global_layers = [il for il, is_swa in enumerate(swa_pattern) if not is_swa]

    def head_dim_of(il: int) -> int:
        if head_dim_global is not None and il in global_layers:
            return head_dim_global
        return head_dim_local

    rank_bounded = [il for il in kv_owning if q_dim < head_dim_of(il)]
    rank_bounded_any = len(rank_bounded) > 0
    rank_bounded_all_owners = len(rank_bounded) == n_kv_owning and n_kv_owning > 0

    kv_bypass_fraction = (1.0 - n_kv_owning / n_layer) if n_layer else 0.0

    rope_fb = _mv(meta, arch, "rope.freq_base")
    rope_fb_swa = _mv(meta, arch, "rope.freq_base_swa")

    has_shortconv = any("shortconv" in nm or ".conv." in nm for nm in names)
    has_ssm = any(k.startswith(f"{arch}.ssm.") for k in meta) or any("ssm" in nm for nm in names)

    family = classify_family(arch, n_head, n_head_kv_attn, shared_kv_layers,
                             kv_bypass_fraction, has_shortconv, has_ssm)

    return Geom(
        path=path, name=name, arch=arch, n_layer=n_layer, n_embd=n_embd,
        n_head=n_head, q_dim=q_dim, n_head_kv_attn=n_head_kv_attn,
        mqa_ratio=round(mqa_ratio, 3), kv_owning_layers=kv_owning,
        n_kv_owning=n_kv_owning, head_dim_local=head_dim_local,
        head_dim_global=head_dim_global, global_layers=global_layers,
        rank_bounded_layers=rank_bounded, rank_bounded_any=rank_bounded_any,
        rank_bounded_all_owners=rank_bounded_all_owners,
        shared_kv_layers=shared_kv_layers,
        kv_bypass_fraction=round(kv_bypass_fraction, 3),
        rope_freq_base=rope_fb, rope_freq_base_swa=rope_fb_swa,
        has_shortconv=has_shortconv, has_ssm=has_ssm, family=family,
    )


def classify_family(arch, n_head, n_head_kv, shared_kv_layers, kv_bypass_fraction,
                    has_shortconv, has_ssm) -> str:
    if arch in NO_KV_ARCHES:
        return "no-kv-cache"
    if arch not in KNOWN_ARCHES:
        return "UNKNOWN"
    # conv-hybrid: LFM2 short-conv layers bypass KV
    if has_shortconv and kv_bypass_fraction > 0.25:
        return "conv-hybrid"
    # SSM-hybrid: mamba/deltanet layers bypass KV (Qwen3.5)
    if has_ssm and kv_bypass_fraction > 0.25:
        return "SSM-hybrid"
    # shared-KV: gemma4 cross-layer KV reuse (all layers still compute K)
    if shared_kv_layers > 0:
        return "shared-KV-MQA" if n_head_kv <= 1 else "shared-KV-GQA"
    # dense
    return "dense-MHA" if n_head_kv >= n_head else "dense-GQA"


# ---------------------------------------------------------------------------
# POLICY — the measured menu (each branch cites its doc)
# ---------------------------------------------------------------------------
#
# MENU (docs/leankv-adaptive-menu-2026-07.md, production ladder, campaign docs):
#
#   ALWAYS: scale = mse_opt @2-bit / amax @>=3-bit (scale-scheme doc);
#           --norm robust; Q-dim gate ON (per-layer, auto-promotes unsafe tiers).
#   SHIP  : TQ4/TQ4 pure on every family (ladder rule #1 — at/near the quality
#           frontier on all six measured models).
#   Aggressive option is family-specific; see below.

# nominal target-bpw -> tier. TQ stores {2.5,3.5,4.5} bpe for tq{2,3,4}_0.
def _tier_for_target(target: Optional[float]) -> str:
    if target is None or target >= 3.8:
        return "tq4"
    if target >= 2.8:
        return "tq3"
    return "tq2"


D = {  # doc citations
    "ladder": "leankv-tq-production-ladder-2026-07.md",
    "compiler": "leankv-adaptive-kv-compiler.md",
    "scale": "leankv-scale-scheme-study-2026-07.md",
    "perchan": "leankv-perchannel-e2b-study-2026-07.md",
    "e2b": "leankv-e2b-campaign-2026-07.md",
    "abl": "leankv-kv-importance-ablation-2026-07.md",
    "entropy": "leankv-entropy-lfm2-campaign-2026-07.md",
}

ALWAYS_SCALE = {"2bit": "mse_opt", "ge_3bit": "amax",
                "why": f"{D['scale']}: mse_opt closes ~55% of the 2->3-bit gap at "
                       f"2-bit on non-rank-bounded models, free; neutral at >=3-bit"}


def policy(g: Geom, target: Optional[float] = None) -> dict:
    fam = g.family
    tier = _tier_for_target(target)
    flags: list[str] = []
    rationale: list[str] = []

    # ---- universal ship floor ------------------------------------------------
    ship = {
        "default_kv_type": "tq4_0 / tq4_0 (K/V) pure",
        "per_layer_overrides": None,
        "scale": ALWAYS_SCALE,
        "norm": "robust",
        "qdim_gate": "on",
    }
    rationale.append(f"[{D['ladder']} rule #1] TQ4 pure first — measured at/near the "
                     f"quality frontier on all six models; always the ship floor.")
    rationale.append(f"[{D['scale']}] scale = mse_opt @2-bit / amax @>=3-bit; "
                     f"[{D['abl']}] --norm robust; Q-dim gate ON (auto-promotes unsafe tiers).")

    aggressive: dict = {}

    # ---- family branches -----------------------------------------------------
    if fam == "no-kv-cache":
        flags.append("N/A: encoder architecture — no autoregressive KV cache to quantize")
        ship["default_kv_type"] = "n/a"
        return _assemble(g, fam, ship, {}, flags, rationale)

    if fam == "UNKNOWN":
        flags.append("UNVALIDATED: untested architecture — TQ4/TQ4 pure is the safe "
                     "default; MEASURE (kvimp + KLD ladder) before trusting any sub-TQ4 plan")
        rationale.append(f"[{D['ladder']} rule #1] default to TQ4 pure on unknown geometry; "
                         f"do not fabricate an aggressive tier we never measured.")
        aggressive = {"note": "none — measure first"}
        return _assemble(g, fam, ship, aggressive, flags, rationale, target, tier)

    # rank-bounded status drives the aggressive envelope for attention families
    if fam in ("shared-KV-MQA", "shared-KV-GQA"):
        if g.rank_bounded_all_owners:                       # E2B: every owner bounded
            flags.append(f"rank-bounded ALL owners (q_dim {g.q_dim} < head_dim "
                         f"{g.head_dim_local}/{g.head_dim_global}) — NEVER sub-3.5-bpw "
                         f"mixed plans; V is NOT free at 2-bit (cascades via residual)")
            aggressive = {
                "tier_3to4bpw": "raw TQ3/TQ3 (set LEANKV_NO_QDIM_GATE=1; else the gate "
                                "auto-promotes to TQ4) — E2B raw TQ3 KLD 0.272; "
                                "OR asymmetric TQ4-K/TQ3-V (0.161, the legit middle rung)",
                "sub_3bpw": "NO-GO — every 2-bit lever fails closed-loop "
                            f"({D['perchan']}); ships TQ4 / gated-TQ3 definitively",
                "menu_item": "generic sparse-outlier side-channel (~1-2% fp16, +0.2-0.4 bpe) "
                             f"on the current Hadamard codec ({D['perchan']}) if 2-bit ever pursued",
            }
            rationale.append(f"[{D['ladder']}][{D['perchan']}] E2B-class (fully rank-bounded, "
                             f"shared-KV MQA): mixed sub-3.5 plans self-sabotage via tq2 floors; "
                             f"uniform TQ4 dominates. 2-bit exhaustively NO-GO.")
        else:                                               # E4B: only globals bounded
            flags.append(f"rank-bounded GLOBALS only ({g.global_layers}); locals free "
                         f"(q_dim {g.q_dim} > local head_dim {g.head_dim_local})")
            aggressive = {
                "tier_3to4bpw": "TQ3 gated — the Q-dim gate does a PARTIAL promotion "
                                "(K adaptive on the rank-bounded globals / V tq3_0 on locals): "
                                "0.118 KLD @ 26.5 MiB, a free hybrid tier that beats raw TQ3; "
                                "OR A1R@3.5 reuse-weighted budget plan (0.129)",
                "sub_3bpw": "unsafe on the globals — do not go below TQ3",
            }
            rationale.append(f"[{D['ladder']}] E4B-class: wider model, locals escape the "
                             f"rank trap; only the {len(g.global_layers)} globals stay bounded; "
                             f"the gate promotes ONLY those (first non-all-or-nothing gate result).")
        # shared-KV reuse lever (both)
        flags.append(f"shared-KV: {g.shared_kv_layers}/{g.n_layer} layers reuse an earlier "
                     f"cache — reuse-weighted allocation (A1R) wins when a sub-4bpw budget "
                     f"is wanted (measured -42% E2B / -84% E4B vs A1)")
        aggressive["reuse_budget_plan"] = (
            "kvimp pass (runtime) -> kv_bit_allocator.py kv_stats.json --arm A1R "
            "--norm robust --bpw <target> --bmax 4 --emit-types plan.types  "
            f"[{D['e2b']}: reuse is the only signal to beat A1]")
        rationale.append(f"[{D['e2b']}][{D['compiler']}] reuse_count>1 owners -> A1R "
                         f"reuse-weighted bits (validated 2/2: E2B -42%, E4B -84% KLD).")

    elif fam == "conv-hybrid":
        flags.append(f"conv-hybrid: {g.n_kv_owning}/{g.n_layer} attention layers grow KV; "
                     f"the rest are short-conv (fixed state, zero KV growth) — "
                     f"kv_bypass {g.kv_bypass_fraction:.0%}; spend budget on the attn tensors only")
        aggressive = {
            "tier_3to4bpw": "TQ3/TQ3 usable (LFM2.5-1.2B 0.060, 8B-A1B 0.155 — beats most "
                            "models' TQ4)",
            "sub_3bpw": "TQ2 REJECTED — the 6 attention layers carry all retrieval; "
                        "hybrid dilution does NOT buy Qwen-style TQ2 survival "
                        "(1.2B 0.535 / 8B 0.837)",
        }
        rationale.append(f"[{D['entropy']}][{D['compiler']}] LFM2 conv-hybrid: no rank slack "
                         f"(r99 ~89-94% fill), sink at the LAST attn layer (robust norm catches "
                         f"it), reuse inert. Plain A1+robust on the attn tensors; TQ2 rejected.")

    elif fam == "SSM-hybrid":
        flags.append(f"SSM-hybrid: {g.n_kv_owning}/{g.n_layer} attention layers grow KV; "
                     f"the rest are mamba/deltanet (fixed recurrent state) — kv_bypass "
                     f"{g.kv_bypass_fraction:.0%}. Extreme dilution: avoidance is built in.")
        if g.rank_bounded_any:
            tq2_caveat = (f"CAVEAT: these attn layers ARE rank-bounded "
                          f"(q_dim {g.q_dim} < head_dim {g.head_dim_local}), so the Q-dim gate "
                          f"WOULD promote TQ2->TQ4 unless LEANKV_NO_QDIM_GATE=1; the doc's "
                          f"TQ2-survival was measured ungated. Treat as measure-before-ship.")
            flags.append(f"NOTE: attn layers rank-bounded (q_dim {g.q_dim} < {g.head_dim_local}) "
                         f"— TQ2 exception is dilution-driven, not rank-driven; verify per checkpoint")
        else:
            tq2_caveat = (f"This checkpoint is NOT rank-bounded (q_dim {g.q_dim} == head_dim "
                          f"{g.head_dim_local}), so the Q-dim gate leaves TQ2 as-requested — the "
                          f"dilution exception applies directly. Still measure-before-ship per "
                          f"checkpoint.")
        aggressive = {
            "tier_3to4bpw": "TQ3/TQ3 (plain TQ already near-optimal on this family)",
            "sub_3bpw": "TQ2 is the ONE documented exception where 2-bit survives (Qwen-3.5 "
                        f"hybrid, +2.6% only — {D['compiler']}). {tq2_caveat}",
        }
        rationale.append(f"[{D['compiler']}][{D['ladder']}] Qwen-3.5 SSM-hybrid: mamba layers "
                         f"bypass KV, only a few attn layers hold cache; TQ4 pure ships, TQ2 is "
                         f"the sole cross-program 2-bit survivor (extreme dilution). "
                         f"NOTE: the docs measured an 8/36-layer Qwen checkpoint; this file is "
                         f"{g.n_kv_owning}/{g.n_layer} of the same family — same policy, re-measure "
                         f"the exact checkpoint before trusting sub-TQ4.")

    elif fam in ("dense-MHA", "dense-GQA"):
        if g.rank_bounded_any:
            flags.append(f"rank-bounded (q_dim {g.q_dim} < head_dim {g.head_dim_local}) — "
                         f"keep sub-TQ4 to uniform tiers, not mixed plans")
        aggressive = {
            "tier_3to4bpw": "TQ3/TQ3 usable (Hadamard rotation + scalar TQ ladder; TQ3 the "
                            "standard usable rung)",
            "sub_3bpw": "TQ2 dead (scalar Lloyd-Max wall). mse_opt scale recovers ~half the "
                        f"2-bit gap on non-rank-bounded dense models ({D['scale']}) but still "
                        f"below TQ3; VQ is the only sub-3-bit research path.",
        }
        rationale.append(f"[{D['compiler']}][{D['abl']}] dense attention: redundancy is coherent "
                         f"channel outliers -> Hadamard + scalar TQ ladder (TQ4 lossless, TQ3 "
                         f"usable, TQ2 dead). Importance signal proved redundant (corr 0.82 with "
                         f"variance); ship A1 uniform+outlier under a budget.")
        if g.name and "gemma-3-4b" in g.name.lower() or "gemma-3-4b" in os.path.basename(g.path).lower():
            rationale.append(f"[{D['ladder']}] gemma-3-4B legacy production was A1@3.0 (0.392/77%) "
                             f"— superseded by rule #1 (TQ4 pure), which clears it by a wide margin.")

    return _assemble(g, fam, ship, aggressive, flags, rationale, target, tier)


def _chosen_types(g: Geom, tier: str) -> tuple[str, str, list[str]]:
    """(ktype, vtype, warnings) for the tier the operator asked for."""
    warn: list[str] = []
    if tier == "tq4":
        return "tq4_0", "tq4_0", warn
    if tier == "tq3":
        if g.family in ("dense-MHA", "dense-GQA", "conv-hybrid", "SSM-hybrid", "shared-KV-GQA"):
            return "tq3_0", "tq3_0", warn
        if g.family == "shared-KV-MQA":
            warn.append("TQ3 on a fully rank-bounded model: the Q-dim gate auto-promotes to "
                        "TQ4 unless LEANKV_NO_QDIM_GATE=1; asym TQ4-K/TQ3-V (0.161) is the "
                        "safer middle rung.")
            return "tq3_0", "tq3_0", warn
        warn.append("TQ3 not validated on this family — measure first.")
        return "tq4_0", "tq4_0", warn
    # tq2
    if g.family == "SSM-hybrid":
        warn.append("TQ2 is the Qwen-hybrid exception; rank-bounded attn means the gate "
                    "promotes unless LEANKV_NO_QDIM_GATE=1. Measure before ship.")
        return "tq2_0", "tq2_0", warn
    warn.append(f"TQ2 is NOT viable on {g.family} (measured dead outside the Qwen-hybrid "
                f"exception) — refusing; emitting the TQ4 ship floor instead.")
    return "tq4_0", "tq4_0", warn


def _assemble(g, fam, ship, aggressive, flags, rationale, target=None, tier="tq4"):
    kt, vt, warn = _chosen_types(g, tier) if fam not in ("no-kv-cache", "UNKNOWN") \
        else ("tq4_0", "tq4_0", [])
    flags = flags + warn
    return {
        "model": os.path.basename(g.path),
        "arch": g.arch,
        "family": fam,
        "features": {
            "n_layer": g.n_layer, "n_embd": g.n_embd, "n_head": g.n_head,
            "q_dim": g.q_dim, "head_dim_local": g.head_dim_local,
            "head_dim_global": g.head_dim_global,
            "n_head_kv_attn": g.n_head_kv_attn, "mqa_ratio": g.mqa_ratio,
            "kv_owning_layers": f"{g.n_kv_owning}/{g.n_layer}",
            "global_layers": g.global_layers or None,
            "rank_bounded": (f"ALL owners" if g.rank_bounded_all_owners else
                             (f"{len(g.rank_bounded_layers)} layers "
                              f"{g.rank_bounded_layers}" if g.rank_bounded_any else "none")),
            "shared_kv_layers": g.shared_kv_layers or None,
            "kv_bypass_fraction": g.kv_bypass_fraction,
            "rope_freq_base": g.rope_freq_base,
            "rope_freq_base_swa": g.rope_freq_base_swa,
        },
        "ship_config": ship,
        "aggressive_config": aggressive,
        "chosen_tier": {"target_bpw": target, "tier": tier, "k": kt, "v": vt},
        "flags": flags,
        "rationale": rationale,
    }


# ---------------------------------------------------------------------------
# emit a LEANKV_KV_PLAN .types file for the chosen tier
# ---------------------------------------------------------------------------

def emit_plan(g: Geom, decision: dict, path: str) -> None:
    kt = decision["chosen_tier"]["k"]
    vt = decision["chosen_tier"]["v"]
    tier = decision["chosen_tier"]["tier"]
    lines = [
        f"# leankv kv plan | {os.path.basename(g.path)} | family {g.family} | tier {tier}",
        f"# ship floor = TQ4/TQ4 pure; scale mse_opt@2b/amax@>=3b; --norm robust; Q-dim gate ON",
        f"# emitted K={kt} V={vt} on the {g.n_kv_owning} KV-owning layer(s); "
        f"non-KV layers keep engine defaults",
        "# <layer> <ktype> <vtype>",
    ]
    for il in g.kv_owning_layers:
        lines.append(f"{il} {kt} {vt}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote plan ({len(g.kv_owning_layers)} layers, {kt}/{vt}) -> {path}")


# ---------------------------------------------------------------------------
# human-readable printer
# ---------------------------------------------------------------------------

def print_decision(d: dict) -> None:
    print(f"\n=== {d['model']}  [{d['arch']}]  family: {d['family']} ===")
    f = d["features"]
    print(f"  geometry : n_layer={f['n_layer']} n_embd={f['n_embd']} n_head={f['n_head']} "
          f"q_dim={f['q_dim']} head_dim={f['head_dim_local']}"
          f"{'/'+str(f['head_dim_global']) if f['head_dim_global'] else ''} "
          f"mqa={f['mqa_ratio']}")
    print(f"  kv       : owning={f['kv_owning_layers']} bypass={f['kv_bypass_fraction']} "
          f"shared_kv={f['shared_kv_layers']} rank_bounded={f['rank_bounded']}")
    if f["global_layers"]:
        print(f"  globals  : {f['global_layers']}  (rope fb={f['rope_freq_base']} "
              f"swa={f['rope_freq_base_swa']})")
    s = d["ship_config"]
    print(f"  SHIP     : {s['default_kv_type']}  | scale {s['scale']['2bit']}@2b/"
          f"{s['scale']['ge_3bit']}@>=3b | norm {s['norm']} | gate {s['qdim_gate']}")
    ct = d["chosen_tier"]
    print(f"  CHOSEN   : tier={ct['tier']} (target_bpw={ct['target_bpw']}) -> K={ct['k']} V={ct['v']}")
    if d["aggressive_config"]:
        print("  AGGRESSIVE:")
        for k, v in d["aggressive_config"].items():
            print(f"     - {k}: {v}")
    if d["flags"]:
        print("  FLAGS:")
        for fl in d["flags"]:
            print(f"     ! {fl}")
    print("  RATIONALE:")
    for r in d["rationale"]:
        print(f"     - {r}")


# ---------------------------------------------------------------------------
# VALIDATE — reproduce every measured shipping config from the doc
# ---------------------------------------------------------------------------
#
# Ground truth = docs/leankv-tq-production-ladder-2026-07.md decision-gate table
# (+ family policy for the two models not in that 4-row table).

DOC_TRUTH = {
    # substring -> dict of DOCUMENTED ground-truth fields. Ship is uniformly TQ4
    # (ladder rule #1), so matching ship alone is trivial; the discriminating
    # assertions are family / n_kv_owning / rank-bounded pattern / TQ2-refusal.
    # n_kv_owning: shared-KV Gemma = n_layer - shared_kv_layers; hybrids = #attn.
    "gemma-4-E2B":   dict(ship="tq4_0/tq4_0 pure", family="shared-KV-MQA",
                          n_kv_owning=15, rb="all", refuses_tq2=True,
                          aggr_has="raw TQ3",
                          note="ladder: 0.0965 KLD / 88.98%"),
    "gemma-4-E4B":   dict(ship="tq4_0/tq4_0 pure", family="shared-KV-GQA",
                          n_kv_owning=24, rb="globals", refuses_tq2=True,
                          aggr_has="TQ3",
                          note="ladder: 0.0477 KLD / 91.35%; globals-only rb"),
    "LFM2.5-1.2B":   dict(ship="tq4_0/tq4_0 pure", family="conv-hybrid",
                          n_kv_owning=6, rb="no", refuses_tq2=True,
                          aggr_has="TQ3",
                          note="ladder: 0.0258 KLD / 91.68%; TQ2 rejected"),
    "LFM2.5-8B-A1B": dict(ship="tq4_0/tq4_0 pure", family="conv-hybrid",
                          n_kv_owning=6, rb="no", refuses_tq2=True,
                          aggr_has="TQ3",
                          note="ladder: 0.0911 KLD / 85.85%; TQ2 rejected"),
    "gemma-3-4b":    dict(ship="tq4_0/tq4_0 pure", family="dense-GQA",
                          n_kv_owning=None, rb="no", refuses_tq2=True,
                          aggr_has="TQ3",
                          note="ladder rule #1; legacy A1@3.0 superseded"),
    "Qwen3.5":       dict(ship="tq4_0/tq4_0 pure", family="SSM-hybrid",
                          n_kv_owning=None, rb=None, refuses_tq2=False,
                          aggr_has="TQ2",
                          note="SSM-hybrid; TQ2 = the Qwen-hybrid exception"),
}


def _doc_for(fname: str):
    for key, val in DOC_TRUTH.items():
        if key.lower() in fname.lower():
            return key, val
    return None, None


def _emitted_ship_str(d: dict) -> str:
    t = d["ship_config"]["default_kv_type"]
    return "tq4_0/tq4_0 pure" if t.startswith("tq4_0 / tq4_0") else t


def _rb_pattern(g: Geom) -> str:
    if g.rank_bounded_all_owners:
        return "all"
    if g.rank_bounded_any:
        return "globals" if g.global_layers and set(g.rank_bounded_layers) <= set(g.global_layers) else "some"
    return "no"


def validate(models_dir: str) -> int:
    """Assert the DISCRIMINATING fields, not just the (uniform, trivial) ship
    string: family, KV-owning-layer count, rank-bounded pattern, and the TQ2
    refusal at --target-bpw 2. A prober that always says TQ4 passes the ship
    check but fails these."""
    files = sorted(f for f in os.listdir(models_dir) if f.endswith(".gguf"))
    rows = []
    n_fail = 0
    for fn in files:
        key, T = _doc_for(fn)
        if key is None:
            continue  # not a measured model (e.g. bge-* encoders)
        g = probe(os.path.join(models_dir, fn))
        d = policy(g)
        d2 = policy(g, target=2.0)          # to exercise the TQ2 refusal path
        checks: list[tuple[str, bool, str]] = []

        checks.append(("ship", _emitted_ship_str(d) == T["ship"], _emitted_ship_str(d)))
        checks.append(("family", g.family == T["family"], g.family))
        if T.get("n_kv_owning") is not None:
            checks.append(("kv_own", g.n_kv_owning == T["n_kv_owning"],
                           f"{g.n_kv_owning}"))
        if T.get("rb") is not None:
            checks.append(("rank_bnd", _rb_pattern(g) == T["rb"], _rb_pattern(g)))
        # TQ2 refusal: refusing families floor to tq4_0 at target 2; Qwen keeps tq2_0
        tq2_refused = d2["chosen_tier"]["k"] == "tq4_0"
        checks.append(("tq2_refuse", tq2_refused == T["refuses_tq2"],
                       "refused" if tq2_refused else "allowed"))
        # aggressive menu carries the expected tier keyword
        aggr_txt = json.dumps(d["aggressive_config"])
        checks.append(("aggr", T["aggr_has"].lower() in aggr_txt.lower(),
                       T["aggr_has"]))

        failed = [c for c in checks if not c[1]]
        n_fail += len(failed)
        status = "PASS" if not failed else "FAIL:" + ",".join(c[0] for c in failed)
        rows.append((fn.replace("-Q4_K_M.gguf", "").replace(".gguf", ""),
                     g.family, f"{g.n_kv_owning}/{g.n_layer}", _rb_pattern(g),
                     "y" if tq2_refused else "n", status))

    hdr = ("model", "family", "kv_own", "rb", "tq2✗", "checks(family/kv_own/rb/refuse/aggr)")
    widths = [max(len(hdr[i]), *(len(str(r[i])) for r in rows)) for i in range(len(hdr))] \
        if rows else [len(h) for h in hdr]
    def line(cols):
        return "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(cols))
    print("\n" + "=" * 100)
    print("VALIDATION — discriminating fields vs documented ground truth "
          "(family / KV-owning count / rank-bounded / TQ2-refusal / aggressive menu)")
    print("=" * 100)
    print(line(hdr)); print("  ".join("-" * w for w in widths))
    for r in rows:
        print(line(r))
    print()
    if n_fail:
        print(f"*** {n_fail} FIELD CHECK(S) FAILED across {len(rows)} models — policy miscoded. ***")
        return 1
    print(f"All {len(rows)} measured models: every discriminating field "
          f"(family, KV-owning count, rank-bounded pattern, TQ2 refusal, aggressive "
          f"menu) matches the measured ground truth. Non-trivial validation.")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", nargs="?", help="path to a model.gguf")
    ap.add_argument("--target-bpw", type=float, default=None,
                    help="nominal bpw the operator wants (>=3.8 TQ4 ship, 2.8-3.8 TQ3, "
                         "<2.8 TQ2). Default: ship floor (TQ4).")
    ap.add_argument("--emit-plan", default=None, metavar="FILE",
                    help="write a LEANKV_KV_PLAN .types file for the chosen tier")
    ap.add_argument("--json", action="store_true", help="print the decision dict as JSON")
    ap.add_argument("--validate", action="store_true",
                    help="self-check: reproduce every measured model's documented ship config")
    ap.add_argument("--models-dir", default="/home/junc/rikuri/rikurinode/models",
                    help="directory of *.gguf for --validate")
    args = ap.parse_args()

    if args.validate:
        sys.exit(validate(args.models_dir))

    if not args.model:
        ap.error("give a model.gguf, or use --validate")

    g = probe(args.model)
    d = policy(g, args.target_bpw)
    if args.json:
        print(json.dumps(d, indent=2))
    else:
        print_decision(d)
    if args.emit_plan:
        if g.family in ("no-kv-cache", "UNKNOWN"):
            print(f"[refusing to emit plan for family {g.family}]", file=sys.stderr)
        else:
            emit_plan(g, d, args.emit_plan)


if __name__ == "__main__":
    main()
