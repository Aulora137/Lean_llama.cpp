#!/usr/bin/env python3
"""
kv_bit_allocator.py  (v2 — hardened)

Offline per-KV-head bit allocator for TQ-family KV-cache quantization.

Consumes kv_stats.json (from kv_importance_collect.cpp) plus optional per-head
entropy and emits a bit plan that hits a TARGET AVERAGE bpw EXACTLY — so
ablation arms are directly comparable.

--------------------------------------------------------------------------
v2 changes vs v1 (all found by synthetic-stats stress testing, 2026-07-14):

FIX-1  Normalization is GLOBAL, not per-layer. v1 normalized importance/var
       within each layer's kv-head list; with MQA (n_head_kv == 1, i.e.
       Gemma-4 E2B) every list has length 1, so every stat collapsed to 1.0
       and A3 became bit-identical to A1 — the go/no-go experiment would
       have returned a guaranteed false "no-go". Cross-layer signal is now
       preserved: importance is max-normalized globally per kind (k and v
       separately — their units differ), var is kept on its raw common
       scale (k_var and v_var are both cache-element energies, hence
       comparable; a single joint scale factor is applied for numeric
       hygiene only). Legacy behavior available via --norm per-layer.

FIX-2  Per-ELEMENT value. The collector emits channel-SUMMED stats, which
       scale with head_dim (and, for importance, with the GQA group size).
       v1 fed the sums straight into the marginal-gain formula, so a
       512-dim (global) head outranked a 256-dim (local) head with equal
       per-element statistics — a systematic bias toward global layers on
       hybrid-head-dim models. v2 converts to per-element means first.
       Correct math: total head distortion D_h(b) = var_sum * 2^(-2b);
       step gain = var_sum * 2^(-2b) * 3/4; step cost = head_dim bits;
       gain-per-bit = (var_sum/head_dim) * 2^(-2b) * 3/4 — the MEAN, not
       the sum. ("head_dim cancels" in v1 was wrong: it cancels only if
       you divide.)

FIX-3  Fair tie-breaking. v1's heap broke ties by slot index, so at
       fractional targets the "uniform" A1 arm gave all its extra bits to
       the lowest-numbered layers (measured: layers 0-7 at 4 bits, 8-14 at
       3 bits at target 3.5). v2 uses a seeded random tiebreak — uniform in
       expectation, reproducible via --seed.

FIX-4  Budget filling: v1 stopped ("break") the moment the single
       highest-gain slot was unaffordable, even when cheaper slots (smaller
       head_dim / partitions) still fit. v2 skips the unaffordable slot and
       keeps filling ("continue"), so mixed-geometry budgets land tighter.

FIX-5  Validation. v1 silently reported success on out-of-range targets
       (target 1.5 with b_min 2 "achieved 2.0"; target 9.0 with b_max 8
       "achieved 8.0"). Matched bpw is the experimental control — v2 exits
       with an error instead.

NEW-1  A5 (p-RoPE partition) is implemented, not a stub. On layers with
       rope_fraction < 1.0 the K slot splits into a "rope" and a "pass"
       sub-slot (element counts rope_fraction*head_dim and the rest); each
       gets its own bits. Uses per-partition stats (k_var_rope / k_var_pass
       and optionally k_importance_rope / k_importance_pass) when the
       collector provides them; without them A5 falls back to A4 behavior
       with a warning (equal per-element values -> partitioning cannot
       change the answer, so we don't pretend it did).

NEW-2  Plan provenance: the output JSON echoes arm, config, achieved slack,
       and stats-file checksum-ish metadata so ablation runs are auditable.

Schema notes:
  * stats layers may carry "n_head" (query heads). If present, importance
    is de-scaled by the GQA group size exactly; if absent, the group factor
    is assumed constant across layers (it then cancels in normalization)
    and a note is printed.
  * partitioned K entries in the plan are {"rope": b, "pass": b};
    unpartitioned entries remain plain ints (backward compatible).

Value model per slot (K and V independently, per element):

    V_slot = imp_pe_norm ** w_imp          (if use_importance)
           * reuse       ** w_reuse        (if use_reuse)
           * ent_norm                       (if use_entropy)
           * var_pe                         (always; source sigma^2 scale)

Greedy marginal-gain allocation under high-rate R-D  D(b) ~ V * 2^(-2b):
each step buys one bit for every element of the slot with the highest
per-element marginal distortion reduction, until the exact budget is spent.

Ablation arms (flag combinations, unchanged from v1):

    A1  uniform+outlier   use_importance=F use_reuse=F use_entropy=F
    A2  +entropy          use_entropy=T
    A3  +importance       use_importance=T
    A4  A2+A3             use_importance=T use_entropy=T
    A5  +p-RoPE partition A4 + prope_aware=T
    A6  +reuse weighting  A5 + use_reuse=T
"""

from __future__ import annotations
import json, math, argparse, random, sys, heapq
from dataclasses import dataclass
from typing import Optional


@dataclass
class Cfg:
    target_bpw: float = 3.0
    b_min: int = 2
    b_max: int = 8
    use_importance: bool = True
    use_reuse: bool = True
    use_entropy: bool = False
    prope_aware: bool = False
    w_imp: float = 1.0
    w_reuse: float = 1.0
    norm: str = "global"          # "global" (v2) | "robust" (v3) | "per-layer" (v1)
    sink_mult: float = 20.0       # robust: layer is a sink if val >= this x median
    seed: int = 0
    eps: float = 1e-12


@dataclass
class Slot:
    layer: int
    which: str          # "k" or "v"
    kvh: int
    part: str           # "" (whole head) | "rope" | "pass"
    n_elem: int         # elements this slot costs per bit
    value: float        # per-element V_slot
    tie: float = 0.0    # random tiebreak, fixed per slot
    bits: int = 0


def _warn(msg: str) -> None:
    print(f"[kv_bit_allocator] WARNING: {msg}", file=sys.stderr)


def _note(msg: str) -> None:
    print(f"[kv_bit_allocator] note: {msg}", file=sys.stderr)


# ---------------------------------------------------------------------------
# stats -> slots
# ---------------------------------------------------------------------------

def build_slots(stats: dict, cfg: Cfg, entropy: Optional[dict] = None) -> list[Slot]:
    layers = stats["layers"]
    rng = random.Random(cfg.seed)

    have_n_head = all("n_head" in L for L in layers)
    if not have_n_head and cfg.use_importance:
        _note("stats lack 'n_head'; assuming constant GQA group across layers "
              "(group factor then cancels in global normalization)")

    # ---- pass 1: per-element raw stats per (layer, which, kvh, part) -------
    raw = []   # dicts: layer, which, kvh, part, n_elem, imp_pe, var_pe, reuse, ent
    missing_part_stats = 0
    for L in layers:
        layer, hd   = L["layer"], L["head_dim"]
        nkv         = L["n_head_kv"]
        reuse       = max(L.get("reuse_count", 1), 1)
        group       = (L["n_head"] // nkv) if have_n_head else 1
        rope_frac   = L.get("rope_fraction", 1.0)
        ent_l       = (entropy or {}).get(str(layer), {})

        for which, imp_key, var_key in (("k", "k_importance", "k_var"),
                                        ("v", "v_importance", "v_var")):
            imp_list, var_list = L[imp_key], L[var_key]
            ent_list = ent_l.get(which)
            for kvh in range(nkv):
                imp_pe = imp_list[kvh] / (hd * group)      # per-element mean
                var_pe = var_list[kvh] / hd
                ent    = ent_list[kvh] if ent_list else None

                split = (cfg.prope_aware and which == "k" and rope_frac < 1.0)
                if split:
                    rope_n = max(1, round(hd * rope_frac))
                    pass_n = hd - rope_n
                    vr, vp = L.get("k_var_rope"), L.get("k_var_pass")
                    ir, ip = L.get("k_importance_rope"), L.get("k_importance_pass")
                    if vr is None or vp is None:
                        missing_part_stats += 1
                        split = False   # fall back: whole-head slot
                    else:
                        for part, n_el, v_sum, i_sum in (
                                ("rope", rope_n, vr[kvh], ir[kvh] if ir else None),
                                ("pass", pass_n, vp[kvh], ip[kvh] if ip else None)):
                            raw.append(dict(
                                layer=layer, which=which, kvh=kvh, part=part,
                                n_elem=n_el,
                                var_pe=v_sum / n_el,
                                imp_pe=(i_sum / (n_el * group)) if i_sum is not None else imp_pe,
                                reuse=reuse, ent=ent))
                if not split:
                    raw.append(dict(layer=layer, which=which, kvh=kvh, part="",
                                    n_elem=hd, var_pe=var_pe, imp_pe=imp_pe,
                                    reuse=reuse, ent=ent))

    if cfg.prope_aware and missing_part_stats:
        _warn(f"A5 requested but {missing_part_stats} K slots lack per-partition "
              f"stats (k_var_rope/k_var_pass); those slots fall back to whole-head "
              f"allocation — without partition stats A5 degenerates to A4")

    # ---- pass 2: normalization ---------------------------------------------
    if cfg.norm == "per-layer":                       # v1 legacy (buggy on MQA)
        _warn("--norm per-layer reproduces the v1 behavior that erases "
              "cross-layer (and all MQA) signal; use only for regression tests")
        by_layer: dict[tuple, list] = {}
        for r in raw:
            by_layer.setdefault((r["layer"], r["which"]), []).append(r)
        for rs in by_layer.values():
            mi = max(r["imp_pe"] for r in rs) or 1.0
            mv = max(r["var_pe"] for r in rs) or 1.0
            for r in rs:
                r["imp_pe"] /= mi
                r["var_pe"] /= mv
    elif cfg.norm == "robust":                        # v3: rank + sink-exclude
        # v2 global divides by the max, so a single attention-sink layer
        # (layer 0 on Gemma: var/imp ~100-800x every other layer) normalizes
        # every other layer to ~0.001 and the allocator can no longer tell
        # them apart. Fix: (a) detect sink layers by a median multiple and
        # exclude them from the scale, (b) rank-normalize the rest into (0,1]
        # so magnitude outliers can't dominate. Sinks are pinned to 1.0 so
        # they stay protected without crushing everyone else.
        def _sinks(rows, key):
            vals = sorted(r[key] for r in rows)
            med = (vals[len(vals) // 2] if vals else 1.0) or 1.0
            return {(r["layer"], r["which"]) for r in rows
                    if r[key] >= cfg.sink_mult * med}

        def _rank_norm(rows, key):                    # in place; (0,1] by rank
            sinks = _sinks(rows, key)
            body = [r for r in rows if (r["layer"], r["which"]) not in sinks]
            n = len(body) or 1
            for i, r in enumerate(sorted(body, key=lambda r: r[key])):
                r[key] = (i + 1) / n
            for r in rows:
                if (r["layer"], r["which"]) in sinks:
                    r[key] = 1.0
            return sinks

        for which in ("k", "v"):                      # importance: per kind
            sel = [r for r in raw if r["which"] == which]
            if sel:
                s = _rank_norm(sel, "imp_pe")
                if s:
                    _note(f"robust norm: {which}-importance sink layers "
                          f"{sorted({l for l, _ in s})} pinned (excluded from scale)")
        vs = _rank_norm(raw, "var_pe")                # var: one JOINT rank
        if vs:
            _note(f"robust norm: variance sink layers "
                  f"{sorted({l for l, _ in vs})} pinned (excluded from scale)")

    else:                                             # v2 global
        for which in ("k", "v"):                      # importance: per kind
            sel = [r for r in raw if r["which"] == which]
            if sel:
                mi = max(r["imp_pe"] for r in sel) or 1.0
                for r in sel:
                    r["imp_pe"] /= mi
        mv = max(r["var_pe"] for r in raw) or 1.0     # var: one JOINT scale
        for r in raw:
            r["var_pe"] /= mv

    ents = [r["ent"] for r in raw if r["ent"] is not None]
    me = max(ents) if ents else 1.0
    me = me or 1.0

    # ---- pass 3: value + slots ---------------------------------------------
    slots: list[Slot] = []
    for r in raw:
        v = max(r["var_pe"], cfg.eps)
        if cfg.use_importance:
            v *= max(r["imp_pe"], cfg.eps) ** cfg.w_imp
        if cfg.use_reuse:
            v *= r["reuse"] ** cfg.w_reuse
        if cfg.use_entropy and r["ent"] is not None:
            v *= max(r["ent"] / me, cfg.eps)          # higher entropy -> harder
        slots.append(Slot(r["layer"], r["which"], r["kvh"], r["part"],
                          r["n_elem"], v, rng.random()))
    return slots


# ---------------------------------------------------------------------------
# allocation
# ---------------------------------------------------------------------------

def allocate(slots: list[Slot], cfg: Cfg) -> dict:
    if not slots:
        raise SystemExit("no slots built from stats — empty 'layers'?")
    if not (cfg.b_min <= cfg.target_bpw <= cfg.b_max):
        raise SystemExit(f"target bpw {cfg.target_bpw} outside "
                         f"[b_min={cfg.b_min}, b_max={cfg.b_max}] — "
                         f"a matched-bpw plan cannot exist")
    if cfg.b_min < 1:
        raise SystemExit("b_min must be >= 1")

    for s in slots:
        s.bits = cfg.b_min
    total_elems = sum(s.n_elem for s in slots)
    budget      = cfg.target_bpw * total_elems
    spent       = sum(s.bits * s.n_elem for s in slots)

    def gain(s: Slot) -> float:
        # per-element marginal distortion reduction of the (b -> b+1) step
        return s.value * (2.0 ** (-2.0 * s.bits)) * 0.75

    heap = [(-gain(s), s.tie, i) for i, s in enumerate(slots)]
    heapq.heapify(heap)

    while heap and spent < budget:
        _, _, i = heapq.heappop(heap)
        s = slots[i]
        if s.bits >= cfg.b_max:
            continue                              # retired
        if spent + s.n_elem > budget:
            continue                              # FIX-4: skip, try cheaper slots
        s.bits += 1
        spent  += s.n_elem
        heapq.heappush(heap, (-gain(s), s.tie, i))

    achieved = spent / total_elems
    max_slot = max(s.n_elem for s in slots)
    if abs(achieved - cfg.target_bpw) * total_elems > max_slot + 1e-6:
        _warn(f"achieved {achieved:.4f} deviates from target {cfg.target_bpw} "
              f"by more than one max-slot step — check stats geometry")

    plan: dict = {
        "achieved_bpw": achieved,
        "target_bpw": cfg.target_bpw,
        "slack_bit_elems": budget - spent,
        "layers": {},
    }
    for s in slots:
        lp = plan["layers"].setdefault(str(s.layer), {"k": {}, "v": {}})
        if s.part:
            entry = lp[s.which].setdefault(str(s.kvh), {})
            entry[s.part] = s.bits
        else:
            lp[s.which][str(s.kvh)] = s.bits
    return plan


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def report(plan: dict, stats: dict) -> None:
    geom = {str(L["layer"]): L for L in stats["layers"]}
    print(f"{'layer':>5} {'type':>6} {'hd':>4} {'reuse':>5}   k-bits        v-bits")
    for L in sorted(plan["layers"], key=int):
        g = geom[L]
        def fmt(d):
            out = []
            for h in sorted(d, key=int):
                b = d[h]
                out.append(f"{h}:{b['rope']}r/{b['pass']}p" if isinstance(b, dict)
                           else f"{h}:{b}")
            return " ".join(out)
        kind = "glob" if g.get("is_global") else "loc"
        print(f"{L:>5} {kind:>6} {g['head_dim']:>4} {g.get('reuse_count',1):>5}   "
              f"{fmt(plan['layers'][L]['k']):<13} {fmt(plan['layers'][L]['v'])}")
    print(f"achieved {plan['achieved_bpw']:.4f} bpw "
          f"(target {plan['target_bpw']}, slack {plan['slack_bit_elems']:.0f} bit-elems)")


ARMS = {
    "A1":  dict(use_importance=False, use_reuse=False, use_entropy=False),
    # A1R: the post-ablation reuse arm (2026-07). The Gemma 3-4B ablation showed
    # magnitude-importance is redundant with variance (log-corr 0.82, A3 loses
    # to A1 by 18% at 30 sigma) — so the E2B cross-layer-sharing test isolates
    # the reuse signal on top of the WINNING baseline, without importance:
    # value = var * reuse^w_reuse. reuse_count comes from kvimp geometry
    # (E2B: consumers per owned tensor). Use with --norm robust.
    "A1R": dict(use_importance=False, use_reuse=True,  use_entropy=False),
    "A2":  dict(use_importance=False, use_reuse=False, use_entropy=True),
    "A3":  dict(use_importance=True,  use_reuse=False, use_entropy=False),
    "A4":  dict(use_importance=True,  use_reuse=False, use_entropy=True),
    "A5":  dict(use_importance=True,  use_reuse=False, use_entropy=True, prope_aware=True),
    "A6":  dict(use_importance=True,  use_reuse=True,  use_entropy=True, prope_aware=True),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stats", help="kv_stats.json from the collector")
    ap.add_argument("-o", "--out", default="bit_plan.json")
    ap.add_argument("--bpw", type=float, default=3.0)
    ap.add_argument("--bmin", type=int, default=2)
    ap.add_argument("--bmax", type=int, default=8)
    ap.add_argument("--entropy", default=None, help="optional entropy.json")
    ap.add_argument("--arm", choices=sorted(ARMS), default="A4")
    ap.add_argument("--norm", choices=["global", "robust", "per-layer"], default="global",
                    help="'robust' = rank-normalize + exclude attention-sink layers "
                         "(fixes layer-0 domination); 'per-layer' = v1 legacy (buggy on MQA)")
    ap.add_argument("--sink-mult", type=float, default=20.0,
                    help="robust norm: a layer is a sink if its per-kind value "
                         ">= this multiple of the median (default 20)")
    ap.add_argument("--seed", type=int, default=0, help="tiebreak seed")
    ap.add_argument("--w-imp", type=float, default=1.0)
    ap.add_argument("--w-reuse", type=float, default=1.0)
    ap.add_argument("--report", action="store_true", help="print per-layer table")
    ap.add_argument("--emit-types", default=None, metavar="FILE",
                    help="write per-layer '<il> <ktype> <vtype>' plan for LEANKV_KV_PLAN "
                         "(bits map 2/3/4->tq2_0/tq3_0/tq4_0, >=5->q8_0; use --bmax 4 for an "
                         "exact TQ ladder; GQA reduces per-layer via max bits — exact on MQA)")
    args = ap.parse_args()

    cfg = Cfg(target_bpw=args.bpw, b_min=args.bmin, b_max=args.bmax,
              norm=args.norm, sink_mult=args.sink_mult, seed=args.seed,
              w_imp=args.w_imp, w_reuse=args.w_reuse, **ARMS[args.arm])

    stats   = json.load(open(args.stats))
    entropy = json.load(open(args.entropy)) if args.entropy else None
    if entropy is None and cfg.use_entropy:
        _note(f"arm {args.arm} uses entropy but no --entropy file given; "
              f"entropy term is inert for this run")

    slots = build_slots(stats, cfg, entropy)
    plan  = allocate(slots, cfg)
    plan["arm"]    = args.arm
    plan["config"] = {k: v for k, v in vars(cfg).items()}

    json.dump(plan, open(args.out, "w"), indent=2)
    if args.emit_types:
        B2T = {2: "tq2_0", 3: "tq3_0", 4: "tq4_0"}
        def red(entry):
            bs = []
            for _h, b in entry.items():
                bs.append(max(b.values()) if isinstance(b, dict) else b)
            return max(bs)
        lines = [f"# leankv kv plan | arm {args.arm} | target {args.bpw} bpw | achieved {plan['achieved_bpw']:.4f}",
                 "# <layer> <ktype> <vtype>"]
        for L in sorted(plan["layers"], key=int):
            d = plan["layers"][L]
            kt = B2T.get(red(d["k"]), "q8_0")
            vt = B2T.get(red(d["v"]), "q8_0")
            lines.append(f"{L} {kt} {vt}")
        open(args.emit_types, "w").write("\n".join(lines) + "\n")
        print(f"[{args.arm}] wrote type plan -> {args.emit_types}")
    print(f"[{args.arm}] target {args.bpw:.3f} bpw -> achieved "
          f"{plan['achieved_bpw']:.4f} bpw  ({len(slots)} slots)")
    if args.report:
        report(plan, stats)


if __name__ == "__main__":
    main()
