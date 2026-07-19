# The Adaptive KV Compiler — Design Note (2026-07-17)

**Thesis.** There is no standard transformer KV layout anymore — hybrids (Mamba+attn),
MQA with over-provisioned heads, cross-layer KV sharing, sliding-window ring caches,
MatFormer elasticity — and each family rewards a *different* compression method. A
single quantizer cannot win everywhere. The durable system is not a method but a
**compiler**: a prober that reads the architecture, measures what kind of redundancy
the KV actually has, and picks the transform + allocation per layer. In a
non-standardizing industry, that adaptivity is the moat.

This note pins the concept after two measured campaigns (Gemma 3-4B ablation,
Gemma 4 E2B campaign) so the design doesn't have to be re-derived.

## Evidence: three architectures, three different winning levers

| Architecture | KV redundancy lives in… | Winning lever (measured) |
|---|---|---|
| Dense (Mistral 7B) | coherent channel outliers | Hadamard rotation + scalar TQ (TQ4 +0.28% PPL) |
| Qwen 3.5 hybrid (8/36 layers have KV) | the *architecture* — Mamba layers bypass KV | plain TQ already near-optimal; even TQ2 only +2.6%. Nothing to adapt — avoidance is built in |
| Gemma 3-4B (MQA, single-owner) | variance outliers + one sink layer | A1 uniform+outlier + robust norm (importance signal proved **redundant**, corr 0.82 with variance; A3 lost 18% at 30σ) |
| Gemma 4 E2B (MQA, 15-own/20-share, rank-bounded) | **cross-layer reuse** (owner 13 feeds 17 layers) + **rank slack** (global r95 fill 38.3% of 512) | **A1R reuse-weighted allocation: −42% KLD vs A1 at 99σ**; low-rank ladder armed on globals |
| LFM2.5 (conv hybrid, 6 attn layers on both 1.2B and 8B-A1B MoE) | the architecture — conv layers hold fixed 16 KB state, zero KV growth; no rank slack (r99 89–94%) | plain A1+robust on the 6 attn tensors; sink at LAST attn layer on both sizes (14 / 21); entropy converges to A1, importance loses on both (campaigns 2–3). 8B-A1B total footprint ≈ 5.5 GB → the 8 GB SLM thesis holds; KV is not the binding constraint |

Same 3.0 bpw budget, same quantizer family — and the *correct policy differs per
architecture*. Importance helps nowhere tested; reuse is decisive exactly where the
arch shares KV; low-rank pays exactly where head_dim ≫ q_dim.

## Pipeline

```
                 ┌────────────────────────────────────────────┐
  model.gguf ──▶ │ PROBE (kvimp + rank dump, one CPU pass)    │
                 │  geometry: owned/shared, reuse_count,      │
                 │  head_dim vs q_dim, rope_fraction, window  │
                 │  stats: per-head var/importance, sinks     │
                 │  spectrum: per-layer r95/r99 (SVD of K)    │
                 └───────────────┬────────────────────────────┘
                                 ▼
                 ┌────────────────────────────────────────────┐
                 │ FEATURE VECTOR (per layer)                 │
                 │  {type, reuse, rank_fill_95, sink?,        │
                 │   window_capped?, rope_frac, kv_bypass?}   │
                 └───────────────┬────────────────────────────┘
                                 ▼
                 ┌────────────────────────────────────────────┐
                 │ POLICY MATRIX → per-layer plan             │
                 │  transform stack + bit type + codebook     │
                 └───────────────┬────────────────────────────┘
                                 ▼
                    LEANKV_KV_PLAN types file (+ future:
                    per-layer transform/codebook manifest)
```

All three probe stages exist today (`LEANKV_KVIMP`, `LEANKV_CALIBRATION_DUMP` +
`analyze_k_calib.py`, allocator `--norm robust`). What's missing is the policy layer
that turns measurements into a *combined* manifest instead of a bits-only plan.

## Policy matrix (current knowledge)

| Measured condition | Method | Status |
|---|---|---|
| channel outliers, rank_fill high | Hadamard + scalar TQ ladder | shipped (TQ4/TQ3) |
| reuse_count > 1 on owners | reuse-weighted bits (A1R) | **validated 2/2: E2B −42%, E4B −84% KLD** |
| rank_fill_95 ≪ 1 (e.g. globals 38%) | low-rank projection then TQ the residual | **tested 2026-07-19: NO-GO** — rank-224 recon alone costs 5.5× TQ4's entire KLD; energy retention ≠ task information (production ladder doc). Spectra remain diagnostic |
| rank-bounded (q_dim < head_dim) + sub-3.5-bpw budget | **use uniform TQ4 (or raw TQ3), NOT mixed plans** — allocator tq2 floors self-sabotage; Q-dim gate auto-promotion measured correct 3× | production ladder doc |
| sink layer (≥20× median var/imp) | pin to top bits, exclude from norm scale | shipped (`--norm robust`) |
| q_dim < head_dim (rank-bounded) | Q-dim gate: TQ3/TQ2 unsafe → warn/promote | shipped (per-layer-type gate) |
| distinct layer types, few of each | empirical per-type codebooks (Qwen3-4B +1.75 dB) | in-tree, not wired to the plan |
| sub-3-bit target on dense attn | vector quantization (escapes scalar Lloyd-Max wall) | research (TQ2 postmortem proves the scalar limit) |
| window-capped local layers | deprioritize: memory bounded by window, not context | free insight — allocator should know it |
| KV-bypassing layers (Mamba/SSM/conv) | leave alone; spend budget on the few attn layers | validated (Qwen3.5, LFM2.5) |
| high attention entropy | entropy-weighted bits (arms A2/A4) | **tested 2026-07-18: no gain over variance at matched budget** (converges on LFM2.5, trails on Gemma; see campaign 2 doc) — collector kept as diagnostic |
| coarse ladder, few cells | **sweep nominal targets + audit emitted bit-sums** — budget granularity moves KLD more than signal choice | standing methodology rule (campaign 2 controls) |

## Non-goals

- Not a new quantizer. TQ stays the scalar backend; the compiler chooses *what to
  wrap around it* per layer.
- Not auto-tuning at inference time. Probe + policy run offline (minutes, CPU); the
  runtime consumes a static manifest, exactly like `LEANKV_KV_PLAN` today.

## Roadmap hooks (cheapest first)

1. **Entropy emitter in kvimp** — unblocks A2/A4/A5; the only untested signal family.
2. **Wire empirical codebooks into the plan format** (per-layer-type LUT id).
3. **Low-rank stage on E2B global owners** (3 tensors, rank ≈ 224 = r95_max+margin):
   projection at cache-write, lift at attention; TQ the 224-dim residual. The
   measured 38% fill says this beats spending the same bits scalar-quantizing 512 dims.
4. **Policy layer** — a ~200-line `kv_policy.py` that consumes kv_stats + rank report
   and emits the full manifest; the allocator becomes one subroutine of it.
5. Long-context validation harness (E2B Step 5) — the regime where global-layer
   levers dominate.

## Provenance

- Gemma 3-4B ablation: `docs/leankv-kv-importance-ablation-2026-07.md`
- E2B campaign + results: `docs/leankv-e2b-campaign-2026-07.md`
- TQ2 scalar limit: `docs/RESULTS.md` ("TQ2 — Not Viable")
- Qwen 3.5 hybrid friendliness: `docs/RESULTS.md` (hybrid architecture note)
