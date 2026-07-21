# TQ Block-Scale Statistic — amax vs mse_opt (2026-07-21)

**Question.** The shipping TQ codec sets each 32-element block's scale to `d = max|x|`
(amax), inherited from llama.cpp's Q4_0. Is that the right statistic at low bit-rates,
or does a robust/optimal scale (BitNet-style absmean, RMS, or per-block MSE-optimal)
recover quality — for free, no format change?

**Answer: yes, at 2-bit, on non-rank-bounded models — SHIP `mse_opt` bit-width-gated.**
Per-block MSE-optimal scale closes **~54–55% of the TQ2→TQ3 gap, closed-loop**, on both
LFM2.5 sizes, at zero cost (the scale `d` is still stored fp16; only how it's *computed*
changes). It is neutral at 3-bit and does nothing on the pathological rank-bounded E2B.
Recommended default: **`mse_opt` at 2-bit, `amax` at ≥3-bit.**

## The change (uncommitted → this commit)

`ggml/src/ggml-tq.c`: env `LEANKV_TQ_SCALE = amax | absmean | rms | mse_opt` (default
`amax` = current behavior, zero overhead when unset). `tq_block_scale()` dispatches:
- `absmean` → `c·mean|x|` (BitNet recipe), `rms` → `c·sqrt(mean x²)` (per-bit-width c),
- `mse_opt` → 32-point grid search `d = t·amax`, `t∈[0.24,1.01]`, minimizing per-block
  reconstruction SSE against the Lloyd-Max levels.

No block-format change — dequant is unchanged (`d` read from the block as always). The
Python study path was verified **bit-exact** against this C code before any numbers.

## Offline finding (noise-shaping study, arm B)

`mse_opt` was the *dominant, free* lever across 3 archs: closed **53–71%** of the
2→3-bit softmax-KL gap with zero side info (vs the SQuat subspace constraint, which
added only +8–14 points on top and needs per-layer constants). At 4-bit on raw K the
tuned-constant schemes (absmean/rms) collapse (distribution-fragile); `mse_opt`'s
per-block search does not — which is why it, not the BitNet absmean, is the shippable
form. Cost: `mse_opt` is ~93× the amax encode at 2-bit but still 6.1 Melem/s — *faster
than the TQ3 encoder that already ships* (3.9 Melem/s), so it sits inside an accepted
envelope.

## Closed-loop validation (the decider — offline understates 3–5×)

`llama-perplexity`, real cache, KLD vs same-model F16 base, canonical WikiText-2, c=2048:

| Model · tier | amax (ships) | **mse_opt** | reduction | gap closed |
|---|---|---|---|---|
| LFM2.5-1.2B · TQ2 | 0.5352 (65.5%) | **0.2722 (74.6%)** | −49% | **55%** |
| LFM2.5-8B-A1B · TQ2 | 0.8366 (59.4%) | **0.4695 (69.0%)** | −44% | **54%** |
| E2B · TQ2 (rank-bounded) | 1.1664 | 1.0852 | −7% | ~9% |
| E2B · TQ3 (raw) | ~0.272 | 0.2673 | neutral | — |
| LFM2.5-1.2B · TQ3 | 0.0604 (archive) | 0.0644 | +7% (slightly worse) | — |

Two clean facts:
- **Closed-loop matches offline on the non-rank-bounded models** (55%/54% vs the 53–71%
  offline band) — unlike E2B, where offline (47%) collapsed to 7% closed-loop. `mse_opt`
  delivers exactly where 2-bit is viable and can't help where 2-bit is hopeless; those
  are the same fact.
- **Bit-width matters:** big win at 2-bit, **neutral-to-slightly-worse at 3-bit** (amax's
  full range is worth more once you have enough levels; at 2-bit a single outlier hijacks
  amax and `mse_opt`'s robustness wins). Hence the gate.

## Recommendation

**Ship `mse_opt` as the default 2-bit block scale; keep `amax` at ≥3-bit.** Wire the
gate by bit-width in the quantizer (or expose per-tier defaults). Free ~half-a-gap of
2-bit quality on every non-rank-bounded model — the only unambiguous, no-cost, no-format-
change win found in the entire KV-quant program. Fold into the adaptive menu as: *scale =
mse_opt @ 2-bit / amax @ ≥3-bit*, independent of architecture.

Raw logs: `kld_sc_*`, `kld_msD_*` (local, gitignored). Codec change: `ggml/src/ggml-tq.c`
(this commit). Study harness: `kit-v2/scale_study.py`.
