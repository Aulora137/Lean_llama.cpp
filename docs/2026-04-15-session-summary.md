# LeanKV Session Summary — 2026-04-15

## Work Completed

### 1. Runtime TQ Codebook Infrastructure (Commit 1)
Added a runtime-configurable codebook system (`ggml-tq-runtime.c/h`) that replaces
hardcoded Gaussian Lloyd-Max levels with per-model fitted LUTs. Dequant paths in
`ggml-tq.c`, IQK GEMM, and FA helpers all read from the runtime registry.

### 2. K-Cache Range Registry (Commit 2)
After KV cache allocation, each TQ-quantized K layer's `[data, data+nbytes)` range
is registered so FA helpers can resolve a raw row pointer back to its owning layer
and pick up per-layer codebook overrides.

### 3. Empirical Codebook Calibration (Commit 3)
Auto-calibration fits per-model TQ codebooks from a short warm-up pass on an
embedded mini-corpus. Results are cached to disk; subsequent loads are free.
Includes Lloyd-Max fitting, histogram accumulator, and the `leankv_autocalibrate()`
entry point hooked into `llama_init_from_model`.

### 4. Mixed-Precision Noise Simulation + Hadamard Fix (Commit 4)
Added `LEANKV_MIXED_SIM` env-gated noise injection that simulates TQ quantization
error in an F16 K-cache, enabling PPL measurement without FA kernel changes.

**Critical bug found and fixed:** `llama_init_from_model` unconditionally disabled
`k_cache_hadamard` for non-quantized K types (F16). This silently defeated the
noise simulation, producing catastrophic PPL (~55 instead of ~10). The fix checks
for `LEANKV_MIXED_SIM` before overriding.

### 5. TQ Roundtrip Quality Test (Commit 5)
Standalone C test verifying TQ2_0 and TQ3_0 quantize/dequantize quality against
cosine similarity and SNR thresholds.

---

## Key Measurements (Qwen3-8B-Q4_K_M, wikitext-2, ctx=2048)

| Config | Bits/elem | PPL | vs F16 |
|--------|-----------|-----|--------|
| F16 (baseline) | 16.0 | ~7.5 | — |
| TQ4_0 | 4.5 | ~7.53 | +0.03 |
| TQ3_0 | 3.5 | ~10.30 | +2.8 |
| TQ3+TQ2 mixed (32ch/96ch) | 2.75 | ~22.42 | +14.9 |
| TQ2_0 | 2.5 | ~31.34 | +23.8 |

Noise simulation accuracy: TQ3-only sim PPL (10.31) matches real TQ3 PPL (10.30).

---

## Conclusions

### Mixed-Precision Post-Hadamard is Not Viable
The Hadamard transform equalizes per-channel variance so effectively (kurtosis ~1.18,
max/mean ~1.5) that there are **no meaningful outlier channels** in post-Hadamard
K data. Channel-level mixed precision (TQ3 for "outlier" channels, TQ2 for "normal")
provides no principled benefit — the improvement from 2.75 bpe mixed over 2.5 bpe
pure TQ2 comes entirely from the higher average bit rate, not smart channel selection.

PPL at 2.75 bpe (22.42) is 31% better than pure TQ2 (31.34), but still 2.2x worse
than TQ3 (10.30). The quality cliff between 3.5 and 2.5 bits is too steep for
post-Hadamard Gaussian data.

### Production Tier Stack
- **TQ4_0** (4.5 bpe) — default, near-lossless (+0.03 PPL)
- **TQ3_0** (3.5 bpe) — cold tier, modest degradation (+2.8 PPL)
- **TQ2_1** (2.5 bpe) — emergency/per-model only, significant quality loss

### TQ2_1 Status
TQ2_1 has CUDA/Metal kernels and SIMD vec_dot (88% of F16 speed), but no full IQK
kernel fusion — the 128-element blocks don't fit the 32-element `Q_Unpacker` template.
Usable for models where per-model tuning can offset the quality gap.

---

## Branch
`feature/tq2-outlier-tiered` — 5 new commits pushed to origin.
