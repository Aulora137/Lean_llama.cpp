# TurboQuant — Design for KV Cache Quantization

## Overview

TurboQuant (TQ) is a KV cache quantization system that compresses the
attention key cache from 16-bit floats to 3.5-4.5 bits per element with
near-zero quality loss. It combines three techniques:

1. **Hadamard rotation** — spreads outlier energy uniformly across channels
2. **Lloyd-Max codebooks** — optimal scalar quantization for Gaussian data
3. **Per-block scaling** — each 32-element block gets its own scale factor

The result: 3.6-4.6x K cache compression that works across architectures
(Mistral, Llama, Qwen, Gemma) with no per-model training required.

## Why These Three Techniques

### Hadamard Rotation

Raw K vectors have heavy-tailed distributions. Some channels carry 100x
more variance than others. Quantizing directly wastes codebook levels on
channels that rarely use the extremes and clips channels that do.

The Walsh-Hadamard Transform (WHT) is an orthonormal rotation using only
+1/-1 entries. It:

- Preserves L2 norm exactly (no information loss)
- Equalizes per-channel variance from 100:1 spread to ~1.5:1
- Reduces kurtosis from >100 (heavy tails) to ~1.18 (near-Gaussian)
- Costs O(d log d) using the butterfly algorithm — essentially free

After Hadamard, every channel has roughly the same distribution shape.
This means a single codebook design works for all channels — no need for
per-channel calibration or outlier detection at the channel level.

**Constraint**: WHT requires power-of-2 dimensions. head_dim=128 (2^7)
and head_dim=256 (2^8) work directly. Non-power-of-2 dimensions would
need padding or block-wise application.

### Lloyd-Max Codebooks

Given near-Gaussian post-Hadamard data, Lloyd-Max quantization places
codebook levels to minimize mean squared error for that distribution.

The levels are **not** uniformly spaced:

```
TQ2 (4 levels):  {-1.000, -0.300, +0.300, +1.000}
TQ3 (8 levels):  {-1.000, -0.690, -0.394, -0.128, +0.128, +0.394, +0.690, +1.000}
TQ4 (16 levels): {-1.000, -0.860, -0.725, -0.594, -0.467, -0.344, -0.223, -0.104,
                   +0.104, +0.223, +0.344, +0.467, +0.594, +0.725, +0.860, +1.000}
```

The levels concentrate near zero where Gaussian density is highest.
This is mathematically optimal — no other 8-level (or 4-level, or
16-level) symmetric codebook produces lower MSE for Gaussian input.

### Per-Block Scaling

Each TQ block (32 elements) stores one scale factor `d` and the quantized
indices. The scale captures local magnitude:

```
quantize:   index[i] = find_nearest(x[i] / d)
dequantize: x_hat[i] = levels[index[i]] * d
```

The scale is optimized via coordinate descent: assign indices, compute
least-squares optimal `d`, reassign, iterate. This matches production
code and was proven equivalent to attention-aware optimization (see
"What Does Not Work" below).

## Type Specifications

| Type | Bits/elem | Levels | Block size | GGML type |
|------|-----------|--------|------------|-----------|
| TQ4_0 | 4.5 | 16 | 32 | `GGML_TYPE_TQ4_0` |
| TQ3_0 | 3.5 | 8 | 32 | `GGML_TYPE_TQ3_0` |
| TQ2_0 | 2.5 | 4 | 32 | `GGML_TYPE_TQ2_0` |

Each block stores: 1 x fp16 scale + packed indices.

- TQ4: 32 x 4-bit = 16 bytes indices + 2 bytes scale = 18 bytes / 32 elem
- TQ3: 32 x 3-bit = 12 bytes indices + 2 bytes scale = 14 bytes / 32 elem
- TQ2: 32 x 2-bit = 8 bytes indices + 2 bytes scale = 10 bytes / 32 elem

## Auto-Calibration (Phase 7a)

The runtime codebook system fits per-model TQ codebooks from a short
warmup pass on an embedded calibration corpus (651 tokens, multi-domain).

**Pipeline:**

1. On first model load, compute tensor fingerprint (FNV-1a hash)
2. Check `~/.cache/leankv/` for cached codebook
3. If cache miss: run warmup pass, accumulate K-value histograms per layer,
   fit Lloyd-Max codebooks via iterative refinement, save to cache
4. Install fitted codebook levels into the runtime LUT
5. Subsequent loads: instant cache hit, zero overhead

In practice, the fitted codebooks closely match the theoretical Gaussian
defaults for most models. The system exists to catch edge cases where
the post-Hadamard distribution deviates from Gaussian.

**Key files:**

- `ggml/src/ggml-tq-runtime.c/h` — runtime codebook registry
- `src/ggml-tq-calib.c/h` — Lloyd-Max fitting + histogram accumulation
- `src/leankv-calib.cpp/h` — auto-calibration entry point
- `src/leankv-codebook.cpp/h` — fingerprint cache read/write

## Rank-Deficiency Safety

Models where `n_embd / n_head < head_dim` (e.g., Qwen3-4B: 2560/32=80
vs head_dim=128) have structurally rank-deficient K projections. At
load time, the system detects this ratio and auto-promotes TQ2/TQ3
requests to TQ4_0 to prevent quality collapse.

```cpp
float rank_ratio = (float)n_embd / ((float)n_head * (float)head_dim);
if (rank_ratio < 1.0f) {
    // Force TQ4_0 — aggressive quantization unsafe for rank-deficient models
}
```

## Production Tiers

| Tier | Type | Compression | Typical PPL delta | When to use |
|------|------|-------------|-------------------|-------------|
| Default | TQ4_0 | 3.6x | <+0.3% | Always safe |
| Aggressive | TQ3_0 | 4.6x | +1-3% | Long context, memory-constrained |

```bash
# Default — near-lossless on all tested models
llama-server -ctk tq4_0 -ctv f16

# Aggressive — solid quality, maximum compression
llama-server -ctk tq3_0 -ctv f16
```

## What Does Not Work

The following approaches were experimentally tested and ruled out:

### TQ2 (2.5 bits/element)

Four codebook levels cannot preserve dot-product fidelity in dense
attention. This is an information-theoretic limit, not a codebook design
problem. The quality cliff from 8 levels (TQ3) to 4 levels (TQ2) is
fundamental: +25% to +117% PPL degradation on dense models.

### Attention-Aware Scaling

Optimizing the block scale factor `d` for dot-product error instead of
MSE was tested via grid search over 41 scale factors. Result: 0.02 dB
improvement (15.74 vs 15.72 dB SNR). Negligible.

The math explains why: post-Hadamard, Q dimensions are near-i.i.d.
When Q is i.i.d., `E[(q*e)^2] = sigma^2_q * sum(e_i^2)`, which is
exactly proportional to MSE. MSE-optimal scaling IS attention-aware.

### Mixed-Precision Channel Selection (Post-Hadamard)

Allocating more bits to "outlier" channels and fewer to "normal" channels
after Hadamard rotation provides no benefit. Hadamard equalizes variance
so effectively (max/mean ~1.5, kurtosis ~1.18) that no meaningful outlier
structure remains. The PPL improvement of mixed TQ3+TQ2 (2.75 bpe) over
pure TQ2 (2.5 bpe) comes entirely from the higher average bit rate, not
from smart channel selection.

### Static W_K Variance for Adaptive Policy

Using W_K weight row norms to predict per-layer quantization sensitivity
works on Mistral but fails on Llama 3-8B (flat W_K variance, yet worst
TQ sensitivity) and Qwen3-8B. The sensitivity mechanism on these models
is invisible to static weight analysis. Runtime calibration (measuring
actual K-vector distributions) is the path forward.

## Architecture

```
User request:  -ctk tq4_0

Model load:
  1. Detect head_dim, n_embd/n_head ratio
  2. Apply rank-deficiency safety (promote if ratio < 1.0)
  3. Auto-calibrate codebook (cache hit or warmup pass)
  4. Allocate KV cache with TQ4 K tensors + F16 V tensors

Each attention layer forward pass:
  1. Compute K = W_K * hidden_state
  2. Apply Hadamard rotation: K_rot = WHT(K)
  3. Quantize: K_q = TQ4_quantize(K_rot, codebook)
  4. Store K_q in KV cache (4.5 bits/element)

Each attention score computation:
  1. Load K_q from cache
  2. Dequantize: K_hat = TQ4_dequantize(K_q, codebook)
  3. Compute score = Q * K_hat^T (via IQK fused kernel)
  4. Continue with softmax, V multiply
```

The IQK (ik_llama.cpp) fused Flash Attention kernels perform dequantize
and dot-product in a single pass, avoiding materialization of the full
fp32 K matrix. TQ4 and TQ3 have dedicated kernel paths for both x86
(AVX2/AVX512) and ARM (NEON) backends, plus CUDA vec_dot_KQ kernels.

## Key Files

| File | Purpose |
|------|---------|
| `ggml/src/ggml-tq.c` | TQ quantize/dequantize/vec_dot implementations |
| `ggml/src/ggml-tq-runtime.c/h` | Runtime codebook LUT registry |
| `ggml/src/ggml-tq-outlier.c/h` | Outlier detection (experimental) |
| `src/ggml-tq-calib.c/h` | Lloyd-Max fitting |
| `src/leankv-calib.cpp/h` | Auto-calibration pipeline |
| `src/leankv-codebook.cpp/h` | Fingerprint cache |
| `ggml/src/iqk/fa/iqk_fa_templates.h` | IQK Flash Attention helpers (TQ4/TQ3/TQ2) |
| `ggml/src/iqk/iqk_gemm_legacy_quants.cpp` | IQK mul_mat kernels (TQ4/TQ3/TQ2) |
| `ggml/src/ggml-cuda/fattn-vec-common.cuh` | CUDA Flash Attention vec_dot_KQ |
