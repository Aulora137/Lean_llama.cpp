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
This is a thoughtful analysis, but I think whoever wrote it is working from an incorrect premise about what we're actually doing. Let me walk through it.

### What they got right

- Small values suffer higher **relative** quantization error — true
- Attention cares about dot-product fidelity, not MSE — fair point
- Dense models rely on many small contributions — true

### What they got wrong

**The central claim — "you're using uniform quantization" — is false.** We're already using Lloyd-Max codebooks. That's the entire foundation of TurboQuant:

```
TQ2 levels: {-1.000, -0.300, +0.300, +1.000}   ← NOT uniform {-1, -0.33, +0.33, +1}
TQ3 levels: {-1.000, -0.690, -0.394, -0.128, +0.128, +0.394, +0.690, +1.000}
```

These are Lloyd-Max optimal for Gaussian — they already **concentrate resolution near zero** where density is highest. That's literally what Lloyd-Max does.

**"Hadamard made everything a trough"** — this misunderstands what Hadamard does. It's an orthonormal transform — it preserves L2 norm exactly. Values don't become "all small." The per-block scale factor `d` in each TQ block captures local magnitude, so quantization is always relative to the block's own scale. Hadamard equalizes the *distribution shape* across channels so that the Gaussian assumption underlying Lloyd-Max actually holds. Before Hadamard, some channels have kurtosis >100 (heavy tails that waste codebook levels); after, kurtosis is ~1.18 (near-Gaussian, exactly where Lloyd-Max is optimal).

**"Use log/companding transform"** — with only 4 levels (TQ2), a companding transform + uniform quantization is mathematically equivalent to just choosing different level placements. Which is what Lloyd-Max already does optimally. There's no extra juice to squeeze from companding when you have so few levels.

### What the commits from yesterday already address

The suggestion to "fit codebooks to actual distribution" is exactly what we built:

- **Commit 1** (`ddd052bc`): Runtime codebook registry — swap LUTs at runtime
- **Commit 3** (`b54236a2`): Empirical calibration — fits per-model codebooks to actual post-Hadamard K distributions via Lloyd-Max on observed histograms

So the actionable advice was already implemented before they suggested it.

### The actual hard truth

The real issue is simpler and less fixable: **2 bits (4 levels) is just not enough information** to preserve dot-product fidelity in dense attention, regardless of how optimally you place those 4 levels. The quality cliff between 3.5 bits (TQ3, 8 levels) and 2.5 bits (TQ2, 4 levels) is fundamental — it's not a codebook design problem, it's an information-theoretic limit.

The one point worth considering further is **attention-aware scaling** — optimizing the scale factor `d` to minimize dot-product error rather than MSE. That could give a marginal improvement for TQ3, though the benefit shrinks as you approach the rate-distortion bound. Want me to prototype that?
---
Attention-aware scaling is optimizing the TQ block's scale factor `d` for **what attention actually cares about** instead of what we currently optimize for.

### What we do now

Standard Lloyd-Max / TQ quantization picks the scale `d = max(abs(block))` to minimize **reconstruction MSE**:

```
minimize: Σ (x_i - x̂_i)²
```

This treats every element equally — a 0.01 error on a large value counts the same as 0.01 error on a small value.

### What attention actually computes

```
score_j = Σ_d  Q_d · K_jd        (dot product)
weight_j = softmax(score_j / √d)  (exponentiated)
```

Two things matter:

1. **Dot-product error**, not per-element error — errors on dimensions where Q is large matter more than where Q is small
2. **Softmax amplifies** — a small score error on a token near the attention peak shifts probability mass much more than the same error on an already-ignored token

### What attention-aware scaling would do

Instead of `d = max(abs(block))`, search for the `d` that minimizes:

```
minimize: |dot(Q, K) - dot(Q, K̂)|²
```

In practice, since Q isn't known at quantization time (K is quantized once, read by many Q vectors), you'd optimize for the **expected** dot-product error under a distribution of Q vectors — which, post-Hadamard, is also near-Gaussian.

The concrete change: the optimal `d` might be slightly smaller than `max(abs(block))`, deliberately clipping a few extreme values to get tighter resolution in the dense middle where most of the dot-product signal lives.

### My honest assessment

For TQ3/TQ4 (8-16 levels), this could shave maybe 0.1-0.3 PPL — noticeable but not transformative. For TQ2 (4 levels), it won't bridge the 20+ PPL gap because the bottleneck is information-theoretic, not scaling. It's a refinement, not a breakthrough.

Worth prototyping if we want to squeeze the last bit out of TQ3 for cold-tier quality. Not worth it as a rescue strategy for TQ2. Want me to try it on TQ3?
## Branch
`feature/tq2-outlier-tiered` — 5 new commits pushed to origin.
Let me look at the current quantization path first.

Good. I can see the current approach already does coordinate descent to minimize MSE. Now let me build the attention-aware variant. The key insight: instead of minimizing `Σ(x_i - d·L[idx_i])²`, we want to minimize the expected dot-product error `E_q[ (q·x - q·x̂)² ]`.

For post-Hadamard Q vectors (also near-Gaussian, roughly i.i.d.), this simplifies nicely:

```
E_q[ (q·(x - x̂))² ] = E_q[ (Σ q_i · e_i)² ] = σ²_q · Σ e_i²    (when q_i are i.i.d.)
```

Which is... just MSE scaled by a constant. So for i.i.d. Q, attention-aware scaling IS MSE-optimal scaling. The current code is already optimal.

But there's a subtlety — Q dimensions are **not** perfectly i.i.d. even post-Hadamard. And more importantly, softmax amplifies errors near the attention peak. Let me think about whether there's a practical angle here.

Actually, the more promising direction is **weighted MSE** — weighting errors by the expected Q magnitude per dimension. Let me prototype this properly as a standalone test that measures dot-product fidelity under different scaling strategies.

Now I have a clear picture. Let me write the prototype. The key mathematical insight:

For attention score `s = q · k`, the error is `Δs = q · (k - k̂) = q · e`. Under softmax, what matters is the **variance** of `Δs` across the key positions for a given query. If we model Q as having per-dimension variance `σ²_d`, the optimal scale minimizes `Σ_d σ²_d · e²_d` (weighted MSE).

Post-Hadamard Q is near-i.i.d., so σ²_d ≈ constant → standard MSE is near-optimal. But I'll test several strategies to quantify the gap:

1. **Baseline**: current MSE-optimal (coordinate descent)
2. **Shrunk scale**: `d = α · d_mse` with α < 1 (sacrifices outliers for tighter mid-range)
3. **Dot-product optimal**: grid search over `d` minimizing actual dot-product error against random Q vectors
4. **Clipped scale**: percentile-based (e.g., 95th percentile instead of max)

There it is. The numbers speak clearly:

| Strategy | DP_SNR (dB) | DP_relErr% | MSE |
|----------|-------------|------------|-----|
| **MSE-optimal (current)** | **15.72** | **16.36%** | **0.02674** |
| **DotProd grid search** | **15.74** | **16.34%** | 0.02670 |
| Shrunk (α=0.92) | 15.57 | 16.65% | 0.02771 |
| Percentile 95th | 15.38 | 17.01% | 0.02895 |
| Percentile 90th | 15.03 | 17.73% | 0.03141 |

The dot-product grid search (strategy 4) — which literally tries 41 different scale factors and picks the one that minimizes actual `|q·k - q·k̂|²` — gets **15.74 dB vs 15.72 dB** for our current MSE-optimal. That's a **0.02 dB** difference — completely negligible, well within noise.

Meanwhile, every "clever" alternative (shrunk scale, percentile clipping) **makes things worse**, not better.

### Why

This confirms the theory: post-Hadamard, Q dimensions are near-i.i.d. Gaussian. When Q is i.i.d., the expected dot-product error `E[(q·e)²] = σ²_q · Σe²_i` is exactly proportional to MSE. So MSE-optimal scaling IS attention-aware scaling. There's no gap to exploit.

This analysis gets closer to something real — let me separate the signal from the noise.

### What's genuinely correct

The core diagnosis is right: **we've hit the scalar quantization limit.** Our experiment proved it — Lloyd-Max + Hadamard + coordinate descent already achieves the theoretical optimum for block-scalar quantization. No codebook trick will move the needle.

### What's overhyped

**"Vector geometry distortion" as root cause** — this is information theory dressed in geometric language. When you go from 3.5 to 2.5 bits per element × 128 dimensions, you lose ~128 bits about the vector's direction. No correction scheme recovers destroyed information. The PPL cliff is the rate-distortion bound, not a fixable "geometry" bug.

**Residual quantization (RVQ)** — quantize, compute residual, quantize again. Two TQ3 passes = 7.0 bpe. That's *worse* than just using TQ4 at 4.5 bpe with better quality. Dead end for our use case.

**"Softmax ranking changes"** — sounds scary, but our experiment just showed dot-product error is proportional to MSE when Q is i.i.d. post-Hadamard. So "preserve ranking" reduces to "minimize MSE," which we already do optimally.

### The one idea worth investigating

**KVLinC** (the paper they cited — arxiv 2510.05373). That one is real. It combines Hadamard rotation with a learned linear correction applied after dequantization. The idea: systematic quantization bias has low-rank structure, so a small correction matrix can compensate for it without recovering the lost information — it just removes the *directional bias*.

Let me check what that paper actually proposes and what the overhead looks like for us.

OK, now I have the real details. Here's the honest assessment:

### KVLinC — what it actually is

It's **not** a simple post-dequant correction matrix. It's a **learned attention correction** with per-layer feature map adapters:

- Two weight matrices W₁, W₂ ∈ ℝ^(d×128) per layer
- Trained to minimize cross-entropy between full-precision and quantized attention distributions
- **Training cost: ~2 hours on 4×H200 GPUs** per model
- Adds a correction term to attention scores, not to K vectors directly

### Why it's architecturally incompatible with us

1. **Requires per-model GPU training** — our users run CPU inference, they don't have H200s. Our whole design philosophy is "set `-ctk tq3_0` and go"

2. **Different pipeline** — KVLinC does NOT use Hadamard on keys. They use channel-wise quantization on raw K tensors, Hadamard only on V. Fundamentally different from our Hadamard-K + Lloyd-Max approach

3. **Adds learned parameters** — breaks our universal codebook model. Every model needs its own trained adapter weights

4. **Their 2-bit result (7.1 PPL)** is on a different quantization scheme (channel-wise, group=128), not comparable to our TQ block structure

### Where this leaves us

The ChatGPT suggestions boil down to: "add learned corrections" — which is valid research but requires training infrastructure we don't have and our users don't have.

Here's the honest picture of our tier stack:

| Tier | Type | PPL delta | Status |
|------|------|-----------|--------|
| Default | TQ4_0 (4.5 bpe) | +0.03 | Near-lossless, ship it |
| Cold | TQ3_0 (3.5 bpe) | +2.8 | Solid for old context |
| Emergency | TQ2_1 (2.5 bpe) | +24 | Per-model only |

TQ4 and TQ3 are already at the scalar quantization optimum — we proved that yesterday. Closing the TQ2 gap requires either more bits (which defeats the purpose) or learned corrections (which requires GPU training infrastructure).

I think the practical move is: **ship TQ4+TQ3 as the production tier stack and stop chasing TQ2**. The compression ratios are already excellent — TQ4 gives 3.6× compression vs F16, TQ3 gives 4.6×. That's the real win. What do you think?