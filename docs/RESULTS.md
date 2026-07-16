# TurboQuant KV Cache Quantization — Results

Perplexity (PPL) measured on WikiText-2 raw, 160+ chunks, context 2048.
Lower PPL is better. Delta is relative to F16 baseline on the same backend.

## TQ4_0 (4.5 bits/element) — Production Default

| Model | Backend | F16 PPL | TQ4_0 PPL | Delta | K Cache |
|-------|---------|---------|-----------|-------|---------|
| Mistral 7B | CUDA (RTX 4090) | 5.1638 | 5.1781 | +0.28% | 36 MiB (was 128) |
| Mistral 7B | Metal (M2 Air, 2026-07 †) | 5.1678 | 5.1781 | +0.20% | 36 MiB |
| Qwen3-8B | CUDA | 8.6097 | 8.7932 | +2.13% | 40.5 MiB (was 144) |
| Gemma 3-4B | CUDA | 12.5221 | 12.3760 | **-1.17%** | 38.25 MiB (was 136) |
| Llama 3-8B | CUDA | 7.4059 | 7.4197 | +0.19% | 36 MiB (was 128) |
| Qwen3-4B | CUDA | 12.9359 | 12.6261 | **-2.39%** | 40.5 MiB |
| Qwen 3.5-9B | CUDA | 7.1404 | 7.1453 | +0.07% | 9 MiB (was 32) |
| Qwen 3.5-9B | CPU (AVX2, Ryzen 7) | 7.2591 | 7.2722 | +0.02% | 18 MiB (was 64) |
| Qwen 3.5-9B | Metal (M2 Air, 2026-07) | 7.2533 | 7.2965 | +0.60% | 9 MiB (was 32) |

† Re-run 2026-07-15 on the canonical dataset
(docs/metal-tq4-tq3-results-canonical.txt). The original Apr 14-15
fill-in used a non-canonical wiki.test.raw fetched seconds before the
run (TQ4 5.1103 / TQ3 5.1743, which faked a -1.1% TQ4 "improvement"
against the pre-swap F16 baseline — see the ANALYSIS section of
docs/metal-qwen35-tq4-tq3-results.txt). Canonical Metal TQ4 matches
CUDA to 4 decimals (5.1781 vs 5.1781).

TQ4_0 is near-lossless on every tested architecture. On Gemma and
Qwen3-4B it actually **improves** PPL — the Hadamard rotation acts as a
mild regularizer that helps certain models.

K cache compression: **3.6x** vs F16.

## TQ3_0 (3.5 bits/element) — Aggressive Tier

| Model | Backend | F16 PPL | TQ3_0 PPL | Delta | K Cache |
|-------|---------|---------|-----------|-------|---------|
| Mistral 7B | CUDA (RTX 4090) | 5.1638 | 5.2464 | +1.60% | 28 MiB (was 128) |
| Mistral 7B | Metal (M2 Air, 2026-07 †) | 5.1678 | 5.2446 | +1.49% | 28 MiB |
| Qwen3-8B | CUDA | 8.6097 | 8.8888 | +3.24% | 31.5 MiB (was 144) |
| Gemma 3-4B | CUDA | 12.5221 | 12.3214 | **-1.60%** | 29.75 MiB (was 136) |
| Llama 3-8B | CUDA | 7.4059 | 7.5526 | +1.98% | 28 MiB (was 128) |
| Qwen3-4B | CUDA | 12.9359 | 12.6261 | -2.39% | 40.5 MiB |
| Qwen 3.5-9B | CUDA | 7.1404 | 7.1663 | +0.36% | 7 MiB (was 32) |
| Qwen 3.5-9B | CPU (AVX2, Ryzen 7) | 7.2591 | 7.2875 | +0.04% | 14 MiB (was 64) |
| Qwen 3.5-9B | Metal (M2 Air, 2026-07) | 7.2533 | 7.3287 | +1.04% | 7 MiB (was 32) |

TQ3_0 stays within +3.3% PPL on all models. Gemma again improves. Qwen3-4B
is auto-promoted to TQ4_0 by the rank-deficiency safety net (n_embd/n_head
< 1.0), so TQ3 and TQ4 report the same number there.

K cache compression: **4.6x** vs F16.

## Cross-Backend Consistency (Mistral 7B, 160 chunks)

| Config | CPU (AVX2) | Metal (M2) | CUDA (4090) | Spread |
|--------|-----------|------------|-------------|--------|
| F16 | 5.1627 | 5.1678 | 5.1638 | 0.005 |
| TQ4_0 | — | 5.1781 | 5.1781 | 0.000 |
| TQ3_0 | — | 5.2446 | 5.2464 | 0.002 |

All three backends are consistent. Metal TQ numbers are the 2026-07-15
canonical-dataset re-run; TQ4 matches CUDA to four decimals and TQ3 is
within 0.002. Qwen 3.5-9B gives the same picture on the hybrid arch:
Metal F16 7.2533 vs Ryzen CPU 7.2591 (delta 0.006)
(docs/metal-qwen35-tq4-tq3-results-canonical.txt).

## TQ2 — Not Viable

TQ2_0 (2.5 bits/element, 4 codebook levels) was tested extensively and
produces unacceptable quality loss on dense-attention models:

- Qwen3-8B: **+117% PPL** (8.61 -> 18.66)
- Llama 3-8B: **+72% PPL** (7.41 -> 12.71)
- Mistral 7B: **+25% PPL** (5.16 -> 6.46)

This is an **information-theoretic limit**: 4 Lloyd-Max levels cannot
preserve dot-product fidelity in dense attention regardless of codebook
optimization. Experimental proof: grid-searching 41 scale factors to
minimize actual dot-product error `|q*k - q*k_hat|^2` yields only 0.02 dB
improvement over MSE-optimal — confirming the current codebook is already
at the theoretical ceiling for scalar quantization.

The one exception is **Qwen 3.5-9B** (hybrid Mamba+attention, only 8 of 36
layers use KV cache), where TQ2_0 costs just +2.57% PPL. This works because
the Mamba layers bypass quantization entirely, limiting error accumulation.

TQ2 is archived as a research finding. Production deployments should use
TQ4_0 (default) or TQ3_0 (aggressive).

## Qwen 3.5-9B — Hybrid Architecture Note

Qwen 3.5-9B uses a Mamba+attention hybrid where only 8 of 36 layers have
KV cache. This makes it exceptionally TQ-friendly:

| Config | PPL | Delta | K Cache |
|--------|-----|-------|---------|
| F16 | 7.1404 | — | 32 MiB |
| TQ4_0 | 7.1453 | +0.07% | 9 MiB |
| TQ3_0 | 7.1663 | +0.36% | 7 MiB |
| TQ2_0 | 7.3239 | +2.57% | 5 MiB |

Even TQ3_0 at 7 MiB K cache is practically lossless. This model is currently
deployed on the Aulora bitcoin node with TQ4_0 KV cache.

## Production Deployment

The Aulora/Rikuri node (Ryzen 7 7735U, CPU-only) runs three llama-server
instances from this tree (as of 2026-07-14):

| Port | Model | KV cache |
|------|-------|----------|
| :8080 | Gemma 3-4B Q4_K_M (chat) | tq4_0 K + V |
| :8081 | Qwen 3.5-2B Q4_K_M (intent router) | tq4_0 K + V, `-fa on` |
| :8082 | bge-base-en-v1.5 (embeddings) | n/a |

## Upstream Merge Regression Gate — 2026-07-14

After merging upstream ik_llama.cpp @ 6d78a87c (340 commits, ~3 months of
drift) plus the deltanet F16 state-cache fix (508ad76e), the full gate was
re-run on the same hardware, model, and eval as the April baselines
(Qwen 3.5-9B Q4_K_M, WikiText-2 raw, 145 chunks, ctx 2048, CPU AVX2).
K and V both quantized — compare against the April `_TQ4_TQ4` / `_TQ3_TQ3`
logs, not the K-only table rows above.

| KV config | Post-merge PPL | April PPL | Result |
|-----------|---------------|-----------|--------|
| F16 / F16 | 7.2591 | 7.2591 | exact match |
| TQ4_0 / TQ4_0 | 7.2912 | 7.2912 | exact match |
| TQ3_0 / TQ3_0 | 7.3409 | 7.3472 | -0.09% (slightly better) |

KV self size matched April exactly (18 MiB TQ4, 14 MiB TQ3, 64 MiB F16).
Hadamard auto-enable and the calibration codebook engaged normally.
Raw logs: `LeanKV/prototype/eval/results/logs/postmerge-2026-07/` (local).
Metal TQ kernels re-validated post-merge on M2 (2026-07-15, canonical
dataset — see the Metal rows and dagger note above). CUDA TQ kernels are
carried in this tree but not yet re-validated post-merge (need a 4090 build).
