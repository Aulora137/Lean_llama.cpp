# Lean_llama.cpp

**llama.cpp fork with TurboQuant KV cache quantization (TQ2 / TQ3 / TQ4) — CPU + Metal + CUDA**

This is the implementation repository for [LeanKV](https://github.com/Aulora137/LeanKV). It's a fork of [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) (which itself is a performance-focused fork of [llama.cpp](https://github.com/ggerganov/llama.cpp)) with TurboQuant KV cache compression added.

## What's Added

- **TQ4_0** — 4-bit KV cache quantization (4.5 bits/element, ~72% memory reduction, near-lossless)
- **TQ3_0** — 3-bit KV cache quantization (3.5 bits/element, ~78% memory reduction, near-lossless)
- **TQ2_0 / TQ2_1** — 2-bit KV cache (uniform + mixed-precision outlier-aware variants, 6.4× compression)
- **Hadamard rotation** — Automatic Walsh-Hadamard pre-rotation when using TQ types
- **CPU IQK kernels** — Optimized SIMD for AVX2/AVX512 and ARM NEON (mul_mat + Flash Attention)
- **Metal Flash Attention** — TQ2/TQ3/TQ4 dequant in Apple Silicon FA kernels
- **CUDA Flash Attention** — `vec_dot_fattn_vec_KQ` kernels with DP4A int8 dot products (graph splits 66 → 2)
- **Optimal rounding** — Coordinate descent + least-squares scale for TQ3

## Key Files

| File | Description |
|------|-------------|
| `ggml/src/ggml-tq.c` | TQ2/TQ3/TQ4 quantize/dequantize + optimal rounding |
| `ggml/src/ggml-common.h` | Block structs + codebook LUT tables |
| `ggml/src/ggml.c` | Type traits registration |
| `ggml/src/iqk/iqk_flash_attn.cpp` | CPU IQK Flash Attention (TQ2/TQ3/TQ4 support) |
| `ggml/src/iqk/fa/iqk_fa_templates.h` | HelperTQ20/TQ21/TQ30/TQ40 SIMD dequant |
| `ggml/src/iqk/iqk_gemm_legacy_quants.cpp` | K-side mul_mat kernels (AVX2 + NEON) |
| `ggml/src/ggml-metal.metal` | Metal FA kernels with TQ dequant |
| `ggml/src/ggml-cuda/fattn-vec-common.cuh` | CUDA FA `vec_dot_KQ` TQ kernels (DP4A) |

## Building

```bash
git clone --recurse-submodules https://github.com/Aulora137/Lean_llama.cpp.git
cd Lean_llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

## Usage

```bash
# TQ4 KV cache (recommended — essentially lossless)
./build/bin/llama-cli -m model.gguf -ctk tq4_0 -ctv tq4_0 -c 4096 \
  -p "Hello, how are you?" -n 64

# TQ3 KV cache (maximum compression)
./build/bin/llama-cli -m model.gguf -ctk tq3_0 -ctv tq3_0 -c 4096 \
  -p "Hello, how are you?" -n 64

# Perplexity benchmark
./build/bin/llama-perplexity -m model.gguf -ctk tq4_0 -ctv tq4_0 \
  -f wiki.test.raw -c 2048
```

## Platform Support

| Platform | TQ4 | TQ3 | TQ2 | Flash Attention |
|----------|-----|-----|-----|-----------------|
| x86_64 (AVX2/AVX512) | Yes | Yes | Yes | Full IQK (FA + mul_mat) |
| Apple Silicon (NEON) | Yes | Yes | Yes | Full IQK (FA + mul_mat) |
| Apple Silicon (Metal) | Yes | Yes | Yes | Validated on M2 |
| NVIDIA (CUDA) | Yes | Yes | Yes | Validated on RTX 4090 |

## Validation

Cross-backend validation on WikiText-2 (160 chunks, 6 models across 3 architectures).
CPU / Metal / CUDA produce consistent PPL on identical workloads (F16 and TQ2 configs
within ±0.05 PPL; TQ4/TQ3 within ±0.07).

Representative results (Mistral 7B, CUDA RTX 4090):

| Config | K-cache | PPL | Δ vs F16 |
|--------|--------:|----:|---------:|
| F16 | 128.00 MiB | 5.1638 | — |
| TQ4_0 | 36.00 MiB | 5.1781 | +0.28% |
| TQ3_0 | 28.00 MiB | 5.2464 | +1.60% |
| TQ2_1 | 22.00 MiB | 5.9726 | +15.66% |
| TQ2_0 | 20.00 MiB | 6.4612 | +25.12% |

Highlights:

- **Gemma 3-4B**: TQ3_0 *improves* PPL (-1.6%) — Hadamard rotation acts as regularization
- **Qwen 3.5-9B hybrid**: TQ2_0 at +2.6% PPL with 6.4× compression (best aggressive target)
- **CUDA batch**: 36 configs × 160 chunks in 39 minutes on RTX 4090 at 94%+ F16 throughput
- **Metal batch**: full 160-chunk M2 validation matches CPU within ±0.02 PPL

Global default (if unsure): **TQ3_0 uniform K+V** — near-lossless everywhere, 4.6× compression.

### Documentation (LeanKV repository)

| Document | Contents |
|----------|----------|
| [RESULTS.md](https://github.com/Aulora137/LeanKV/blob/main/docs/RESULTS.md) | Full test results — unit tests, synthetic eval, perplexity across 6 models, cross-backend validation |
| [CUDA-RESULTS.md](https://github.com/Aulora137/LeanKV/blob/main/docs/CUDA-RESULTS.md) | CUDA Flash Attention implementation, DP4A dispatch, 160-chunk RTX 4090 batch |
| [TQ2-METAL-RESULTS.md](https://github.com/Aulora137/LeanKV/blob/main/docs/TQ2-METAL-RESULTS.md) | Apple M2 Metal validation, TQ2/TQ2_1 master comparison, decode speed |
| [DESIGN-FOR-QUANTIZATION.md](https://github.com/Aulora137/LeanKV/blob/main/docs/DESIGN-FOR-QUANTIZATION.md) | TurboQuant algorithm, Hadamard rotation, Lloyd-Max codebooks, bit-packing |

## Upstream

This fork inherits all of ik_llama.cpp's improvements (IQK matrix multiplication, Flash Attention, new quantization types, etc.). See [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) for upstream features.

## License

MIT
