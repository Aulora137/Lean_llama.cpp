# Lean_llama.cpp

**llama.cpp fork with TurboQuant KV cache quantization (TQ3/TQ4)**

This is the implementation repository for [LeanKV](https://github.com/Aulora137/LeanKV). It's a fork of [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) (which itself is a performance-focused fork of [llama.cpp](https://github.com/ggerganov/llama.cpp)) with TurboQuant KV cache compression added.

## What's Added

- **TQ4_0** — 4-bit KV cache quantization (4.5 bits/element, 75% memory reduction, +0.02 PPL)
- **TQ3_0** — 3-bit KV cache quantization (3.5 bits/element, 81% memory reduction, +0.05 PPL)
- **Hadamard rotation** — Automatic Walsh-Hadamard pre-rotation when using TQ types
- **IQK kernels** — Optimized SIMD kernels for both AVX2/AVX512 and ARM NEON
- **Optimal rounding** — Coordinate descent + least-squares scale for TQ3

## Key Files

| File | Description |
|------|-------------|
| `ggml/src/ggml-tq.c` | TQ3/TQ4 quantize/dequantize + optimal rounding |
| `ggml/src/ggml-common.h` | Block structs + codebook LUT tables |
| `ggml/src/ggml.c` | Type traits registration |
| `ggml/src/iqk/iqk_flash_attn.cpp` | IQK Flash Attention (TQ3/TQ4 support) |
| `ggml/src/iqk/fa/iqk_fa_templates.h` | HelperTQ30/HelperTQ40 SIMD dequant |
| `ggml/src/iqk/iqk_gemm_legacy_quants.cpp` | K-side mul_mat kernels (AVX2 + NEON) |

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

| Platform | TQ4 | TQ3 | IQK Acceleration |
|----------|-----|-----|-----------------|
| x86_64 (AVX2) | Yes | Yes | Full (FA + mul_mat) |
| Apple Silicon (NEON) | Yes | Yes | Full (FA + mul_mat) |
| CUDA / Metal | Not yet | Not yet | Planned |

## Upstream

This fork inherits all of ik_llama.cpp's improvements (IQK matrix multiplication, Flash Attention, new quantization types, etc.). See [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) for upstream features.

## License

MIT
