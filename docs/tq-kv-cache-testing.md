# TQ KV Cache — Test Procedures

## Overview

Validate TQ2_0, TQ3_0, TQ4_0 KV cache quantization + outlier channel
permutation across all tiers before proceeding to CUDA/Metal kernels and
auto-migration.

All tests run through the optimized IQK flash attention path (AVX2 on Ryzen,
NEON on Apple Silicon). Hadamard rotation is auto-enabled for all TQ types.

---

## Prerequisites

### 1. Build from branch

```bash
cd ~/Lean_llama.cpp
git checkout feature/tq2-outlier-tiered
git pull origin feature/tq2-outlier-tiered

mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
```

### 2. Download test data

```bash
cd ~/Lean_llama.cpp

# WikiText-2 test set for perplexity
wget -q https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip
unzip -o wikitext-2-raw-v1.zip
# Result: wikitext-2-raw/wiki.test.raw (~1.2 MB)
```

### 3. Models

| Model | Why | Download |
|-------|-----|----------|
| **Qwen2.5-7B-Instruct Q4_K_M** | Main test model, good balance of quality and speed | [HF link](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF) |
| **Qwen3.5-9B Q4_K_M** (if available) | Larger model, better PPL resolution | Use existing `qwen35-9b-instruct-q4_k_m.gguf` |

Set the model path:
```bash
MODEL=~/models/qwen2.5-7b-instruct-q4_k_m.gguf
# or
MODEL=~/LeanInfer/models/qwen35-9b-instruct-q4_k_m.gguf
```

---

## Test 1: Sanity — Coherent Generation

Quick check that each config produces coherent text. Run all 8 configs:

```bash
cd ~/Lean_llama.cpp/build
PROMPT="The capital of France is"

for cfg in \
  "f16 f16" \
  "tq4_0 f16" \
  "tq3_0 f16" \
  "tq2_0 f16" \
  "tq4_0 f16 --kv-outlier-frac 0.25" \
  "tq3_0 f16 --kv-outlier-frac 0.25" \
  "tq2_0 f16 --kv-outlier-frac 0.25" \
  "tq3_0 tq3_0"; do

  set -- $cfg
  CTK=$1; CTV=$2; shift 2; EXTRA="$*"
  echo "=== K=$CTK V=$CTV $EXTRA ==="
  ./bin/llama-cli -m $MODEL -ngl 0 -ctk $CTK -ctv $CTV $EXTRA \
    -c 2048 -p "$PROMPT" -n 32 --no-display-prompt 2>/dev/null | head -3
  echo
done
```

**Expected:** All configs produce grammatical English. FP16 baseline should
say "Paris". TQ4 should match closely. TQ3/TQ2 may diverge but should be
coherent. Note any config that produces garbage.

---

## Test 2: Perplexity — Quality Measurement

This is the key test. Lower perplexity = better.

```bash
cd ~/Lean_llama.cpp/build
WIKI=~/Lean_llama.cpp/wikitext-2-raw/wiki.test.raw

echo "=== F16 baseline ==="
./bin/llama-perplexity -m $MODEL -f $WIKI -c 2048 -ngl 0 -ctk f16 -ctv f16

echo "=== TQ4_0 ==="
./bin/llama-perplexity -m $MODEL -f $WIKI -c 2048 -ngl 0 -ctk tq4_0 -ctv f16

echo "=== TQ3_0 ==="
./bin/llama-perplexity -m $MODEL -f $WIKI -c 2048 -ngl 0 -ctk tq3_0 -ctv f16

echo "=== TQ2_0 ==="
./bin/llama-perplexity -m $MODEL -f $WIKI -c 2048 -ngl 0 -ctk tq2_0 -ctv f16

echo "=== TQ3_0 + outlier 25% ==="
./bin/llama-perplexity -m $MODEL -f $WIKI -c 2048 -ngl 0 -ctk tq3_0 -ctv f16 --kv-outlier-frac 0.25

echo "=== TQ2_0 + outlier 25% ==="
./bin/llama-perplexity -m $MODEL -f $WIKI -c 2048 -ngl 0 -ctk tq2_0 -ctv f16 --kv-outlier-frac 0.25
```

**Expected results** (approximate, for a 7-9B model):

| Config | PPL delta vs F16 | K-cache bits/elem |
|--------|-----------------|-------------------|
| F16 | 0 (baseline) | 16.0 |
| TQ4_0 | < +0.05 | 4.5 |
| TQ3_0 | < +0.10 | 3.5 |
| TQ3_0 + outlier | < TQ3 alone | 3.5 |
| TQ2_0 | < +0.50 | 2.5 |
| TQ2_0 + outlier | < TQ2 alone | 2.5 |

**Key questions to answer:**
1. Does outlier permutation improve PPL for TQ3? For TQ2?
2. Is TQ2 + outlier close to plain TQ3?
3. Is TQ4 truly near-lossless?

---

## Test 3: Memory — KV Cache Size

Verify compression ratios match expectations:

```bash
cd ~/Lean_llama.cpp/build

for CTK in f16 tq4_0 tq3_0 tq2_0; do
  echo "=== K=$CTK ==="
  ./bin/llama-cli -m $MODEL -ngl 0 -ctk $CTK -ctv f16 -c 4096 \
    -p "Hello" -n 1 2>&1 | grep "KV self size"
done
```

**Expected** (for head_dim=128, n_layers=28-40, context=4096):

| K type | K-cache size | Compression vs F16 |
|--------|-------------|-------------------|
| F16 | ~X MiB | 1.0x |
| TQ4_0 | ~X/3.6 MiB | 3.6x |
| TQ3_0 | ~X/4.6 MiB | 4.6x |
| TQ2_0 | ~X/6.4 MiB | 6.4x |

---

## Test 4: Long Context — Quality at Scale

Test that quality holds up at longer contexts:

```bash
cd ~/Lean_llama.cpp/build

# Generate a long context test (if wiki.test.raw is too short, repeat it)
for CTX in 2048 4096 8192; do
  echo "=== Context=$CTX, TQ3+outlier ==="
  ./bin/llama-perplexity -m $MODEL -f $WIKI -c $CTX -ngl 0 \
    -ctk tq3_0 -ctv f16 --kv-outlier-frac 0.25 --chunks 3
done
```

**Expected:** PPL should remain stable or improve slightly with longer context
(more data for softmax). If PPL degrades sharply at longer contexts, that
indicates a bug in the cache management or permutation.

---

## Test 5: V-Cache Quantization

Test quantizing both K and V caches (V-cache requires flash attention):

```bash
cd ~/Lean_llama.cpp/build

echo "=== K=TQ3, V=TQ3 (both quantized) ==="
./bin/llama-perplexity -m $MODEL -f $WIKI -c 2048 -ngl 0 \
  -ctk tq3_0 -ctv tq3_0

echo "=== K=TQ4, V=TQ4 ==="
./bin/llama-perplexity -m $MODEL -f $WIKI -c 2048 -ngl 0 \
  -ctk tq4_0 -ctv tq4_0

echo "=== K=TQ3, V=TQ2 (aggressive) ==="
./bin/llama-perplexity -m $MODEL -f $WIKI -c 2048 -ngl 0 \
  -ctk tq3_0 -ctv tq2_0
```

**Expected:** V-cache quantization adds additional PPL degradation on top of
K-cache quantization. TQ4+TQ4 should still be near-lossless. TQ3+TQ2 is
the aggressive end.

---

## Test 6: Speed — Tokens per Second

Measure throughput to ensure IQK FA kernels are working:

```bash
cd ~/Lean_llama.cpp/build
PROMPT=$(python3 -c "print('The ' * 500)")

for CTK in f16 tq4_0 tq3_0 tq2_0; do
  echo "=== K=$CTK ==="
  ./bin/llama-cli -m $MODEL -ngl 0 -ctk $CTK -ctv f16 \
    -c 2048 -p "$PROMPT" -n 32 2>&1 | grep "eval time"
done
```

**Expected:** TQ types should be similar speed or slightly faster than F16
(smaller cache = better cache locality). If any TQ type is significantly
slower, the IQK kernel may be falling back to a scalar path.

---

## Test 7: AVX2 Kernel Validation (Ryzen-specific)

Verify AVX2 SIMD kernels are being used (not scalar fallback):

```bash
cd ~/Lean_llama.cpp/build

# Check that IQK is active
./bin/llama-cli -m $MODEL -ngl 0 -ctk tq3_0 -ctv f16 \
  -c 512 -p "Hello" -n 1 2>&1 | grep -i "iqk\|ik_llama\|backend"
```

**Expected:** Should show IQK-related backend initialization. If it falls
back to plain GGML, the SIMD kernels aren't being used.

---

## Recording Results

Please record all output in a file:

```bash
cd ~/Lean_llama.cpp
./docs/run-tq-tests.sh 2>&1 | tee docs/tq-test-results-ryzen.txt
```

The key numbers I need:
1. **PPL for each of the 6 configs** in Test 2 (the most important test)
2. **KV cache sizes** from Test 3
3. **Any configs that produce garbage** from Test 1
4. **Speed (tok/s)** from Test 6

---

## Troubleshooting

**Build fails:** Make sure you're on `feature/tq2-outlier-tiered` branch and
have pulled latest. The build requires C11 and C++17.

**Perplexity test hangs:** Add `--chunks 3` to limit evaluation length.
Full WikiText-2 test takes ~5-10 min per config on a 9B model with Ryzen.

**Outlier flag has no effect:** Check the log output for
`outlier K: N/M channels` line. If missing, the model's W_K tensor
may not be accessible (shouldn't happen for standard GGUF models).

**Garbage output with TQ2:** Expected for very small models (<1B).
TQ2 is designed for 7B+ models. Try TQ3 instead.

**CUDA/Metal not used:** These tests are CPU-only (`-ngl 0`).
GPU kernels for TQ types are not yet implemented — that's the next step
after validating CPU results.
