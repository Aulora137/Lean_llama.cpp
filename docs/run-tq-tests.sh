#!/bin/bash
# TQ KV Cache — Automated Test Runner
# Usage: ./docs/run-tq-tests.sh [model_path] [wiki_path]
#
# Run from Lean_llama.cpp root directory.
# Results saved to docs/tq-test-results-$(hostname).txt

set -e

MODEL="${1:-$(find ~/models ~/LeanInfer/models -name '*q4_k_m.gguf' 2>/dev/null | head -1)}"
WIKI="${2:-$(find . ~/Lean_llama.cpp -name 'wiki.test.raw' 2>/dev/null | head -1)}"
BIN="./build/bin"
OUTFILE="docs/tq-test-results-$(hostname).txt"

if [ ! -f "$MODEL" ]; then
    echo "ERROR: Model not found. Usage: $0 <model.gguf> [wiki.test.raw]"
    echo "Suggested: Download Qwen2.5-7B-Instruct-Q4_K_M.gguf from HuggingFace"
    exit 1
fi

echo "Model: $MODEL"
echo "Wiki:  $WIKI"
echo "Output: $OUTFILE"
echo ""

{
echo "======================================================================"
echo "TQ KV Cache Test Results"
echo "Date:     $(date)"
echo "Host:     $(hostname)"
echo "CPU:      $(grep 'model name' /proc/cpuinfo 2>/dev/null | head -1 | cut -d: -f2 || sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'unknown')"
echo "Model:    $MODEL"
echo "Branch:   $(git branch --show-current)"
echo "Commit:   $(git log --oneline -1)"
echo "======================================================================"

# ── Test 1: Sanity Check ──────────────────────────────────────────────
echo ""
echo "====== TEST 1: Sanity — Coherent Generation ======"
PROMPT="The capital of France is"

for cfg in \
  "f16 f16 ." \
  "tq4_0 f16 ." \
  "tq3_0 f16 ." \
  "tq2_0 f16 ." \
  "tq4_0 f16 --kv-outlier-frac_0.25" \
  "tq3_0 f16 --kv-outlier-frac_0.25" \
  "tq2_0 f16 --kv-outlier-frac_0.25"; do

  CTK=$(echo $cfg | cut -d' ' -f1)
  CTV=$(echo $cfg | cut -d' ' -f2)
  EXTRA=$(echo $cfg | cut -d' ' -f3 | sed 's/_/ /g' | sed 's/\.$//;s/^\.$//')

  echo ""
  echo "--- K=$CTK V=$CTV $EXTRA ---"
  $BIN/llama-cli -m "$MODEL" -ngl 0 -ctk $CTK -ctv $CTV $EXTRA \
    -c 2048 -p "$PROMPT" -n 32 --no-display-prompt 2>/dev/null | head -3
done

# ── Test 2: Perplexity ────────────────────────────────────────────────
if [ -f "$WIKI" ]; then
echo ""
echo ""
echo "====== TEST 2: Perplexity (PPL) — Quality Measurement ======"
echo "(Lower is better. Key test — record these numbers.)"

for cfg in \
  "f16 f16 ." \
  "tq4_0 f16 ." \
  "tq3_0 f16 ." \
  "tq2_0 f16 ." \
  "tq3_0 f16 --kv-outlier-frac_0.25" \
  "tq2_0 f16 --kv-outlier-frac_0.25"; do

  CTK=$(echo $cfg | cut -d' ' -f1)
  CTV=$(echo $cfg | cut -d' ' -f2)
  EXTRA=$(echo $cfg | cut -d' ' -f3 | sed 's/_/ /g' | sed 's/\.$//;s/^\.$//')

  echo ""
  echo "--- K=$CTK V=$CTV $EXTRA ---"
  $BIN/llama-perplexity -m "$MODEL" -f "$WIKI" -c 2048 -ngl 0 \
    -ctk $CTK -ctv $CTV $EXTRA 2>&1 | grep -E "perplexity|PPL|Final"
done
else
  echo ""
  echo "====== TEST 2: SKIPPED (wiki.test.raw not found) ======"
  echo "Download: wget https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip"
fi

# ── Test 3: Memory ────────────────────────────────────────────────────
echo ""
echo ""
echo "====== TEST 3: KV Cache Memory Usage ======"

for CTK in f16 tq4_0 tq3_0 tq2_0; do
  echo ""
  echo "--- K=$CTK ---"
  $BIN/llama-cli -m "$MODEL" -ngl 0 -ctk $CTK -ctv f16 -c 4096 \
    -p "Hello" -n 1 2>&1 | grep "KV self size"
done

# ── Test 5: V-Cache Quantization ──────────────────────────────────────
if [ -f "$WIKI" ]; then
echo ""
echo ""
echo "====== TEST 5: V-Cache Quantization ======"

for cfg in \
  "tq4_0 tq4_0" \
  "tq3_0 tq3_0" \
  "tq3_0 tq2_0"; do

  CTK=$(echo $cfg | cut -d' ' -f1)
  CTV=$(echo $cfg | cut -d' ' -f2)

  echo ""
  echo "--- K=$CTK V=$CTV ---"
  $BIN/llama-perplexity -m "$MODEL" -f "$WIKI" -c 2048 -ngl 0 \
    -ctk $CTK -ctv $CTV 2>&1 | grep -E "perplexity|PPL|Final"
done
fi

# ── Test 6: Speed ─────────────────────────────────────────────────────
echo ""
echo ""
echo "====== TEST 6: Speed (tok/s) ======"

# Generate a moderate-length prompt
PROMPT_LONG="The quick brown fox jumps over the lazy dog. "
PROMPT_LONG="$PROMPT_LONG$PROMPT_LONG$PROMPT_LONG$PROMPT_LONG"
PROMPT_LONG="$PROMPT_LONG$PROMPT_LONG$PROMPT_LONG$PROMPT_LONG"

for CTK in f16 tq4_0 tq3_0 tq2_0; do
  echo ""
  echo "--- K=$CTK ---"
  $BIN/llama-cli -m "$MODEL" -ngl 0 -ctk $CTK -ctv f16 \
    -c 2048 -p "$PROMPT_LONG" -n 32 2>&1 | grep -E "eval time|prompt eval time"
done

echo ""
echo "======================================================================"
echo "Tests complete: $(date)"
echo "======================================================================"

} 2>&1 | tee "$OUTFILE"

echo ""
echo "Results saved to: $OUTFILE"
