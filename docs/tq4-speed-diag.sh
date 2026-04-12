#!/bin/bash
# TQ4 Speed Anomaly Diagnostic
# Run from Lean_llama.cpp root. Tests whether TQ4 24% slowdown persists
# after binary search optimization, and isolates the bottleneck.
#
# Usage: ./docs/tq4-speed-diag.sh [model_path]

set -e

MODEL="${1:-$(find ~/models ~/LeanInfer/models -name '*q4_k_m.gguf' 2>/dev/null | head -1)}"
BIN="./build/bin"

if [ ! -f "$MODEL" ]; then
    echo "ERROR: Model not found. Usage: $0 <model.gguf>"
    exit 1
fi

echo "Model: $MODEL"
echo "Testing TQ4 speed anomaly..."
echo ""

# Generate a moderate-length prompt (same as main test)
PROMPT_LONG="The quick brown fox jumps over the lazy dog. "
PROMPT_LONG="$PROMPT_LONG$PROMPT_LONG$PROMPT_LONG$PROMPT_LONG"
PROMPT_LONG="$PROMPT_LONG$PROMPT_LONG$PROMPT_LONG$PROMPT_LONG"

echo "====== Test 1: Speed comparison (3 runs each, median) ======"
echo "(Running each config 3x to reduce noise)"
echo ""

for CTK in f16 tq4_0 tq3_0 tq2_0; do
  echo "--- K=$CTK (3 runs) ---"
  for run in 1 2 3; do
    $BIN/llama-cli -m "$MODEL" -ngl 0 -ctk $CTK -ctv f16 \
      -c 2048 -p "$PROMPT_LONG" -n 32 2>&1 | grep "prompt eval time"
  done
  echo ""
done

echo ""
echo "====== Test 2: Longer prompt (more attention work) ======"
echo ""

# 4x longer prompt to amplify attention overhead
PROMPT_VLONG="$PROMPT_LONG$PROMPT_LONG$PROMPT_LONG$PROMPT_LONG"

for CTK in f16 tq4_0 tq3_0; do
  echo "--- K=$CTK (long prompt) ---"
  $BIN/llama-cli -m "$MODEL" -ngl 0 -ctk $CTK -ctv f16 \
    -c 4096 -p "$PROMPT_VLONG" -n 16 2>&1 | grep "prompt eval time"
  echo ""
done

echo ""
echo "====== Test 3: Decode-only (isolates generation speed) ======"
echo ""

for CTK in f16 tq4_0 tq3_0 tq2_0; do
  echo "--- K=$CTK (decode only, short prompt) ---"
  $BIN/llama-cli -m "$MODEL" -ngl 0 -ctk $CTK -ctv f16 \
    -c 2048 -p "Hello" -n 64 2>&1 | grep -E "prompt eval time|eval time"
  echo ""
done

echo ""
echo "====== Test 4: Q4_0 KV cache comparison ======"
echo "(Tests if standard Q4_0 shows same anomaly)"
echo ""

# Q4_0 uses same nibble packing but without Hadamard
for CTK in f16 q4_0 tq4_0; do
  echo "--- K=$CTK ---"
  $BIN/llama-cli -m "$MODEL" -ngl 0 -ctk $CTK -ctv f16 \
    -c 2048 -p "$PROMPT_LONG" -n 32 2>&1 | grep "prompt eval time" 2>/dev/null || echo "(type not supported)"
  echo ""
done

echo "Done. Key questions:"
echo "1. Does TQ4 still show 24% slowdown after binary search optimization?"
echo "2. Does the gap increase with longer prompts (attention-bound)?"
echo "3. Is decode speed similar across types (weight-matmul-bound)?"
echo "4. Does Q4_0 show the same slowdown as TQ4?"
