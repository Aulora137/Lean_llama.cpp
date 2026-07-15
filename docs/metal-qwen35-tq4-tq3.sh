#!/bin/bash
# Metal TQ4/TQ3 Regression — Qwen 3.5-9B Q4_K_M on M2, post upstream-merge
#
# Verifies the 2026-07 rebased main (fa4c2e60) reproduces the April CUDA
# supplemental baselines on Metal:
#   F16/F16    : 7.1404 +/- 0.04631  (145 chunks)
#   TQ4_0/F16  : 7.1453 +/- 0.04636
#   TQ3_0/F16  : 7.1663 +/- 0.04651
# Pass criterion (expected-ppl-targets.md): within ~±0.05 of baseline;
# >±0.1 is significant, >±0.2 pathological.
#
# Run from Lean_llama.cpp root on the M2. Wrapped in caffeinate upstream.

set -e

MODEL="${1:-$HOME/LeanInfer/models/qwen35-9b-instruct-q4_k_m.gguf}"
WIKI="${2:-$HOME/Lean_llama.cpp/wikitext-2-raw/wiki.test.raw}"
BIN="./build/bin/llama-perplexity"
OUT="docs/metal-qwen35-tq4-tq3-results-canonical.txt"

if [ ! -f "$MODEL" ]; then
    echo "ERROR: Model not found: $MODEL"
    echo "Usage: $0 [model_path] [wiki_path]"
    exit 1
fi

if [ ! -f "$WIKI" ]; then
    echo "ERROR: Wiki dataset not found: $WIKI"
    exit 1
fi

{
echo "====================================================================="
echo "Metal TQ4/TQ3 Regression — Qwen 3.5-9B Q4_K_M, 145+ chunks"
echo "Date:     $(date)"
echo "Host:     $(hostname)"
echo "Backend:  Metal GPU (-ngl 99)"
echo "Model:    $MODEL"
echo "Branch:   $(git branch --show-current)"
echo "Commit:   $(git log --oneline -1)"
echo "Baseline: April CUDA supplemental (docs/cuda-batch-results.txt)"
echo "====================================================================="

echo ""
echo "====== F16/F16 baseline ======"
echo "Started: $(date)"
"$BIN" -m "$MODEL" -f "$WIKI" -c 2048 -ngl 99 \
    -ctk f16 -ctv f16 2>&1 | \
    grep -E "KV self size|Final estimate|^\[[0-9]+\]"
echo "Finished: $(date)"

echo ""
echo "====== TQ4_0/F16 uniform ======"
echo "Started: $(date)"
"$BIN" -m "$MODEL" -f "$WIKI" -c 2048 -ngl 99 \
    -ctk tq4_0 -ctv f16 2>&1 | \
    grep -E "KV self size|Final estimate|^\[[0-9]+\]"
echo "Finished: $(date)"

echo ""
echo "====== TQ3_0/F16 uniform ======"
echo "Started: $(date)"
"$BIN" -m "$MODEL" -f "$WIKI" -c 2048 -ngl 99 \
    -ctk tq3_0 -ctv f16 2>&1 | \
    grep -E "KV self size|Final estimate|^\[[0-9]+\]"
echo "Finished: $(date)"

echo ""
echo "====================================================================="
echo "All configs complete: $(date)"
echo "====================================================================="
} 2>&1 | tee "$OUT"
