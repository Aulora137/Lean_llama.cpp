#!/bin/bash
# Metal TQ4/TQ3 Gold-Standard Fill-In — 160-chunk validation on M2
#
# Fills the ‡ gaps in RESULTS.md Section 19 with actual TQ4/F16 and TQ3/F16
# 160-chunk WikiText-2 PPL on Mistral 7B. Combined with the overnight run
# from Apr 13, this completes the Metal gold-standard table.
#
# EXPECTED RUNTIME: ~4.5 hours on M2 Metal (~90 sec/chunk × 160 × 2 configs)
# Kick off before bed, results ready in the morning.
#
# Run from Lean_llama.cpp root on the M2.

set -e

MODEL="${1:-$HOME/LeanInfer/models/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf}"
WIKI="${2:-$HOME/Lean_llama.cpp/wikitext-2-raw/wiki.test.raw}"
BIN="./build/bin/llama-perplexity"
OUT="docs/metal-tq4-tq3-results.txt"

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
echo "Metal TQ4/TQ3 Gold-Standard Fill-In — Mistral 7B, 160 chunks"
echo "Date:     $(date)"
echo "Host:     $(hostname)"
echo "Backend:  Metal GPU (-ngl 99)"
echo "Model:    $MODEL"
echo "Branch:   $(git branch --show-current)"
echo "Commit:   $(git log --oneline -1)"
echo "====================================================================="

echo ""
echo "====== TQ4_0/F16 uniform ======"
echo "Started: $(date)"
"$BIN" -m "$MODEL" -f "$WIKI" -c 2048 -ngl 99 \
    -ctk tq4_0 -ctv f16 2>&1 | \
    grep -E "KV self size|Final estimate|^\[[0-9]+\]"
echo "Finished: $(date)"

echo ""
echo ""
echo "====== TQ3_0/F16 uniform ======"
echo "Started: $(date)"
"$BIN" -m "$MODEL" -f "$WIKI" -c 2048 -ngl 99 \
    -ctk tq3_0 -ctv f16 2>&1 | \
    grep -E "KV self size|Final estimate|^\[[0-9]+\]"
echo "Finished: $(date)"

echo ""
echo "====================================================================="
echo "Both configs complete: $(date)"
echo "====================================================================="
} 2>&1 | tee "$OUT"
