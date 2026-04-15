#!/bin/bash
# TQ3_0 160-chunk rerun (TQ4 already done)
# Wrapped in caffeinate to prevent Mac sleep

set -e

MODEL="$HOME/LeanInfer/models/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"
WIKI="$HOME/Lean_llama.cpp/wikitext-2-raw/wiki.test.raw"
BIN="./build/bin/llama-perplexity"
OUT="docs/metal-tq3-rerun-results.txt"

{
echo "====================================================================="
echo "Metal TQ3_0 Rerun — Mistral 7B, 160 chunks"
echo "Date:     $(date)"
echo "Host:     $(hostname)"
echo "Branch:   $(git branch --show-current)"
echo "Commit:   $(git log --oneline -1)"
echo "Note:     Wrapped in caffeinate to prevent sleep"
echo "====================================================================="

echo ""
echo "====== TQ3_0/F16 uniform ======"
echo "Started: $(date)"
"$BIN" -m "$MODEL" -f "$WIKI" -c 2048 -ngl 99 \
    -ctk tq3_0 -ctv f16 2>&1 | \
    grep -E "KV self size|Final estimate|^\[[0-9]+\]"
echo "Finished: $(date)"

echo ""
echo "====================================================================="
echo "Complete: $(date)"
echo "====================================================================="
} 2>&1 | tee "$OUT"
