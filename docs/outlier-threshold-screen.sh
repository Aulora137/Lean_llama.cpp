#!/bin/bash
# Screening: 7 outlier-detection variants at 3 chunks on Mistral 7B
# Finds the top 2-3 for full 160-chunk validation.
#
# Ref baseline (3 chunks, Mistral 7B, adaptive with default 2.0× threshold):
#   TQ2_0 uniform: 20.00 MiB, PPL 8.1534
#   Adaptive:      20.31 MiB, PPL 8.0484
#   TQ2_1 uniform: 22.00 MiB, PPL 7.5571

set -e

MODEL=/home/junc/Aulora/bitcoin-node-stack/models/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf
WIKI=/home/junc/LeanKV/prototype/eval/wikitext-2-raw/wiki.test.raw
BIN=./build/bin
OUT=docs/outlier-threshold-screen.txt

{
echo "======================================================================"
echo "Outlier Threshold Screening — Mistral 7B (3 chunks)"
echo "Date: $(date)"
echo "Commit: $(git log --oneline -1)"
echo "======================================================================"

run_variant() {
    local name="$1"
    local metric="$2"
    local threshold="$3"
    echo ""
    echo "====== $name  (metric=$metric, threshold=$threshold) ======"
    LEANKV_OUTLIER_METRIC="$metric" LEANKV_OUTLIER_THRESHOLD="$threshold" \
        $BIN/llama-perplexity -m "$MODEL" -f "$WIKI" -c 2048 -ngl 0 \
        -ctk tq2_0 -ctv f16 --kv-outlier-frac -1 -t 8 --chunks 3 2>&1 | \
        grep -E "adaptive K-cache|auto-detect summary|KV self size|Final estimate|outlier K policy"
}

# V0: Current default (backward compat sanity)
run_variant "V0 Current default"           0 2.0

# V1-V3: Lower n_moderate threshold
run_variant "V1 n_moderate 1.5x"            0 1.5
run_variant "V2 n_moderate 1.2x"            0 1.2
run_variant "V3 n_moderate 1.1x"            0 1.1

# V4-V5: Total-variance metric (relative to cross-layer median)
run_variant "V4 total_var 1.1x median"      2 1.1
run_variant "V5 total_var 1.0x median"      2 1.0

# V6: max_ratio
run_variant "V6 max_ratio 2.5x"             1 2.5

# V7: hybrid
run_variant "V7 hybrid (nmod 2x OR max 2.5x)" 3 2.5

echo ""
echo "======================================================================"
echo "Screening complete: $(date)"
echo "======================================================================"

} 2>&1 | tee "$OUT"
