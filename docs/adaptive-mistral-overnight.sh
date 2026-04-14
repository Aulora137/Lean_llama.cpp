#!/bin/bash
# Overnight PPL validation of Phase 3 adaptive K-cache selection on Mistral 7B
# Runs 4 configs at 145 chunks each, saves to docs/adaptive-mistral-results.txt
#
# Hypothesis: adaptive should match or beat uniform TQ2_1 at <93% of its memory
# Expected time: ~80 min per config × 4 = ~5-6 hours

set -e

MODEL=/home/junc/Aulora/bitcoin-node-stack/models/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf
WIKI=/home/junc/LeanKV/prototype/eval/wikitext-2-raw/wiki.test.raw
BIN=./build/bin
OUT=docs/adaptive-mistral-results.txt

{
echo "======================================================================"
echo "Mistral 7B — Adaptive K-Cache Selection PPL Validation"
echo "Date:     $(date)"
echo "Host:     $(hostname)"
echo "CPU:      $(grep 'model name' /proc/cpuinfo 2>/dev/null | head -1 | cut -d: -f2)"
echo "Model:    $MODEL"
echo "Branch:   $(git branch --show-current)"
echo "Commit:   $(git log --oneline -1)"
echo "======================================================================"

for cfg in \
    "f16       f16    .         F16_baseline" \
    "tq2_0     f16    .         TQ2_0_uniform" \
    "tq2_1     f16    .         TQ2_1_uniform" \
    "tq2_0     f16    -1        TQ2_0_adaptive"; do

    CTK=$(echo $cfg | awk '{print $1}')
    CTV=$(echo $cfg | awk '{print $2}')
    FRAC=$(echo $cfg | awk '{print $3}')
    NAME=$(echo $cfg | awk '{print $4}')

    echo ""
    echo ""
    echo "====== $NAME  (ctk=$CTK ctv=$CTV frac=$FRAC) ======"
    echo "Started: $(date)"

    if [ "$FRAC" = "." ]; then
        $BIN/llama-perplexity -m "$MODEL" -f "$WIKI" -c 2048 -ngl 0 \
            -ctk $CTK -ctv $CTV -t 8 2>&1 | \
            grep -E "KV self size|adaptive K-cache|auto-detect summary|Final estimate|^\[[0-9]+\]"
    else
        $BIN/llama-perplexity -m "$MODEL" -f "$WIKI" -c 2048 -ngl 0 \
            -ctk $CTK -ctv $CTV -t 8 --kv-outlier-frac $FRAC 2>&1 | \
            grep -E "KV self size|adaptive K-cache|auto-detect summary|Final estimate|^\[[0-9]+\]"
    fi

    echo "Finished: $(date)"
done

echo ""
echo "======================================================================"
echo "All configs complete: $(date)"
echo "======================================================================"

} 2>&1 | tee "$OUT"
