#!/bin/bash
# Metal GPU PPL validation — Mistral 7B, 160 chunks
# Compares Metal GPU (-ngl 99) to CPU baselines from adaptive-mistral-results.txt
#
# CPU baselines (Ryzen, 160 chunks):
#   F16:           5.1627 ± 0.029
#   TQ2_0 uniform: 6.4229 ± 0.036
#   TQ2_1 uniform: 5.9784 ± 0.033
#   V1 adaptive:   5.9940 ± 0.033  (21.69 MiB K, 1.5% less than TQ2_1)
#   V0 adaptive:   6.3413 ± 0.036  (20.31 MiB K, old 2.0× threshold)
#
# Pass criteria: Metal within ±0.1 PPL of CPU for each config.
# If dramatically worse → Metal TQ2 Hadamard bug, fix before CUDA.
#
# Expected time: ~3-4 hours total (4 configs × ~45-60 min each on M2 Metal)

set -e

MODEL=/Users/hchome/LeanInfer/models/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf
WIKI=/Users/hchome/LeanKV/prototype/eval/wikitext-2-raw/wikitext-2-raw/wiki.test.raw
BIN=./build/bin
OUT=docs/metal-mistral-results.txt

{
echo "======================================================================"
echo "Mistral 7B — Metal GPU PPL Validation (160 chunks)"
echo "Date:     $(date)"
echo "Host:     $(hostname)"
echo "Backend:  Metal GPU (-ngl 99)"
echo "Model:    $MODEL"
echo "Branch:   $(git branch --show-current)"
echo "Commit:   $(git log --oneline -1)"
echo "======================================================================"

for cfg in \
    "f16       f16    .         F16_baseline" \
    "tq2_1     f16    .         TQ2_1_uniform" \
    "tq2_1     f16    -1        V1_adaptive" \
    "tq2_0     f16    .         TQ2_0_uniform"; do

    CTK=$(echo $cfg | awk '{print $1}')
    CTV=$(echo $cfg | awk '{print $2}')
    FRAC=$(echo $cfg | awk '{print $3}')
    NAME=$(echo $cfg | awk '{print $4}')

    echo ""
    echo ""
    echo "====== $NAME  (ctk=$CTK ctv=$CTV frac=$FRAC) ======"
    echo "Started: $(date)"

    if [ "$FRAC" = "." ]; then
        $BIN/llama-perplexity -m "$MODEL" -f "$WIKI" -c 2048 -ngl 99 \
            -ctk $CTK -ctv $CTV 2>&1 | \
            grep -E "KV self size|adaptive K-cache|auto-detect summary|outlier K policy|spectrum skew|Final estimate|^\[[0-9]+\]"
    else
        $BIN/llama-perplexity -m "$MODEL" -f "$WIKI" -c 2048 -ngl 99 \
            -ctk $CTK -ctv $CTV --kv-outlier-frac $FRAC 2>&1 | \
            grep -E "KV self size|adaptive K-cache|auto-detect summary|outlier K policy|spectrum skew|Final estimate|^\[[0-9]+\]"
    fi

    echo "Finished: $(date)"
done

echo ""
echo "======================================================================"
echo "All configs complete: $(date)"
echo "======================================================================"

} 2>&1 | tee "$OUT"
