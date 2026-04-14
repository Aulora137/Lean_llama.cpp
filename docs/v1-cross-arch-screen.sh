#!/bin/bash
# V1 cross-architecture screening (3 chunks): validate that 1.5× threshold
# generalizes beyond Mistral 7B.

set -e

WIKI=/home/junc/LeanKV/prototype/eval/wikitext-2-raw/wiki.test.raw
BIN=./build/bin
OUT=docs/v1-cross-arch-screen.txt

screen_model() {
    local NAME="$1"
    local MODEL="$2"
    local LABEL="$3"

    echo ""
    echo "====== $NAME ($LABEL) ======"

    if [ ! -f "$MODEL" ]; then
        echo "MISSING: $MODEL"
        return
    fi

    echo ""
    echo "--- F16/F16 ---"
    $BIN/llama-perplexity -m "$MODEL" -f "$WIKI" -c 2048 -ngl 0 \
        -ctk f16 -ctv f16 -t 8 --chunks 3 2>&1 | \
        grep -E "KV self size|Final estimate|downgrad"

    echo ""
    echo "--- TQ2_0/F16 uniform ---"
    $BIN/llama-perplexity -m "$MODEL" -f "$WIKI" -c 2048 -ngl 0 \
        -ctk tq2_0 -ctv f16 -t 8 --chunks 3 2>&1 | \
        grep -E "KV self size|Final estimate|downgrad"

    echo ""
    echo "--- V1 adaptive (1.5×) ---"
    LEANKV_OUTLIER_METRIC=0 LEANKV_OUTLIER_THRESHOLD=1.5 \
        $BIN/llama-perplexity -m "$MODEL" -f "$WIKI" -c 2048 -ngl 0 \
        -ctk tq2_0 -ctv f16 --kv-outlier-frac -1 -t 8 --chunks 3 2>&1 | \
        grep -E "adaptive K-cache|auto-detect summary|KV self size|Final estimate|outlier K policy|downgrad"

    echo ""
    echo "--- TQ2_1/F16 uniform ---"
    $BIN/llama-perplexity -m "$MODEL" -f "$WIKI" -c 2048 -ngl 0 \
        -ctk tq2_1 -ctv f16 -t 8 --chunks 3 2>&1 | \
        grep -E "KV self size|Final estimate|downgrad"
}

{
echo "============================================================="
echo "V1 Cross-Architecture Screening (3 chunks)"
echo "Date: $(date)"
echo "Commit: $(git log --oneline -1)"
echo "============================================================="

screen_model "Qwen3.5-9B" \
    "/home/junc/Aulora/bitcoin-node-stack/models/Qwen3.5-9B-Q4_K_M.gguf" \
    "head_dim=256, hybrid Mamba+attention, 8 attn layers"

screen_model "Gemma3-4B" \
    "/home/junc/Aulora/bitcoin-node-stack/models/gemma-3-4b-it-Q4_K_M.gguf" \
    "head_dim=256, dense, 34 layers"

screen_model "Qwen2.5-0.5B" \
    "/home/junc/LeanInfer/models/qwen2.5-0.5b-instruct-q4_k_m.gguf" \
    "head_dim=64, dense, 24 layers"

echo ""
echo "============================================================="
echo "Screening complete: $(date)"
echo "============================================================="
} 2>&1 | tee "$OUT"
