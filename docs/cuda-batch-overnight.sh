#!/bin/bash
# CUDA Batch Overnight Run — Cross-architecture TQ validation on RTX 4090
#
# Tests 3 models × 6 KV cache configs = 18 total 160-chunk PPL runs on
# WikiText-2. Uses the Phase 3.5 V1 adaptive policy with head_dim-aware
# defaults already committed in the feature branch.
#
# EXPECTED RUNTIME: ~2-3 hours on RTX 4090 (prompt eval ~7400 t/s, so
# 160 × 2048 tokens per config = 327k tokens per config, ~44 sec per
# config prompt eval + some decode time).
#
# PREREQUISITES (run once after cloning on a new Vast.ai instance):
#   cd ~/Lean_llama.cpp && git checkout feature/tq2-outlier-tiered
#   mkdir -p build && cd build
#   cmake .. -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
#   cmake --build . -j$(nproc) --target llama-perplexity llama-cli
#
# Dataset:
#   wget https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip
#   unzip wikitext-2-raw-v1.zip -d wikitext-2-raw
#
# Then run this script from Lean_llama.cpp root.

set -e

# ─── Configuration ──────────────────────────────────────────────────────
MODELS_DIR="${MODELS_DIR:-$HOME/models}"
WIKI="${WIKI:-$HOME/Lean_llama.cpp/wikitext-2-raw/wiki.test.raw}"
BIN="./build/bin/llama-perplexity"
OUT="docs/cuda-batch-results.txt"
NGL="${NGL:-99}"        # Full GPU offload
CTX="${CTX:-2048}"

# ─── Model URLs (unsloth Q4_K_M) ────────────────────────────────────────
declare -A MODEL_URLS=(
    ["Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"]="https://huggingface.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"
    ["Qwen3-8B-Q4_K_M.gguf"]="https://huggingface.co/unsloth/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf"
    ["gemma-3-4b-it-Q4_K_M.gguf"]="https://huggingface.co/unsloth/gemma-3-4b-it-GGUF/resolve/main/gemma-3-4b-it-Q4_K_M.gguf"
)

# ─── Sanity checks ──────────────────────────────────────────────────────
mkdir -p "$MODELS_DIR"

if [ ! -f "$WIKI" ]; then
    echo "ERROR: WikiText-2 not found at $WIKI"
    echo "Download with:"
    echo "  wget https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip"
    echo "  unzip wikitext-2-raw-v1.zip -d wikitext-2-raw"
    exit 1
fi

if [ ! -x "$BIN" ]; then
    echo "ERROR: $BIN not found or not executable"
    echo "Build first with:"
    echo "  cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON"
    echo "  cmake --build build -j\$(nproc)"
    exit 1
fi

# ─── Download missing models ────────────────────────────────────────────
for MODEL_FILE in "${!MODEL_URLS[@]}"; do
    LOCAL="$MODELS_DIR/$MODEL_FILE"
    if [ ! -f "$LOCAL" ]; then
        echo "Downloading $MODEL_FILE ..."
        wget -q --show-progress "${MODEL_URLS[$MODEL_FILE]}" -O "$LOCAL" || {
            echo "FAILED to download $MODEL_FILE — skipping"
            rm -f "$LOCAL"
            continue
        }
    else
        echo "Found: $MODEL_FILE ($(du -h "$LOCAL" | cut -f1))"
    fi
done

# ─── Run one config ─────────────────────────────────────────────────────
run_config() {
    local model_label="$1"
    local model_path="$2"
    local cfg_label="$3"
    local ctk="$4"
    local ctv="$5"
    local extra="$6"

    echo ""
    echo "====== $model_label : $cfg_label ======"
    echo "Started: $(date)"
    echo "ctk=$ctk ctv=$ctv $extra"

    if [ ! -f "$model_path" ]; then
        echo "SKIP: model not found"
        return
    fi

    # shellcheck disable=SC2086
    "$BIN" -m "$model_path" -f "$WIKI" -c "$CTX" -ngl "$NGL" \
        -ctk "$ctk" -ctv "$ctv" $extra 2>&1 | \
        grep -E "KV self size|Final estimate|adaptive K-cache|auto-detect summary|outlier K policy|downgrad|skew|spectrum"

    echo "Finished: $(date)"
}

# ─── Main batch ─────────────────────────────────────────────────────────
{
echo "====================================================================="
echo "CUDA Batch Overnight — 3 models × 6 configs (18 total 160-chunk runs)"
echo "Date:     $(date)"
echo "Host:     $(hostname)"
echo "GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Branch:   $(git branch --show-current)"
echo "Commit:   $(git log --oneline -1)"
echo "====================================================================="

for model_entry in \
    "Mistral-7B:$MODELS_DIR/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf" \
    "Qwen3-8B:$MODELS_DIR/Qwen3-8B-Q4_K_M.gguf" \
    "Gemma3-4B:$MODELS_DIR/gemma-3-4b-it-Q4_K_M.gguf"; do

    MODEL_LABEL="${model_entry%%:*}"
    MODEL_PATH="${model_entry#*:}"

    if [ ! -f "$MODEL_PATH" ]; then
        echo ""
        echo "====== $MODEL_LABEL : MODEL MISSING ($MODEL_PATH) ======"
        continue
    fi

    run_config "$MODEL_LABEL" "$MODEL_PATH" "F16_baseline"      f16   f16 ""
    run_config "$MODEL_LABEL" "$MODEL_PATH" "TQ4_0_uniform"     tq4_0 f16 ""
    run_config "$MODEL_LABEL" "$MODEL_PATH" "TQ3_0_uniform"     tq3_0 f16 ""
    run_config "$MODEL_LABEL" "$MODEL_PATH" "TQ2_1_uniform"     tq2_1 f16 ""
    run_config "$MODEL_LABEL" "$MODEL_PATH" "V1_adaptive"       tq2_0 f16 "--kv-outlier-frac -1"
    run_config "$MODEL_LABEL" "$MODEL_PATH" "TQ2_0_uniform"     tq2_0 f16 ""
done

echo ""
echo "====================================================================="
echo "All configs complete: $(date)"
echo "====================================================================="
} 2>&1 | tee "$OUT"

echo ""
echo "Results saved to: $OUT"
echo "Summary: grep 'Final estimate' $OUT"
