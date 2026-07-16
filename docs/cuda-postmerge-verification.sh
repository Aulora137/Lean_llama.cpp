#!/bin/bash
# CUDA Post-Merge TQ Verification — RTX 4090 (Vast.ai or any CUDA box)
#
# Validates the 2026-07 upstream merge (main @ >= 69372a5d) on CUDA — the
# last unvalidated backend. CPU (Ryzen) and Metal (M2) already passed.
#
# SETUP on a fresh instance:
#   git clone https://github.com/hchengit/Lean_llama.cpp.git ~/Lean_llama.cpp
#   cd ~/Lean_llama.cpp
#   cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
#   cmake --build build -j$(nproc) --target llama-perplexity llama-cli
#   wget https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip
#   unzip wikitext-2-raw-v1.zip -d wikitext-2-raw
#   bash docs/cuda-postmerge-verification.sh
#
# Models expected in $MODELS_DIR (download if missing — URLs below):
#   Qwen3.5-9B-Q4_K_M.gguf   (primary: hybrid arch, deltanet, TQ-friendliest)
#   Mistral-7B-Instruct-v0.3-Q4_K_M.gguf  (gold: cross-backend reference)

set -e

MODELS_DIR="${MODELS_DIR:-$HOME/models}"
WIKI="${WIKI:-$HOME/Lean_llama.cpp/wikitext-2-raw/wiki.test.raw}"
BIN="./build/bin/llama-perplexity"
OUT="docs/cuda-postmerge-2026-07-results.txt"
CTX="${CTX:-2048}"
NGL="${NGL:-99}"

# ─── STEP ZERO: dataset + model provenance (the 2026-04 taint lesson) ───
# Chunk-count equality does NOT prove dataset equality. Hard-fail here.
CANON_MD5="7c0137fc034ddbc56a296bce31b4f7fb"
CANON_BYTES="1290590"

md5=$(md5sum "$WIKI" | cut -d' ' -f1)
bytes=$(stat -c %s "$WIKI" 2>/dev/null || stat -f %z "$WIKI")
if [ "$md5" != "$CANON_MD5" ] || [ "$bytes" != "$CANON_BYTES" ]; then
    echo "FATAL: wiki.test.raw is NOT the canonical dataset."
    echo "  got  md5=$md5 bytes=$bytes"
    echo "  want md5=$CANON_MD5 bytes=$CANON_BYTES"
    echo "Re-fetch: https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip"
    exit 1
fi
echo "dataset: CANONICAL ($md5, $bytes bytes)"

# Model revision check. Reference sha256 from the Ryzen node (the copy the
# CPU gate numbers were produced with). If the 9B sha differs, PPL deltas
# vs the CPU gate are NOT interpretable — stop and reconcile revisions
# (April's CUDA F16 7.1404 vs CPU 7.2591 gap is suspected to be exactly
# this; see docs/metal-qwen35-tq4-tq3-results.txt).
QWEN_REF_SHA="03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8"

QWEN="$MODELS_DIR/Qwen3.5-9B-Q4_K_M.gguf"
MISTRAL="$MODELS_DIR/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"

{
echo "=== CUDA post-merge verification — $(git rev-parse --short HEAD) ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "dataset md5: $md5 (canonical)"
for m in "$QWEN" "$MISTRAL"; do
    if [ -f "$m" ]; then
        sha=$(sha256sum "$m" | cut -d' ' -f1)
        echo "model $(basename "$m") sha256: $sha"
        if [ "$m" = "$QWEN" ] && [ "$sha" != "$QWEN_REF_SHA" ]; then
            echo "  WARNING: differs from Ryzen reference $QWEN_REF_SHA"
            echo "  -> comparisons vs the CPU gate are void; transfer the Ryzen copy or reconcile."
        fi
    else
        echo "model $(basename "$m"): MISSING — skipping its runs"
    fi
done
} | tee "$OUT"

run() { # run <model> <label> <extra args...>
    local model="$1" label="$2"; shift 2
    [ -f "$model" ] || return 0
    echo "--- $(basename "$model") $label ---" | tee -a "$OUT"
    "$BIN" -m "$model" -f "$WIKI" -c "$CTX" -ngl "$NGL" "$@" 2>&1 \
        | grep -E "Final estimate|KV self size" | tee -a "$OUT"
}

# ─── Qwen 3.5-9B (primary) ─────────────────────────────────────────────
# Expected (April CUDA, K-only, sha caveat above): F16 7.1404 · TQ4 7.1453 · TQ3 7.1663
# Cross-backend anchors (canonical, same-sha): CPU F16/F16 7.2591, TQ4/TQ4 7.2912,
# TQ3/TQ3 7.3409 · Metal F16 7.2533, TQ4/F16 7.2965, TQ3/F16 7.3287
run "$QWEN" "F16/F16"
run "$QWEN" "TQ4/F16"  -ctk tq4_0
run "$QWEN" "TQ3/F16"  -ctk tq3_0
run "$QWEN" "TQ4/TQ4"  -ctk tq4_0 -ctv tq4_0
run "$QWEN" "TQ3/TQ3"  -ctk tq3_0 -ctv tq3_0

# ─── Mistral 7B (gold cross-backend reference) ─────────────────────────
# Expected (April CUDA actuals): F16 5.1638 · TQ4/F16 5.1781 · TQ3/F16 5.2464
# Metal canonical re-run matched TQ4 to 4 decimals (5.1781), TQ3 within 0.002.
run "$MISTRAL" "F16/F16"
run "$MISTRAL" "TQ4/F16"  -ctk tq4_0
run "$MISTRAL" "TQ3/F16"  -ctk tq3_0

echo "=== done — full summary in $OUT ==="
echo "Commit it back:  git add $OUT && git commit -m 'docs: CUDA post-merge TQ verification results' && git push origin main"
