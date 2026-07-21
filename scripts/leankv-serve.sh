#!/usr/bin/env bash
# leankv-serve.sh — auto-configure llama-server's KV quantization from the
# geometry prober (kit-v2/kv_policy.py), then exec the server.
#
# The prober reads the model's GGUF header, classifies its architecture family,
# and picks the measured-correct KV cache types (see docs/leankv-adaptive-menu-
# 2026-07.md). The block-scale statistic (mse_opt@2b / amax@>=3b) is already the
# engine default (LEANKV_TQ_SCALE=auto), so this wrapper only injects -ctk/-ctv.
#
# Usage:
#   scripts/leankv-serve.sh <model.gguf> [--target-bpw N] [-- <extra llama-server args>]
#
#   --target-bpw 4  (default)  -> the ship floor (TQ4/TQ4 pure)
#   --target-bpw 3|2           -> the family-appropriate aggressive tier, with
#                                 the prober's refusals/gate honored
# Everything after `--` is passed through to llama-server verbatim.
#
# Env:
#   KV_POLICY   path to kv_policy.py            (default: <repo>/kit-v2/kv_policy.py)
#   LLAMA_SERVER path to the llama-server binary (default: <repo>/build/bin/llama-server)
#   LEANKV_SERVE_DRYRUN=1  print the resolved command and exit (don't launch)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KV_POLICY="${KV_POLICY:-$HERE/kit-v2/kv_policy.py}"
LLAMA_SERVER="${LLAMA_SERVER:-$HERE/build/bin/llama-server}"

if [ $# -lt 1 ]; then
    echo "usage: $0 <model.gguf> [--target-bpw N] [-- <extra llama-server args>]" >&2
    exit 2
fi

MODEL="$1"; shift
TARGET_BPW=4
EXTRA=()
while [ $# -gt 0 ]; do
    case "$1" in
        --target-bpw) TARGET_BPW="$2"; shift 2 ;;
        --) shift; EXTRA=("$@"); break ;;
        *) EXTRA+=("$1"); shift ;;
    esac
done

[ -f "$MODEL" ]        || { echo "leankv-serve: model not found: $MODEL" >&2; exit 1; }
[ -f "$KV_POLICY" ]    || { echo "leankv-serve: kv_policy.py not found: $KV_POLICY" >&2; exit 1; }
[ -x "$LLAMA_SERVER" ] || { echo "leankv-serve: llama-server not found/executable: $LLAMA_SERVER" >&2; exit 1; }

# Probe the model and pull the chosen KV types + family + first-line rationale.
POLICY_JSON="$(python3 "$KV_POLICY" "$MODEL" --target-bpw "$TARGET_BPW" --json 2>/dev/null)" \
    || { echo "leankv-serve: kv_policy probe failed for $MODEL" >&2; exit 1; }

read -r FAMILY KTYPE VTYPE HAS_KV <<EOF
$(printf '%s' "$POLICY_JSON" | python3 -c '
import json, sys
d = json.load(sys.stdin)
ct = d.get("chosen_tier", {})
fam = d.get("family", "?")
k = ct.get("k") or ""
v = ct.get("v") or ""
has_kv = "1" if (fam not in ("no-kv-cache",) and k and v) else "0"
print(fam, k, v, has_kv)
')
EOF

echo "── leankv-serve: KV policy for $(basename "$MODEL") ─────────────────────────" >&2
echo "   family        : $FAMILY" >&2
echo "   target-bpw    : $TARGET_BPW" >&2
if [ "$HAS_KV" = "1" ]; then
    echo "   KV cache      : -ctk $KTYPE -ctv $VTYPE   (scale = engine default: mse_opt@2b/amax@>=3b)" >&2
else
    echo "   KV cache      : none (encoder / no causal KV) — not setting -ctk/-ctv" >&2
fi
# Surface the prober's rationale/warnings so the operator sees WHY (esp. refusals/gate).
printf '%s' "$POLICY_JSON" | python3 -c '
import json, sys
d = json.load(sys.stdin)
for f in d.get("flags", []):
    print("   note          : " + f, file=sys.stderr)
' || true
echo "────────────────────────────────────────────────────────────────────────────" >&2

CMD=("$LLAMA_SERVER" --model "$MODEL")
[ "$HAS_KV" = "1" ] && CMD+=(-ctk "$KTYPE" -ctv "$VTYPE")
CMD+=("${EXTRA[@]}")

if [ "${LEANKV_SERVE_DRYRUN:-0}" = "1" ]; then
    printf '%q ' "${CMD[@]}"; echo
    exit 0
fi
exec "${CMD[@]}"
