# LeanKV E2B Campaign Runbook — Gemma 4 E2B/E4B (2026-07)

**Context.** The Gemma 3-4B ablation (`docs/leankv-kv-importance-ablation-2026-07.md`)
killed magnitude-importance: A3 lost to A1 by 18% KLD at 30σ (importance ≈ variance,
log-corr 0.82). Shipping baseline is **A1 + `--norm robust`**. This runbook executes
the pivot the July plan pre-registered: test the **#1-ranked thread — reuse-aware
allocation** — on Gemma 4's cross-layer KV sharing, plus the rank-pressure ladder for
its 512-dim global heads. Everything below runs on a real E2B (or E4B) GGUF; the
tooling is already in-tree (kvimp collector, `LEANKV_KV_PLAN` consumer, robust norm,
A1R arm).

**Prereqs.** Tree at or after the A1R commit. Canonical dataset only:
`md5 wiki.test.raw` = `7c0137fc034ddbc56a296bce31b4f7fb` (1,290,590 bytes).
Calibration corpus from `wiki.valid.raw` (held out from eval).

---

## Step 0 — Read the beast (minutes)

Load the model once and record from the loader + kvimp banner:

```bash
LEANKV_KVIMP=1 LEANKV_KVIMP_PATH=/tmp/probe.json \
  ./build/bin/llama-cli -m gemma-4-e2b.gguf -ngl 0 -c 512 -n 1 -p "hello"
```

Record: `n_embd`, `n_head`, head_dim per layer type (expect 256 local / 512 global),
`n_layer_kv_from_start` (expect 15 owned / 20 shared on E2B), rope fractions
(expect 0.25 on globals), K=V tying (expect none on E2B). This grades the
external-article numbers (hidden 1,536? window 512?) and sets rank expectations:
aggregate Q rank ≤ n_embd ⇒ average global-head fill ≈ n_embd/(8·512).

Sanity-check the kvimp banner: owned/shared split correct, `reuse_count` populated
on owners, `rope_fraction` < 1 on globals, partition fields present in the JSON.
**VERIFY-A:** the owner mapping (same-type backward scan, mirrored from
`gemma4_mtp_target_kv_layer`) matches the shared-layer attention path in the graph.
**VERIFY-B:** rope'd channels are the leading `rope_n_rot` dims per head.

## Step 1 — Within-head active dims (~10 min)

```bash
LEANKV_CALIBRATION_DUMP=1 LEANKV_CALIBRATION_DUMP_PATH=/tmp/e2b_calib.bin \
  ./build/bin/llama-cli -m gemma-4-e2b.gguf -ngl 0 -c 1024 -n 4 -f calib.txt
python3 <LeanKV>/scripts/analyze_k_calib.py /tmp/e2b_calib.bin
```

Report r95/r99 **split by layer type** (local vs global). Prediction: globals ≪
locals; if global r99 lands near n_embd/n_head territory, the rank-pressure story
holds and Step 4's ladder is armed. Layer 0: expect an anomaly (sink and/or rank
collapse — it has been special on Qwen3-4B and Gemma 3).

## Step 2 — Calibrate for allocation (~5 min)

```bash
head -c 31000 wiki.valid.raw > calib_long.txt
LEANKV_KVIMP=1 LEANKV_KVIMP_PATH=kv_stats_e2b.json \
  ./build/bin/llama-cli -m gemma-4-e2b.gguf -ngl 0 -c 8192 -n 1 -f calib_long.txt
```

## Step 3 — The reuse go/no-go (A1 vs A1R)

A1R = variance × reuse^w, importance OFF — isolates the reuse signal on the winning
baseline. Both arms share the budget, the norm, and the ladder.

```bash
python3 kit-v2/kv_bit_allocator.py kv_stats_e2b.json --arm A1  --bpw 3.0 --bmax 4 \
    --norm robust --emit-types plan_A1.types
python3 kit-v2/kv_bit_allocator.py kv_stats_e2b.json --arm A1R --bpw 3.0 --bmax 4 \
    --norm robust --emit-types plan_A1R.types
diff plan_A1.types plan_A1R.types    # expect bits shifted toward owned/high-reuse layers

# F16 base once, then one KLD run per arm:
./build/bin/llama-perplexity -m gemma-4-e2b.gguf -f wiki.test.raw -c 2048 \
    --kl-divergence-base base_f16_e2b.kld
for arm in A1 A1R; do
  LEANKV_KV_PLAN=plan_${arm}.types ./build/bin/llama-perplexity -m gemma-4-e2b.gguf \
    -f wiki.test.raw -c 2048 --kl-divergence-base base_f16_e2b.kld --kl-divergence \
    | grep -E "Mean    KLD|Same top"
done
```

**GO:** A1R Mean KLD < A1 at matched bpw ⇒ reuse-aware allocation is real — the
first Gemma-4-native allocation result. If borderline, sweep `--w-reuse 0.5|1|2`.
**NO-GO:** proceed to Step 4 regardless — the ladder does not depend on reuse.

Note: the rank gate will WARN (not rewrite) on rank-bounded layers under explicit
plans; that warning is expected on E2B globals and is itself data.

## Step 4 — The global-layer ladder (if Step 1 shows collapse)

Cheapest-first; each rung is one plan file or one existing mechanism:

1. **Layer-type policy** — locals tq3_0, globals tq4_0 (hand-write the .types file
   from the Step-0 layer map, or gate output).
2. **Empirical codebook per layer type** — runtime-swappable LUTs already in-tree
   (the Qwen3-4B fix, +1.75 dB).
3. **A5 p-RoPE partition** — collector already emits rope/pass stats; consumer-side
   sub-head types not yet implemented (future work; treat as analysis-only for now).
4. **Last resort:** Phase-7b SVD rotation on the ~3 global owner tensors only.

## Step 5 — Long-context validation

Whatever wins Steps 3–4: re-measure at 8K–32K context (KLD + a needle-style
retrieval probe). E2B's 512-token local windows ring-cap local KV, so **global-layer
KV dominates memory at long context** — the regime the whole exercise is for.

## Reporting

Append results to this doc (or a sibling results doc) with: model sha, tree commit,
dataset md5, per-arm Mean KLD ± err and Same-top %, and the Step-0/1 hparam + rank
tables. Update the ablation doc's conclusion if the reuse verdict lands.
