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

---

# RESULTS — 2026-07-17 (Ryzen 7 7735U, CPU AVX2)

| | |
|---|---|
| Model | `gemma-4-E2B-it-Q4_K_M.gguf` (unsloth/gemma-4-E2B-it-GGUF) |
| Model sha256 | `740185b21d22ceb83a11c3aa62ad5842ef32c70f6096d756bbee85a1e4ec34b8` |
| Tree | `e2543108` + this doc |
| Dataset | canonical `wiki.test.raw`, md5 `7c0137fc034ddbc56a296bce31b4f7fb` |
| Calibration | kvimp on `wiki.valid.raw`, 7,608 tok (allocation); 732-tok `calib.txt` (rank dump) |

## Step 0 — geometry (predictions graded)

Loader + kvimp confirmed: arch `gemma4`, 35 layers, n_embd **1536**, n_head 8, MQA
(n_head_kv 1), head_dim **256 local / 512 global**, sliding_window 512,
**15 owned / 20 shared** KV layers, globals at il 4,9,14,19,24,29,34.

- ✅ Predicted hidden 1536, 15/20 owned/shared split, 256/512 head dims — exact.
- ❌ **rope_fraction = 1.0 on every layer** (rope dim = head_dim: 512/512, 256/256).
  The "0.25 p-RoPE on globals" prediction is falsified → **Step-4 rung 3 (A5
  p-RoPE partition) is retired on E2B** — there is nothing to partition.
- **Reuse concentration** (the A1R signal): layer 13 owns KV for **17** downstream
  layers, layer 14 for **5**; all other owners reuse=1. Robust-norm sink detection
  flagged layers [0, 13] (k-importance) and [0] (variance).
- Loader warning (relevant, correct): `13 layer(s) have q_dim < head_dim
  (rank-bounded KV subspace)` — q_dim = 1536/8 = 192 < 256/512 everywhere.

## Step 3 — reuse go/no-go: **GO** (decisive)

Matched 3.0 bpw (`--bmax 4`, `--norm robust`), K+V per-layer plans, 144 chunks,
KLD vs same-model F16 base. A1R = variance × reuse, importance OFF.

| Arm | Mean KLD ↓ | Same-top ↑ | KV plan on owners 13/14 |
|-----|-----------|-----------|------------------------|
| A1 (uniform+outlier) | 1.7154 ± 0.0059 | 54.82% | 13: tq2/tq3 · 14: tq4/tq3 |
| **A1R (reuse)** | **0.9934 ± 0.0042** | **65.07%** | 13: tq4/tq4 · 14: tq4/tq4 |

**A1R cuts Mean KLD 42%** (Δ = 0.722 ≈ 99σ) and gains 10.2 points same-top by
re-protecting exactly the two shared owners that variance-only allocation starves
(layer 13 looks quiet locally but feeds 17 downstream layers). This is the first
Gemma-4-native allocation result, and the first arm ever to beat the A1 baseline
in this program (magnitude-importance lost on Gemma 3-4B; reuse wins on E2B).
`--w-reuse` sweep skipped per the decision rule (not borderline).

Caveats: absolute F16 PPL on raw WikiText is 153.36 (it-tuned, on-device
multimodal model) — internal same-model KLD comparisons are unaffected. E2B at
3.0 bpw is much more quantization-sensitive than Gemma 3-4B (A1 KLD 1.72 vs
0.39): consistent with every layer being rank-bounded (TQ3/TQ2 unsafe territory
per the Q-dim gate) — see Step 1.

## Step 1 — empirical K-rank (r95/r99), local vs global

732 K-vectors/layer, post-RoPE post-K-norm, f16 cache (clean probe), SVD per
layer. Full table: `e2b_rank_report.txt` (local artifact).

| Group | n | r95 min/med/max (fill) | r99 min/med/max (fill) |
|-------|---|------------------------|------------------------|
| local (256) | 12 | 116/**134**/147 (**52.1%**) | 192/206/213 (80.3%) |
| global (512) | 3 | 167/**196**/223 (**38.3%**) | 289/334/347 (65.2%) |

- **Prediction confirmed almost exactly at 95% energy:** predicted global fill
  n_embd/(8·512) = 37.5%; measured **38.3%** (median r95 196 ≈ q_dim 192).
- Notable: with MQA, the 8 query heads *could* jointly probe up to the full 512
  dims — yet r95 says they use ~196. The heads' Q-subspaces overlap heavily.
  The r99 tail (334) shows real energy beyond q_dim, so truncation should target
  r95+margin, not r99.
- **The most-reused owners are also the most compressible:** layer 14 (5×
  global) has the lowest global rank (r95 167, steepest decay 3.3e-04); layer 13
  (17× local) is near-lowest of the locals (r99 197, decay 7.3e-04).
- Layer 0 shows **no rank anomaly** (r99 206, in family) — unlike Qwen3-4B
  (r99 = 3). Its sink behavior on E2B is variance/importance-only.

**The low-rank ladder (Step 4) is armed for the global layers:** a rank-224
projection (r95 max + margin) on the 3 global owners retains ≥95% energy while
cutting global K storage 56% *before* quantization — and globals dominate KV
memory at long context (locals are ring-capped at the 512 window). Combined
strategy suggested by Steps 1+3 jointly: low-rank the globals, spend the saved
bits protecting the shared owners (A1R already does the latter).

## Verdict / next

1. **Ship-on-E2B allocation: A1R + `--norm robust`** (42% better than A1).
2. Step 4 next rungs (in order of cheapness): layer-type policy plan is subsumed
   by A1R; **empirical per-layer-type codebooks** (rung 2) and **rank-224
   low-rank projection on globals** (rung 4, now justified by measured r95) are
   the live follow-ups. Rung 3 (p-RoPE) retired — no partition exists.
3. Step 5 (long-context validation) still pending — it is where the global-layer
   levers pay off.
4. Cross-arch verdict update for the ablation doc: reuse is the first signal to
   beat A1, but it only exists on shared-KV architectures. On single-owner
   models (Gemma 3, Qwen) A1 remains the shipping config.
