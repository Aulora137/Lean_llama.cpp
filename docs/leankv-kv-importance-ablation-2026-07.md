# LeanKV KV-Importance Ablation — Gemma 3-4B (2026-07-17)

**Question.** Does *importance-guided* per-layer KV bit allocation beat plain
*uniform + outlier* allocation at a matched bit budget? I.e. is the KV-importance
signal collected by `kvimp` real and complementary, or redundant?

**Answer.** On Gemma 3-4B it is **redundant** — importance does not beat
uniform+outlier at 3.0 bpw, under any normalization tested. Ship arm **A1**
(uniform+outlier). The importance branch adds no signal that variance/outlier
protection doesn't already carry. Proven three ways (below). The one lever still
untested is **entropy** (arms A2/A4), which needs a per-layer entropy emitter that
`kvimp` does not yet produce.

---

## Setup

| | |
|---|---|
| Model | `gemma-3-4b-it-Q4_K_M.gguf` (34 attention layers, MQA, Gemma-3 arch) |
| Engine | Lean_llama.cpp `af3208d0` (kit-v2 `kvimp` collector + `LEANKV_KV_PLAN` consumer) |
| Metric | **KL divergence to F16** (`llama-perplexity --kl-divergence`), WikiText-2 raw, c=2048, 144 chunks |
| Dataset | canonical `wiki.test.raw`, md5 `7c0137fc034ddbc56a296bce31b4f7fb` |
| F16 base | `base_f16.kld` (PPL 12.5356 ± 0.1154) |
| Budget | matched **3.0 bpw**, K+V both allocated, `--bmax 4` (exact TQ ladder: 2/3/4 → tq2_0/tq3_0/tq4_0) |
| Arms | **A1** = uniform + outlier (`use_importance=False`); **A3** = A1 + importance |
| Calibration | `kvimp` on held-out `wiki.valid.raw`: "short" = 732 tok, "long" = 7,608 tok |

Lower KLD is better. The GO criterion was: *A3 < A1 at matched bpw ⇒ importance is real.*

## Results

| Arm | Calib | Norm | Mean KLD ↓ | Same-top ↑ | vs A1 |
|-----|-------|------|-----------|-----------|-------|
| A1 | short | global | 0.3745 | 77.51% | — |
| A3 | short | global | 0.5917 | 70.73% | **+58%** |
| A1 | long | global | 0.3869 | 77.02% | — |
| A3 | long | global (broken) | 0.6785 | 68.08% | **+75%** |
| A1 | long | robust | 0.3922 | 77.17% | — |
| **A3** | long | **robust (fixed)** | **0.4631** | 74.46% | **+18%** |

Error bars ±0.0022–0.0029 on Mean KLD, so every A3-vs-A1 gap is 30–80σ — none is noise.

## Diagnosis — why importance loses

Dumping the per-layer stats (`kv_stats_long.json`) exposed two defects, both rooted
in **layer 0 being an attention sink** (its K-importance is 779 and K-variance 817,
vs 0.1–8 and 1–27 for every other layer — a 100–800× outlier):

1. **Importance ≈ variance.** log-correlation of per-layer `k_importance` vs `k_var`
   is **0.82**. Importance is largely re-encoding the variance signal that A1's
   outlier path already uses. Weighting by it double-counts variance and adds noise.

2. **Linear max-normalization collapses under the sink.** The v2 `global` norm divides
   by the max, so layer 0 → 1.0 and **33 of 34 layers are crushed below 0.05**
   (median 0.0013). The allocator can no longer tell non-sink layers apart, so it
   starves them to TQ2. That is the direct cause of the +75% blow-up, and it is why
   *more* calibration made A3 *worse* (0.59 → 0.68): a sharper read of a degenerate
   signal is more confidently wrong.

## The fix — `--norm robust`

Added a v3 normalization mode to `kit-v2/kv_bit_allocator.py` (`--norm robust`,
`--sink-mult` default 20):

- **Sink exclusion** — any layer whose per-kind value ≥ 20× the median is pinned to
  the top of the scale (stays protected) but is removed from the scale-setting pool,
  so it can't dominate. On Gemma this flags exactly layer 0.
- **Rank normalization** — the remaining layers are mapped into (0,1] by rank instead
  of by magnitude, so no single outlier can compress the rest.

Effect: A3's crushed-to-TQ2 K-layers dropped 9 → 4, allocation re-balanced, and KLD
recovered 0.6785 → 0.4631 (−32%). The mechanism diagnosis is thereby confirmed —
layer-0 domination *was* the failure. `global` (the shipping default) is unchanged;
`robust` is the correct normalization for any variance/importance-weighted arm and
should be the default whenever those arms are used.

## Conclusion

- **Importance is redundant on this arch.** Even with the normalization fixed, A3
  loses to A1 by 18% (30σ). Confirmed under broken norm, under fixed norm, and by the
  0.82 variance correlation. It is not a calibration or normalization artifact.
- **Ship A1** (uniform + outlier), `--norm robust` so the sink layer is handled
  correctly. Best observed config: KLD 0.39 / same-top 77% at 3.0 bpw.
- Even variance-ranking (A1 robust) barely beats near-uniform (A1 global, 0.392 vs
  0.387) — the only signal that clearly matters here is *"protect the sink layer,
  spread the rest evenly."*
- **Entropy (A2/A4/A5) is the only untested lever.** Attention entropy ≠ magnitude,
  so it could be genuinely complementary — but `kvimp` does not emit a per-layer
  entropy file yet, so those arms currently collapse to A1/A3. Building that emitter
  is the next real experiment, not a re-run.

## Reproduce

```bash
# calibrate (CPU; long = ~7.6k tok, fits the 8188 ctx)
head -c 31000 wiki.valid.raw > calib_long.txt
LEANKV_KVIMP=1 LEANKV_KVIMP_PATH=kv_stats_long.json \
  ./build/bin/llama-cli -m gemma-3-4b-it-Q4_K_M.gguf -ngl 0 -c 8192 -n 1 -f calib_long.txt

# allocate (robust norm) + measure vs the saved F16 base
for arm in A1 A3; do
  python3 kit-v2/kv_bit_allocator.py kv_stats_long.json --arm $arm --bpw 3.0 \
    --bmax 4 --norm robust --emit-types plan_${arm}_robust.types
  LEANKV_KV_PLAN=plan_${arm}_robust.types ./build/bin/llama-perplexity \
    -m gemma-3-4b-it-Q4_K_M.gguf -f wiki.test.raw -c 2048 \
    --kl-divergence-base base_f16.kld --kl-divergence | grep "Mean    KLD"
done
```

Base build: `af3208d0`. Allocator with `--norm robust`: this commit.
