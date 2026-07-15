# Expected PPL Targets — Quick Validation Reference

Use this table to verify CUDA/Metal batch runs produce correct results.
All numbers are 160-chunk WikiText-2, n_ctx=2048, Mistral 7B Q4_K_M unless
noted otherwise.

**BEFORE COMPARING ANYTHING: verify the dataset.**
`md5 wikitext-2-raw/wiki.test.raw` must be
`7c0137fc034ddbc56a296bce31b4f7fb` (canonical, 1,290,590 bytes).
A non-canonical variant (md5 9c7a807f..., 1,292,014 bytes) was silently
fetched onto the M2 on 2026-04-14 and shifted Qwen 3.5-9B F16 PPL by
+1.27 while still producing the same 145-chunk count — chunk-count
equality does NOT prove dataset equality. Every M2 number from Apr 14
to Jul 15 ran on the variant, including the Apr 14-15 Mistral Metal
TQ4/TQ3 gold fill-in; canonical re-run in progress
(docs/metal-tq4-tq3-results-canonical.txt).

**Canonical-dataset gold (2026-07-15, M2 Metal, Qwen 3.5-9B, 145ch):**
F16/F16 7.2533 (Ryzen CPU gate: 7.2591, delta 0.006) ·
TQ4_0/F16 7.2965 (+0.60%) · TQ3_0/F16 7.3287 (+1.04%).

## Gold Standard: Mistral 7B (CPU Ryzen + Metal M2, already validated)

Both backends matched within ±0.02 PPL on the Phase 3.5 overnight run.

| Config | CPU PPL | Metal PPL | CUDA PPL expected | Pass if within |
|--------|--------:|----------:|------------------:|---------------:|
| F16/F16 | 5.1627 | 5.1678 | **5.16-5.18** | ±0.05 of 5.163 |
| TQ2_0/F16 | 6.4229 | 6.4120 | **6.40-6.44** | ±0.05 of 6.417 |
| V1 adaptive (1.5×) | 5.9940 | 6.0135 | **5.99-6.02** | ±0.05 of 6.004 |
| TQ2_1/F16 | 5.9784 | 5.9883 | **5.97-6.00** | ±0.05 of 5.983 |
| TQ4_0/F16 | ⏳ pending | ⏳ pending | **~5.21** ‡ | — |
| TQ3_0/F16 | ⏳ pending | ⏳ pending | **~5.32** ‡ | — |

‡ TQ4/TQ3 estimates come from 3-chunk ratios across all tested backends:
TQ4 shows ~+1% over F16 and TQ3 shows ~+3% over F16 consistently on
TinyShakespeare and WikiText-2 3-chunk. Tonight's M2 Metal run will
replace these estimates with actual gold-standard numbers.

**V1 adaptive layer distribution** (should be identical across backends):
- 11 × TQ2_0 (flat by 1.5× threshold)
- 19 × TQ2_1 (moderate outliers)
- 2 × TQ3_0 (heavy outliers)
- Total K-cache: 21.69 MiB at 2048 ctx

**Spectrum skew** on Mistral 7B head_dim=128:
- max/median ≈ 2.66×
- Label: **LOW**
- threshold auto-selected: 1.50

## Qwen3-8B (from M2 3-chunk WikiText-2, CUDA 160-chunk pending)

Previous M2 3-chunk on WikiText-2 gave TQ2_1 at +188% which was an
artifact — the fixed branch with Phase 3.5 should produce much saner
numbers. CUDA 160-chunk will establish the real baseline.

| Config | 3-chunk est | CUDA 160ch expected | Rationale |
|--------|------------:|--------------------:|-----------|
| F16 | 9.47 | **~9.5** | matches 3-chunk |
| TQ4_0 | 9.86 | **~9.5-9.6 (+0.5-1%)** | 3-chunk was +4%, 160ch usually lower |
| TQ3_0 | 10.04 | **~9.7-9.8 (+2-3%)** | 3-chunk was +6%, 160ch usually lower |
| TQ2_1 | 27.27 | **~10.5-11.5 (+10-20%)** | 3-chunk artifact; real delta expected ~15% |
| TQ2_0 | 38.29 | **~11-13 (+15-37%)** | 3-chunk artifact; Qwen3-8B has shown degraded output |
| V1 adaptive | — | **~10.5-11.0** | should match TQ2_1 |

**head_dim=128, Q/KV ratio=1.0** → V1 threshold = 1.50 (same as Mistral)

**Known concern**: Qwen3-8B was the model where TQ2_0 broke coherent
generation (echoed instructions back). TQ2_1's mixed precision rescued
it. If CUDA TQ2_0 PPL is dramatically worse than TQ2_1 (e.g., >20% gap),
that's consistent with the known sensitivity and confirms the fix.

## Gemma 3-4B (head_dim=256, Google dense)

Known to **improve with TQ3** in Phase 2b (PPL delta -0.102). Gemma 3
is the "TQ-loves-regularization" case.

| Config | CPU (Phase 2b, 145ch) | CUDA 160ch expected |
|--------|----------------------:|--------------------:|
| F16 | 12.536 | **~12.5** |
| TQ4_0/TQ4_0 | 12.416 (-0.12) | **~12.4 (-0.1)** |
| TQ3_0/TQ3_0 | 12.434 (-0.10) | **~12.4 (-0.1)** |
| TQ2_1 | 3-chunk only: 18.57 (+17%) | **~12.7-13.0 (+1-4%)** |
| TQ2_0 | 3-chunk only: 19.33 (+22%) | **~13.0-13.5 (+4-8%)** |
| V1 adaptive | — | **~12.5-12.9** |

**head_dim=256, Q/KV ratio=1.25** → V1 threshold = 2.00 (conservative)

**V1 adaptive should produce on Gemma 3-4B**:
- ~15 × TQ2_0 (flat layers)
- ~15-19 × TQ2_1 (moderate)
- 0 × TQ3_0 (no heavy tails with 2.0× threshold)
- K-cache: ~22-24 MiB (close to uniform TQ2_1)

**Why the 3-chunk Gemma numbers look scary**: Gemma 3's F16 baseline
is 12.5 on 3-chunk but only 12.536 on 145-chunk. Small sample → inflated
delta percentages. Real Gemma TQ behavior in Phase 2b showed **all TQ
configs improved or matched F16**. The CUDA 160-chunk run should
confirm this pattern.

## Cross-Backend Consistency Check

If the CUDA batch comes back dramatically different from these targets
(>±0.1 PPL for F16/TQ4/TQ3, >±0.5 for TQ2), something is wrong. Likely
causes:
1. Hadamard rotation not applied (check `k_cache_hadam = 1` in init log)
2. Different dataset used (must be WikiText-2 raw, not clean)
3. Build flags missing (GGML_CUDA=ON required)
4. Graph splits >2 (FA fallback to non-TQ path)

If the numbers match within tolerance, CUDA is validated and the
backend trio (CPU + Metal + CUDA) is complete.

## What "matches" means statistically

At 160 chunks, stderr is ~±0.035 for Mistral 7B. A delta of 0.035
between two runs is 1σ — within random noise. A delta of 0.10 is
nearly 3σ — significant. A delta of 0.20+ is pathological and points
to a real bug.

The CPU-vs-Metal comparison already established the backends agree
within 0.02 PPL. CUDA should land in the same envelope.
