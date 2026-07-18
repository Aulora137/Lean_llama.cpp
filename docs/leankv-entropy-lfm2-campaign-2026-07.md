# LeanKV Campaign 2 — Entropy Arms + LFM2.5 Matrix + Budget Controls (2026-07-18)

**Questions.** (1) Does attention-entropy–guided allocation (arms A2/A4, newly unblocked
by the kvimp entropy emitter) beat the shipping A1 baseline? (2) How does a conv+attention
hybrid (LiquidAI LFM2.5) behave under the full signal matrix?

**Answers.** (1) **No — at matched effective budget, A1 (variance+outlier, robust norm)
wins or ties everywhere tested.** Entropy's apparent wins were an effective-budget
artifact of the GQA max-reduction in plan emission; matched-budget controls reversed
(Gemma) or equalized (LFM2, plan-identical convergence) them. Importance remains harmful
everywhere. (2) LFM2.5 is extremely TQ-friendly (6 growing KV tensors total, no rank
slack, sink on the *last* attention layer), and the signal ranking replicates.

**The deeper finding:** on coarse type ladders (tq2/3/4), *small effective-budget
increments move KLD more than signal choice does*. Plan emission on GQA models inflates
effective bits nondeterministically per arm (max-reduction over kv-heads), so any
cross-arm comparison MUST audit emitted plan bit-sums. This is now a standing
methodology rule.

## Setup

| | |
|---|---|
| Engine | Lean_llama.cpp `c42ff538` (entropy emitter `10e84634` + LFM2 port merge) |
| Models | `gemma-3-4b-it-Q4_K_M` · `LFM2.5-1.2B-Instruct-Q4_K_M` (sha-verified downloads) |
| Metric | Mean KLD to same-model F16 base, WikiText-2 canonical (`7c0137fc…`), c=2048 |
| Calibration | kvimp + entropy, 7,608 tok of `wiki.valid.raw`, **`-fa off`** (fork defaults FA on; softmax never materializes under FA) |
| Plans | `--bmax 4 --norm robust`, target 3.0 bpw except controls |

## Gemma 3-4B (34 layers, GQA 8/4, all own KV)

| Arm | Nominal target | Emitted bit-sum (68 cells) | Mean KLD ↓ | Same-top ↑ |
|---|---|---|---|---|
| A1 | 3.0 | 213 | 0.3922 | 77.17% |
| A2 entropy | 3.0 | 218 | 0.3723 | 77.73% |
| **A1 control** | **3.1** | **219** | **0.3577** | **78.15%** |
| A4 imp+ent | 3.0 | 218 | 0.4537 | 74.94% |
| A3 importance | 3.0 | 216 | 0.4631 | 74.46% |

At ~matched budget (218 vs 219): **A1 > A2 by 3.9% (~4.8σ)** — entropy's apparent
+5.1% win over the 213-bit A1 was budget, not signal. A4 vs A1-control at equal-ish
budget: importance drags entropy 27% underwater. Signal ranking at matched bits:
**variance ≥ entropy ≫ importance**.

## LFM2.5-1.2B (16 layers, only 6 attention @ {2,5,8,10,12,14}, GQA 32/8, head_dim 64)

F16 base PPL 17.80 ± 0.14 (healthy). KV at F16: 24 MiB total — the whole model has
six growing KV tensors; conv layers hold a fixed 2×2048 f32 state (16 KB) each.

| Arm | Emitted bit-sum (12 cells) | Mean KLD ↓ | Same-top ↑ | KV (c=2048) |
|---|---|---|---|---|
| A1 | 39 | 0.2369 | 77.80% | 5.62 MiB |
| A2 entropy | 41 | 0.1242 | 82.87% | 5.88 MiB |
| A1 control @3.1 | 41 | 0.1242 | 82.87% | 5.88 MiB |
| A3 = A4 importance | 41 | 0.2313 | 78.11% | 5.88 MiB |

Two clean facts:
- **A1@3.1 emitted a plan bit-identical to A2's** — with 12 cells, variance and entropy
  converge to the same allocation; entropy contributed nothing unique.
- **At identical 41-bit budget, importance loses 46%** (0.2313 vs 0.1242) — it chose a
  different plan and wasted the budget. Third architecture where importance fails.
- A3 ≡ A4 exactly (identical plans): in a 12-cell space importance's ordering fully
  determines the greedy outcome; the entropy multiplier flipped nothing.

## LFM2.5 architecture profile (compiler feature vector)

From the Step-0 probe + rank dump (732 vecs/layer, post-RoPE K, SVD):

- 6/16 attention layers; conv layers = fixed 16 KB state each, **zero KV growth**.
- q_dim = 2048/32 = 64 = head_dim → **not rank-bounded**; measured r99 = 59–61/64
  (~94% fill), r95 ≈ 52/64. **No rank slack — low-rank lever inapplicable** (opposite
  of Gemma-4 E2B's 38% global fill).
- **Variance sink = layer 14, the LAST attention layer** (Gemma sinks at layer 0) —
  sink position is architecture-dependent; robust-norm detection handles both.
- No cross-layer sharing → reuse lever inert (A1R ≡ A1).
- At 3 bpw KV ≈ 5.9 MiB @ 2k ctx. TQ4/TQ3 on 6 tensors + fixed conv state is why a
  1.2B LFM2.5 KV footprint is nearly context-invariant in practice.

## Program-level conclusions (five architectures in)

1. **Allocation signal:** variance + outlier + robust norm (A1). Add **reuse** where the
   arch shares KV (E2B: −42%). Nothing else has beaten A1 at matched budget.
2. **Entropy and importance collectors are diagnostic-only.** Entropy is harmless-to-
   convergent; importance is consistently harmful (Gemma-3, Gemma-4-E2B allocation
   analysis, LFM2.5).
3. **Budget granularity beats signal choice** on coarse ladders — the adaptive compiler
   should sweep nominal targets (3.0/3.1/3.2) and audit emitted bit-sums rather than
   trust one target per arm.
4. Methodology rule: **report emitted plan bit-sums with every arm table.**

Raw logs: `kld_{A1,A2,A3,A4}_{gemma,lfm2}.log`, `kld_A1ctl_*.log`,
`lfm2_rank_report.txt`, status `kld_campaign2_status.txt`, `budget_controls_status.txt`
(local, gitignored).

---

# Campaign 3 — LFM2.5-8B-A1B full matrix (overnight 2026-07-18, 01:04–03:22)

The 8 GB-RAM SLM thesis model: `lfm2moe`, 24 layers, **6 attention @ {2,6,10,14,18,21}**,
18 conv (fixed state), 32 experts / 4 active (~1.5B active of 8.3B), GQA 32/8,
head_dim 64, vocab 128 000. Same KV geometry as the 1.2B — six growing tensors,
24 MiB @ F16 c=2048. F16 base PPL 30.58 ± 0.30 (143 chunks; it-tuned MoE on raw
WikiText — internal comparisons only).

**All four arms natively emitted 38-bit plans — identical budgets, zero confound**
(the 12-cell space + robust norm made the control redundant: A1ctl@3.02 reproduced
A1 exactly).

| Arm | Emitted bits | Mean KLD ↓ | Same-top | KV (c=2048) |
|---|---|---|---|---|
| A1 ≡ A2 ≡ A1ctl (identical plans) | 38 | **0.6109 ± 0.0023** | 64.99% | 5.50 MiB |
| A3 ≡ A4 (identical plans) | 38 | 0.6260 ± 0.0023 | 64.60% | 5.50 MiB |

- **Entropy converged to variance's exact plan a third time** (A2 ≡ A1, bit-identical
  results). The convergence is now the norm, not the exception.
- **Importance lost on a fourth architecture** at perfectly matched budget (+2.5%,
  ~4.7σ): its plan dropped layer-6 V to tq2 to over-protect layer-14 V — the same
  quiet-but-wrong reallocation pattern as everywhere else.
- **Sink pattern replicates:** variance sink = layer 21, the LAST attention layer —
  matching the 1.2B (layer 14, also last-attn). *LFM2-family models sink at the final
  attention layer*, not layer 0; robust norm caught it automatically.
- Rank: median r99 57/64 (89% fill), r95 38–50/64; only the sink layer (21) mildly
  rank-deficient. No usable low-rank slack — same verdict as the 1.2B.
- Sensitivity scales up: at the same 3 bpw the 8B carries KLD 0.61 vs the 1.2B's 0.12.
  The MoE's per-token expert routing appears to amplify KV perturbation — worth a
  budget-ladder sweep (3.5/4.0 bpw) before production use.

## The 8 GB thesis, quantified

Weights Q4_K_M 5.2 GB + KV 5.5 MiB @ 2k ctx (TQ, vs 24 MiB F16) + fixed 18×16 KB conv
state. **Total inference footprint ≈ 5.5 GB — an 8.3B-knowledge model with 1.5B-active
compute fits an 8 GB machine with ~2.5 GB to spare**, and KV stays negligible at any
context length (only 6 tensors grow; TQ4 keeps them ~4× smaller). KV quantization is
NOT the binding constraint on this family — weights are. The adaptive-compiler verdict
for LFM2.5: plain A1+robust on the 6 attention tensors, done; spend engineering
elsewhere (weight quant, expert offload policy).

## Final signal scoreboard (five architectures, all matched-budget)

| Signal | Verdict |
|---|---|
| variance + outlier + robust norm (A1) | **the allocation baseline — never beaten at matched budget on single-owner arches** |
| reuse (A1R) | **only signal to beat A1** — −42% KLD on E2B; requires shared-KV arch |
| entropy (A2) | converges to A1's plan (LFM2.5 ×2) or trails it (Gemma, −3.9% at matched bits); diagnostic-only |
| importance (A3/A4) | harmful on all four arches tested (Gemma-3, E2B-by-analysis, LFM2.5-1.2B, 8B-A1B) |
