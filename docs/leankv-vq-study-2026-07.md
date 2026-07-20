# Vector Quantization for KV Cache — Study + Adversarial Verification (2026-07-19)

**Question.** Scalar quantization is at its ceiling (41-scale experiment: 0.02 dB from
optimal; every bit-reallocation scheme measured dead). Vector quantization is the only
remaining approach that attacks the ceiling itself. Does PQ/RVQ beat scalar TQ at
matched budget?

**Verdict: NO-GO for VQ kernel work.** The study reported GO; independent adversarial
verification found the gate passes only under a protocol that systematically overstates
PQ's benefit by 3.6×. One narrow claim survives and is documented below as the only
live thread.

## What was run

`kit-v2/vq_study.py` — PQ (m=4/256cw, m=4/1024cw, m=2/64cw) and RVQ (2-stage), in both
Hadamard-rotated and raw space, on real Q·K attention from three architectures
(gemma4-E2B, LFM2.5-1.2B, gemma3-4b), calib/eval split, strict budget accounting
(code bits + 0.5 for fp16 block scales, matching the TQ ladder). Gate: gap_closure =
(KL_TQ2 − KL_cfg)/(KL_TQ2 − KL_TQ3) ≥ 0.6 at 2.5 bpe on ≥2 of 3 archs.

Study's claim: `pq_m4_k256` reaches +0.608 (E2B), +0.717 (gemma3), +0.300 (LFM2.5) → GO.

## Why the verification overturned it

**1. The report's own runtime control refutes its gate — the arithmetic was never done.**
The study included a closed-loop `llama-perplexity` control (real cache, error feeding
forward through layers). Computing gap_closure from those numbers:
(0.90845 − 0.78455)/(0.90845 − 0.16892) = **+0.168**, versus the claimed +0.608 — a
**3.6× overstatement** (10× on top-token agreement). PQ@2.5bpe sits just above TQ2,
nowhere near TQ3, once the loop is closed. Within-experiment, closed-loop stretches the
TQ2:TQ3 ratio from 3.39× to 5.38× — offline replay understates damage for *every* codec.

**2. Cross-corpus transfer costs 17–21% of the gap closure.** Fresh gemma3-4b K/Q dump
from disjoint text (0 shared 8-grams, verified): `pq_m4_k256/rot` +0.630 → **+0.521**,
`/raw` +0.760 → **+0.603**. At/below the gate on its best architecture. Notably "raw"
(which the report promoted) takes ~2× the cross-corpus penalty — the channel
correlations it exploits are corpus-specific.

**3. The SNR column is contaminated by fit-set leakage**, and the contamination tracks
sample density exactly:

| arch | samples/codeword | reported ΔSNR | held-out ΔSNR |
|---|---|---|---|
| gemma4-E2B | **1.43** | +5.92 dB | **+0.24 dB** |
| gemma3-4b | 5.70 | +4.12 dB | +2.27 dB |
| LFM2.5-1.2B | 11.75 | +2.16 dB | +0.74 dB |

E2B's headline "+5.92 dB reconstruction advantage" is **+0.24 dB on unseen keys**.
`pq_m4_k1024` on E2B reports 41.99 dB at 2.5 bits — physically impossible, pure
memorization (correctly DEGEN-flagged, still tabulated). Calibration used only
365–3008 subvectors per codebook position; the 400k cap was never approached.
**The architecture with honest sample density (LFM2.5, 11.75/cw) is the one that
already failed the gate (+0.300).** That inverse relationship is the tell.

**4. The scalar baseline was not the shipping codec.** The study billed its baseline
0.5 bpe of scale overhead but used one L2 scale per full head_dim vector (0.0625 bpe),
not in-tree TQ2_0/TQ3_0's per-32 amax. Effect: generous at 2 bits, pessimistic at 3.
Against the real shipping codec, E2B `pq_m4_k256` drops **+0.608 → +0.585** (below
gate), and the "beats TQ3 by 27–41%" claim becomes **+19.5–20.2%**.

**5. Budget reality kills the 2-bit case outright.** PQ@2.5bpe vs TQ2@2.5bpe saves
**zero bytes** — the codebook is pure overhead. Break-even context vs TQ3 is **4096
tokens on E2B** (2.25 MiB of codebooks), 1024 on gemma3, 512 on LFM2.5. At the 2048
context used in the closed-loop test, E2B's codebooks cost more than they save.

**What the study got right** (verified independently): budget accounting exact, no
hidden side information, codebook byte sizes correct, baselines internally consistent
(u2_full 0.1869 / u3_full 0.0551 reproduced exactly), seed sensitivity negligible
(spread 0.002–0.004 across 3 seeds), and all 30 gap-closure values arithmetically
exact. The failure was protocol, not sloppiness.

## The one surviving thread

**`pq_m2_k64` at 3.5 bpe beats shipping TQ3_0 at equal budget**, and it survives every
attack that killed the headline: cross-corpus **+34.1%** on gemma3 (0.0606 vs 0.0920),
gap_closure +1.149; adequate sample density (22.8–47/codeword on 2 of 3 archs); not
DEGEN; seed-stable. Corrected magnitude ~**20–34%** (not 27–41%) against the true
shipping codec.

Caveats: it lives at 3.5 bpe (TQ3's tier), so it is an **equal-budget quality win, not
compression**; and it has **never been tested closed-loop** — the one measurement that
demolished the headline config.

## Recommendation

**Do not start VQ2_0 kernel work.** Before VQ is re-decided, three cheap things
(in order): (1) closed-loop control for `pq_m2_k64`@3.5bpe — one perplexity run,
decides the only surviving claim; (2) fix sample density — dump 50–100k tokens across
≥3 corpora and re-measure (hours, and all current numbers are contaminated without it);
(3) make the baseline the actual in-tree TQ2_0/TQ3_0 including TQ3_0's iterative scale
refinement.

**Strategic note:** even a successful VQ tier has limited value for our targets. The
production ladder showed **TQ4 pure ships everywhere** (KLD 0.026–0.097) and that on
the 8 GB thesis model (LFM2.5-8B-A1B) KV is **6.75 MiB against 5.2 GB of weights** —
KV quantization is not the binding constraint. VQ's payoff would be concentrated in
long-context regimes (where codebooks also finally amortize), not in the edge-inference
case this program was built for.

## Methodology rules added

1. **Always compute gate metrics on the closed-loop control if one exists.** Offline
   attention replay understates error ~3–5× because it never lets error compound.
2. **Report held-out SNR**, never SNR over the fit set.
3. **Sample density (samples per codeword) must be reported** with any learned-codebook
   result; < 10 is a memorization warning.
4. **Baselines must be the shipping codec**, not a prototype approximation of it.
5. **Cross-corpus transfer is mandatory** for anything with learned parameters.

Artifacts: `kit-v2/vq_study.py`, `vq_study_report.md` (unmodified; read alongside this
document — its gate verdict is superseded here). Verifier scratch scripts in the
session scratchpad.
