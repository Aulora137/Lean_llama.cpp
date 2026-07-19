# Contour/Shading Quantization — Assessment + Measured Study (2026-07-19)

**Origin:** Henry's brainstorm (`docs/Brainstorm: Imagine 3 bit is a 3 dimensi.md`) —
spend bits by importance ("shading": peaks and troughs dark, background light),
nested Q4→Q3→Q2 tables, direct F16→2-bit fitting, and a remembered experiment where
"the trough mattered more than the peak." This doc records the code-level fact-check,
the measured study, and the verdicts. Method: same falsification discipline as the
importance/entropy/low-rank threads.

## Fact-check of the premises (against our own code)

1. **"Direct F16→2-bit would avoid cascade error" — already implemented, still
   collapses.** TQ tiers are independent Lloyd-Max fits (`ggml-tq.c:45-98`), and
   `LEANKV_CALIBRATION_AUTO` (default ON) fits codebooks directly on real rotated K
   (`leankv-calib.cpp:201`). There is no cascade. TQ2's collapse is the scalar
   ceiling (41-scale experiment: 0.02 dB from optimal).
2. **What actually collapses at 2-bit:** not single-layer fidelity — LeanKV §5.2
   showed 2-bit attention cosine 0.99999 on one layer. It is **ranking flips
   compounding over depth**: 4-level noise (~9.3 dB/dim ceiling) crosses logit-gap
   scale, softmax amplifies, each layer feeds the next. Hence dense collapse
   (+25–117%), Qwen-3.5 survival (8 noise-injecting layers), LFM2.5 in between.
3. **Codebooks are symmetric by construction** (`leankv_lloyd_max_symmetric`);
   outlier selection is variance-based, sign-blind. Troughs are treated exactly like
   peaks today — never zeroed, never privileged.
4. **The remembered trough experiment is not recorded anywhere on this machine**
   (searched both repos, git history incl. deleted files, the 1,728-config sweep
   knobs, shell history, old session transcripts). Re-derived properly below.
5. The brainstorm's "what others tried" section is largely confabulated AI output
   (no Google TurboQuant paper — TQ is our name; no AmesianX/AtomicChat TQ forks;
   E2B is not a "30B-knowledge MoE"; no ik_llama bitmap KV). Real prior art for
   these ideas: H2O/Scissorhands/SnapKV (token importance), KIVI/KVQuant
   (asymmetric K/V), SpinQuant (learned rotations), Any-Precision LLM (nested
   tables), StreamingLLM (sinks).
6. **The nested-table/fade idea is Henry's own parked Phase 4**: `TIERED_KV_CACHE.md`
   design is complete and `requantize_tq4_to_tq3/tq3_to_tq2/tq4_to_tq2` primitives
   are implemented and unit-tested (April). What was missing was a placement policy —
   which the study below tested.

## The study (kit-v2/contour_study.py — real Q·K attention, held-out discipline)

New instrumentation: `LEANKV_CALIBRATION_DUMP_Q_PATH` captures post-RoPE **Q** per
KV-owning layer (companion to the existing K dump). K+Q pairs collected for three
architectures (gemma-4-E2B shared-KV, LFM2.5-1.2B conv-hybrid, gemma-3-4b dense
MQA reference); causal logits built from real Q·K (SWA windows respected, per-arch
attention scales verified); quantization via the LeanKV prototype TurboQuantizer
(Hadamard + block scale + Lloyd-Max — verified bit-identical to a direct prototype
call). Calib = first half of positions, eval = second half; significance statistics
never see eval queries (oracle variants labeled separately). Independently
re-implemented from scratch by an adversarial verifier before the report existed —
all numbers below reproduced by both pipelines.

### Q1 — cross-token mixed precision (the shading bet): **NO-GO**

Top-X% keys at 4-bit + rest at 2-bit vs uniform tiers, mean KL of eval softmax rows:

| model | uniform 2-bit | **uniform 3-bit** | best held-out mix | its bpe | oracle30 |
|---|---|---|---|---|---|
| gemma4-E2B | 0.0948 | **0.0274** | 0.0346 (ho50) | 3.50 | 0.0447 |
| LFM2.5-1.2B | 0.1523 | **0.0415** | 0.0552 (ho30s) | 3.11 | 0.0449 |
| gemma3-4b | 0.2065 | 0.0458 | **0.0325 (ho50)** | 3.50 | 0.0343 |

- No held-out plan beats uniform 3-bit at *lower* budget on any arch; the gate
  (≥2 archs) fails outright.
- The one bright cell — gemma3-4b ho50 beating uniform-3 by 29% — is **same-budget**
  4/2 mixing (3.50 vs 3.50 bpe), not savings, and it inverts on the other two archs.
  A dense-arch-only, same-budget curiosity, not a mechanism.
- Even the **oracle** (perfect future knowledge) loses to uniform 3-bit on E2B and
  LFM2.5. The idea fails even with cheating.

### Q2 — heavy-hitter stability (the assumption underneath): **DEAD**

Jaccard overlap of calib-derived top-16 key sets vs eval queries' actual top-16:
**0.122 (E2B), 0.164 (LFM2.5), 0.097 (gemma3)**. Past attention predicts only
~10–16% of future heavy hitters. Importance churns with query position — this is
*why* Q1 fails, and it aligns with the known weakness of H2O-style eviction.
(Exception: position 0 is a true universal sink — mass 0.07–0.43 vs median ~0.004
on every E2B layer. "First-4 sinks" overstates; it's really first-1-to-2.)

### Q3 — peak vs trough (the lost experiment, re-derived): **DOES NOT REPLICATE**

Clamping the top-p% most-positive logits vs bottom-p% most-negative, per layer,
p ∈ {1,2,5,10}: **0 of 55 layers** across three architectures show trough dominance
(ratio > 1.5); peak clamping dominates by orders of magnitude everywhere. At the
attention-logit level, peaks are what matter. If the original experiment was real,
it measured something else (value-level? a specific arch? different metric) — as
formulated here, the hypothesis is rejected.

## Verdict

- **Do not pursue contour/shading allocation for KV quantization.** Three measured
  negatives (mix loses even to oracle on 2/3 archs; importance unpredictable;
  trough primacy absent), consistent across two independent implementations.
- The idea scoreboard after five falsification campaigns: importance ✗, entropy ✗,
  low-rank ✗, TQ2-dilution ✗, cross-token shading ✗, trough-primacy ✗.
  Survivors: **TQ4 pure** (ships everywhere), **raw TQ3** (memory-desperate rung),
  **reuse-weighting** (shared-KV archs only), **Q-dim gate** (auto-protection).
- **The only credible route below 3 bits remains vector quantization** — the one
  direction that attacks the scalar ceiling itself rather than re-allocating around
  it. Age-based tiering (Phase 4 design + existing requantize primitives) remains
  sound *engineering* for long contexts — as memory management, not as a quality
  play.

Artifacts: `kit-v2/contour_study.py` (torch; run with `~/LeanKV/.venv/bin/python3`),
`contour_study_report.md` (full per-layer tables), Q-capture in `leankv-calib`
(`LEANKV_CALIBRATION_DUMP_Q_PATH`). Verifier's independent recompute lives in the
session scratchpad (methodology cross-checked: budgets exact, no leakage, quantizer
bit-identical, causality asserted).
