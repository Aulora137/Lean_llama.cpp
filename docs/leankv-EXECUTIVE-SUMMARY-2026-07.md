# LeanKV — Executive Summary (KV-Cache Quantization Trial, July 2026)

*The one-page synthesis over the 13 detailed study docs in this directory. Everything
below is measured on-hardware (Ryzen 7 CPU + M2 + a rented 4090), closed-loop where it
matters, and adversarially verified.*

---

## The bottom line on 2-bit

**Lossless 2-bit KV is unachieved — by anyone — for a theoretical reason.** A 4-level
scalar quantizer tops out at ~9.3 dB fidelity and we proved we sit on that ceiling (a
41-scale grid search landed 0.02 dB from optimal). On top of it, softmax turns small
logit noise into flipped attention rankings, which compound over depth (low-margin
queries flip 60–70% under TQ2). That is rate-distortion + softmax sensitivity, not an
engineering gap. The best published adaptive method (KVTuner) calls itself "nearly
lossless" at **3.25–3.9 bit, not 2**.

- **Usable (lossy-but-acceptable) 2-bit** exists on *dense* models with outlier/residual
  tricks (KIVI, KVQuant — often with an fp16 recent-token window, i.e. not pure 2-bit),
  and is **near-free on extreme-dilution hybrids** (Qwen-3.5: only ~8/32 layers hold KV,
  so error can't accumulate — TQ2 +2.6%).
- On the **hard case (rank-bounded MQA, e.g. Gemma-4 E2B), 2-bit genuinely fails**, and
  we exhaustively confirmed nobody's toolkit rescues it.
- **The universal near-lossless floor is ~3 bit** — exactly where our numbers land.

The only structurally-open door to sub-3-bit is **vector quantization** (beats the scalar
ceiling by ≤1.5 dB) or **trained quantization** (BitNet-style QAT — but that trains the
model, and applies to weights, not the KV cache). Everything else is at a proven wall.

---

## Production recommendation (the decision gate)

**Ship TQ4/TQ4 pure on every architecture.** It is at/near the quality frontier
everywhere and compresses KV ~3.6× vs F16. TQ3 is the usable aggressive rung; TQ2 only
where the architecture gives it away.

| Model | Family | Ship (TQ4/TQ4) KLD / same-top | TQ3 usable? | KV @2k |
|---|---|---|---|---|
| Gemma-4 **E2B** | shared-KV-MQA (rank-bounded) | 0.097 / 89.0% | raw TQ3 0.272 (gate-promotes) | 10.1 MiB |
| Gemma-4 **E4B** | shared-KV-GQA | 0.048 / 91.4% | **gated TQ3 0.118** (partial promote) | 31.5 MiB |
| **LFM2.5-1.2B** | conv-hybrid | 0.026 / 91.7% | TQ3 0.060 | 6.75 MiB |
| **LFM2.5-8B-A1B** | conv-hybrid MoE | 0.091 / 85.9% | TQ3 0.155 | 6.75 MiB |
| **Gemma-3-4B** | dense-GQA | TQ4 (ref A1@3.0 = 0.39/77%) | +1.4% PPL — yes | — |
| **Qwen-3.5-9B** | SSM-hybrid | TQ4 | +0.95% PPL — yes; **TQ2 survives** | — |

*Strategic note:* on the 8 GB-RAM thesis model (LFM2.5-8B-A1B) the KV cache is **6.75 MiB
against 5.2 GB of weights** — KV quantization is no longer the binding constraint on
edge inference. Weight quantization is the next lever.

---

## What we tested and killed (the falsification ledger)

Every one measured, most adversarially re-verified; three initially *looked* like wins
(VQ, entropy, contour) and were caught by the verify pass.

| Idea | Verdict | Why |
|---|---|---|
| Magnitude **importance** allocation | ✗ redundant | corr 0.82 with variance; A3 lost 18% @30σ |
| **Entropy** allocation | ✗ no gain | converges to A1's plan / trails at matched budget |
| **Low-rank** projection | ✗ NO-GO | rank-224 recon alone = 5.5× TQ4's whole KLD (energy ≠ task info) |
| **Vector quantization** (PQ/RVQ) | ✗ NO-GO | offline oversold 3.6×; codebook breaks even only >4K ctx |
| **Contour/shading** (per-importance bits) | ✗ NO-GO | mix loses even to the *oracle*; heavy-hitter Jaccard 0.10–0.16 |
| **Trough-primacy** hypothesis | ✗ 0/55 layers | peaks dominate the softmax by orders of magnitude |
| **Per-channel K** (KIVI), post-RoPE | ✗ 0.249 > 0.187 | RoPE smears the channel-outlier structure it needs |
| **Per-channel K**, pre-RoPE (KVQuant premise) | ✗ 0.225 | mechanism real (+9.5% vs post) but still loses to baseline |
| **Ternary** {−1,0,+1} | ✗ | 3 levels < 4 at the same 2 physical bits |
| **Asymmetric** TQ4-K/TQ2-V | ✗ 0.482 | V is NOT free — cascades via the residual stream, not rankings |
| **TQ2 dilution** on conv-hybrid | ✗ rejected | LFM2's 6 attn layers carry all retrieval |

---

## What won (the survivors — all shipped or wired)

| Lever | Result | Status |
|---|---|---|
| **TQ4 pure + Hadamard + Lloyd-Max** | ships everywhere, ~lossless | shipped |
| **mse_opt block scale** | closes ~54–55% of the 2-bit gap, **free** | **shipped as the 2-bit default** (`LEANKV_TQ_SCALE=auto`) |
| **Reuse-weighted allocation (A1R)** | −42% (E2B) / −84% (E4B) vs A1 | allocator arm |
| **Q-dim rank gate** | auto-promotes unsafe TQ3/TQ2 → TQ4 on rank-bounded layers | shipped |
| **robust norm** (rank-normalize + sink-exclude) | correct low-bit allocation | shipped |
| Sparse-outlier side-channel | ~65% gap closed (a menu item, not an E2B rescue) | measured, not built |

---

## Novel contributions

1. **The adaptive geometry→method prober** (`kit-v2/kv_policy.py`): reads a GGUF header,
   classifies the architecture family, and emits the measured-correct KV policy — the
   "prober picks the method" idea the field names as open but nobody has shipped for
   llama.cpp. Validated to reproduce every measured config's *discriminating* fields
   (family, KV-owning count, rank-bounded pattern, TQ2 refusal), not just the ship string.
   Wired to launch via `scripts/leankv-serve.sh`; it has caught 3 stale-doc errors by
   reading models instead of notes.
2. **`mse_opt` as a shipped default** — the one no-cost, no-format-change quality win.
3. **Broadest architecture coverage in the literature** — the only systematic KV-quant
   study spanning shared-KV rank-bounded MQA (Gemma-4) and conv/SSM-hybrids (LFM2.5,
   Qwen-3.5); everyone else tests dense Llama/Qwen/Mistral.
4. **Closed-loop-honest methodology** (below) — the discipline that repeatedly overturned
   plausible offline wins.

---

## Methodology rules earned the hard way

- **Closed-loop is the only truth.** Offline attention replay understates damage 3–5×
  because it never lets error compound (the VQ "win" was +0.61 offline, +0.17 closed-loop;
  mse_opt was 47% offline, 7% closed-loop on E2B). Nothing ships on offline numbers.
- **Adversarially verify.** A second agent re-deriving from scratch caught three
  overstated GO verdicts and two implementation bugs this trial.
- **Baseline against the *shipping* codec**, matched bit-budget, audited emitted bit-sums.
- **Cross-corpus transfer** is mandatory for anything with learned parameters.
- **Report held-out SNR + samples-per-codeword;** <10 samples/codeword is a memorization
  flag.

---

## State of the field (surveyed July 2026)

KIVI (per-channel K / per-token V), KVQuant (outlier residuals, pre-RoPE), SQuat
(query-subspace-orthogonal), KVTuner (offline sensitivity-searched per-layer precision),
RotateKV (outlier-aware rotation) — all real, all converging on the same ~3-bit
near-lossless frontier, none testing the shared-KV-MQA or conv/SSM-hybrid families we
mapped, and none shipping a general geometry→method prober. Our results reproduce SQuat's
published numbers to within the band and independently re-derive its mechanism.

---

## Deliverables (in-tree, both remotes)

- Engine: `mse_opt` bit-width-gated default; `LEANKV_KV_PLAN` per-layer plans; Q-dim gate;
  robust-norm; reuse-weighted allocator; pre-RoPE / Q / K calibration-capture instruments.
- Tools: `kit-v2/kv_policy.py` (prober), `kv_bit_allocator.py` (A1R allocator),
  `scale_study.py` / `perchannel_study.py` / `prerope_study.py` / `noiseshape_study.py` /
  `vq_study.py` / `contour_study.py` (the falsification harnesses).
- Launch: `scripts/leankv-serve.sh` — probe → derive KV flags → exec llama-server.
- 13 dated study docs (see this directory) — the full evidence trail.

---

## Open questions (unbuilt, gated)

1. **Vector quantization revisited** — lattice VQ (no stored codebook) is the one
   sub-3-bit path not blocked by a theorem; PQ died on codebook overhead, not quality.
2. **Weight quantization ladder** — KV is now 0.13% of the 8 GB model's footprint; the
   next big memory lever is weights (Q4→Q3→Q2/IQ2), untested on our targets.
3. **In-engine auto-config-at-load** — the prober is wired at launch; classify-in-C is
   the follow-on.
4. **Long-context validation** — all quality numbers are at 2K ctx; the global-layer /
   window-cap behavior at 32K+ is the regime the compression is ultimately for.

---

*Verdict for the trial: KV quantization for edge inference is solved at TQ4 (ships) and
TQ3 (aggressive); 2-bit is a lossy-with-caveats regime blocked below ~3 bit by physics;
and the durable asset is not a single quantizer but the measured **menu + prober** that
routes each architecture to its answer.*
