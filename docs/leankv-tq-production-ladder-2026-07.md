# TQ Production-Readiness Ladder — E2B / E4B / LFM2.5 + Low-Rank Phase A (2026-07-19)

**Question (Henry):** likelihood of successfully applying TQ4/TQ3 to E2B (then E4B)?
Does the method adapt across architectures? What about LFM2.5?

**Answer, measured:** **TQ4 pure is production-ready on all four models** (KLD
0.026–0.097, same-top 86–92%) — better than the Gemma-3-4B config already serving in
production (0.392/77%). The earlier pessimistic E2B numbers were an artifact of
*mixed low-bpw plans*, not the model. Low-rank projection is a measured **NO-GO**.
TQ2 stays dead everywhere outside the Qwen-hybrid exception.

Engine: `ee32170a` (mid-ladder note: the first E2B run used the pre-low-rank binary;
the no-op guarantee was bit-verified, results comparable). Metric: Mean KLD to
same-model F16 base, canonical WikiText-2 (`7c0137fc…`), c=2048, K+V as stated.

## Decision gates (production recommendations)

| Model | Ship config | Mean KLD ↓ | Same-top | KV @2k ctx | vs F16 KV |
|---|---|---|---|---|---|
| Gemma-4 **E2B** | **TQ4/TQ4 pure** | 0.0965 | 88.98% | 10.1 MiB | 3.6× smaller |
| Gemma-4 **E4B** | **TQ4/TQ4 pure** | 0.0477 | 91.35% | 31.5 MiB | ~3.6× |
| **LFM2.5-1.2B** | **TQ4/TQ4 pure** | 0.0258 | 91.68% | 6.75 MiB | 3.6× |
| **LFM2.5-8B-A1B** | **TQ4/TQ4 pure** | 0.0911 | 85.85% | 6.75 MiB | 3.6× |
| — aggressive option (E2B only) | raw TQ3/TQ3 (`LEANKV_NO_QDIM_GATE=1`) | 0.2722 | 81.31% | 7.9 MiB | 4.6× |
| — mid rung (E4B) | A1R@3.5 (robust, reuse) | 0.1293 | 85.85% | 28.25 MiB | — |

Reference: Gemma-3-4B A1@3.0 (production today) = 0.3922 / 77.17%. Every ship pick
above clears it by a wide margin.

## E2B full ladder

| Config | Mean KLD | Same-top | KV |
|---|---|---|---|
| TQ4/TQ4 pure | **0.0965** | 88.98% | 10.12 MiB |
| TQ3 gated | ≡ TQ4 (0.0965) | ≡ | 10.12 MiB (all promoted) |
| TQ3 raw | 0.2722 | 81.31% | 7.88 MiB |
| TQ2 gated | ≡ TQ4 (0.0965) | ≡ | 10.12 MiB (all promoted) |
| TQ2 raw | 1.1664 | 59.97% | 5.62 MiB |
| A1R@3.0 (campaign 1) | 0.9934 | 65.07% | 7.88 MiB |
| A1R@3.5 | 0.9084 | 67.73% | 9.00 MiB |
| A1R@4.0 | 0.2298 | 82.93% | 10.12 MiB |

**Q-dim gate validated in production form:** uniform TQ3/TQ2 requests on this fully
rank-bounded model were silently promoted to TQ4 — three runs produced results
identical to TQ4-pure to six digits. Request an unsafe tier, get a safe one.

**Mixed plans lose to uniform tiers on rank-bounded models:** raw TQ3 (0.27 @ 7.9 MiB)
dominates A1R@3.5 (0.91 @ 9.0 MiB). The allocator's tq2 floor cells are what murdered
the campaign-1 numbers — the model was never the problem; the 3.0-bpw target was.
Rule: on rank-bounded arches, use uniform TQ4 (or raw TQ3 if memory-desperate);
reach for mixed plans only at ≥4.0 bpw targets (A1R@4.0 ties TQ4's size at slightly
worse quality — no reason to prefer it on E2B).

## E4B (first measurement of this model)

Geometry: 42 layers, **24 own / 18 shared**, n_embd 2560, head_dim 256 local / 512
global (globals at il 5,11,17,23) → **q_dim 320 > 256: locals NOT rank-bounded**
(the wider model escapes E2B's trap; only the 4 globals remain bounded).
Sinks: variance [0, 1], k-importance [0, 22] — third distinct sink layout, all
auto-detected. Rank fill: locals r95 60.2%, globals 52.1%.

| Config | Mean KLD | Same-top | KV |
|---|---|---|---|
| TQ4/TQ4 pure | **0.0477** | 91.35% | 31.50 MiB |
| A1R@3.5 | 0.1293 | 85.85% | 28.25 MiB |
| A1R@3.0 | 0.2749 | 79.53% | 24.88 MiB |
| A1@3.0 | 1.7184 | 55.13% | 25.12 MiB |

**Reuse confirmed a second time, bigger:** A1R@3.0 beats A1@3.0 by **−84%** (E2B was
−42%). Two shared-KV models, one lever, decisive both times. F16 base PPL 55.19.

## LFM2.5 rungs

| Config | Mean KLD | Same-top | KV |
|---|---|---|---|
| 1.2B TQ4/TQ4 | **0.0258** | 91.68% | 6.75 MiB |
| 1.2B TQ2/TQ2 | 0.5352 | 65.52% | 3.75 MiB |
| 8B-A1B TQ4/TQ4 | **0.0911** | 85.85% | 6.75 MiB |
| 8B-A1B A1@4.0 | 0.1024 | 85.04% | 6.75 MiB |
| 8B-A1B A1@3.5 | 0.2282 | 77.64% | 6.12 MiB |
| 8B-A1B TQ2/TQ2 | 0.8366 | 59.39% | 3.75 MiB |

**TQ2 hybrid-dilution hypothesis: rejected.** Six attention layers do not buy
Qwen-style TQ2 survival — those six carry all the retrieval. TQ2 remains unusable
outside the Qwen-3.5 exception.

## Low-Rank Phase A — measured NO-GO, Phase B cancelled

Rank-224 reconstruction (95.07% calibration energy) on E2B's 3 global owners:

| Config | Mean KLD | Same-top |
|---|---|---|
| LR224 + F16 KV (isolated rank effect) | 0.5346 | 74.24% |
| LR224 + TQ4 pure | 1.4685 | 56.74% |
| LR224 + A1R@3.5 | 1.2027 | 61.50% |

The projection **alone** costs 5.5× more KLD than full TQ4 quantization (0.53 vs
0.097), and it composes super-additively with quantization (0.53 ⊕ 0.097 → 1.47).
**Calibration-energy retention ≠ task-information retention** — the discarded 4.9%
(the r99 tail at 334–347) was signal, not noise. The Phase-A reconstruction-first
design did exactly its job: falsified the storage idea for ~90 minutes of compute
before any attention-path engineering. The implementation stays in-tree
(`LEANKV_KV_LOWRANK`, verified correct) as a research instrument; Phase B storage
is cancelled at this rank. Any revival needs r99-grade ranks (~410+ on E4B globals),
where the memory win (~20%) can't compete with TQ4's 3.6× — archived.

## Program rules confirmed / added

1. **TQ4 pure first.** On every new architecture, measure uniform TQ4 before any
   allocator run — it has been at or near the quality frontier on all six models.
2. Mixed plans are for ≥3.5–4.0 bpw targets or non-rank-bounded arches; sub-3.5
   mixes on rank-bounded models self-sabotage via tq2 floors.
3. The Q-dim gate's auto-promotion is measured-correct — leave it on in production.
4. Reuse-weighting whenever the arch shares KV (2/2 wins, −42% and −84%).
5. Energy spectra guide *diagnosis*, not *truncation* — low-rank storage is dead
   until something changes the energy≠information picture.

Raw logs: `kld_ladder_status.txt`, `kld_e4b_status.txt`, `kld_lowrank_status.txt`
and per-run `kld_*.log` (local, gitignored). Companion docs:
`leankv-e2b-campaign-2026-07.md`, `leankv-entropy-lfm2-campaign-2026-07.md`,
`leankv-adaptive-kv-compiler.md`.
