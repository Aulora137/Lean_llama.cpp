# The Adaptive KV-Quant Menu + Prober — geometry → method (2026-07-21)

**What this is.** `kit-v2/kv_policy.py` is the *prober* the adaptive-KV-compiler
design note (`leankv-adaptive-kv-compiler.md`) called for: a geometry → method
decision function. It reads a GGUF header (no model execution), classifies the
architecture family, and emits a KV-quant plan grounded in the **measured** LeanKV
menu — the production ladder, the E2B campaign, the scale-scheme study, the
per-channel study, the importance/entropy ablations. Every branch cites the doc it
comes from. On architectures we never measured, it says *UNVALIDATED* instead of
faking a number.

It replaces "re-derive the policy from six docs every time" with one runnable
function, and it is the front half of the compiler pipeline
(`PROBE → FEATURES → POLICY → LEANKV_KV_PLAN`). The `kv_bit_allocator.py`
reuse-weighted budget allocator becomes one subroutine it recommends, not the entry
point.

---

## The menu — one row per architecture family

Geometry signature is what the prober reads from the GGUF; method is the measured
policy. **Ship config is TQ4/TQ4 pure on every family** (ladder rule #1 — TQ4 pure
has been at/near the quality frontier on all six measured models). The families
differ in their *aggressive* option and their *flags*.

| Family | Geometry signature (probed) | Ship | Aggressive option | Why (doc) |
|---|---|---|---|---|
| **dense-MHA / dense-GQA** | all layers own KV; `q_dim ≥ head_dim`; no sharing, no bypass | TQ4/TQ4 pure | TQ3/TQ3 usable; **TQ2 dead** (scalar Lloyd-Max wall) | `kv-importance-ablation`, `scale-scheme`: Hadamard + scalar TQ ladder; importance redundant (corr 0.82 w/ variance) |
| **shared-KV-MQA** (E2B) | all layers have K tensors **but** `shared_kv_layers>0`; `q_dim < head_dim` on **every** owner (fully rank-bounded); MQA (`n_head_kv=1`) | TQ4/TQ4 pure | **raw TQ3** (needs `LEANKV_NO_QDIM_GATE=1`; else gate promotes) *or* asym **TQ4-K/TQ3-V** (0.161). **sub-3bpw NO-GO.** Budget plan: **A1R reuse-weighted** | `production-ladder`, `perchannel-e2b` (2-bit exhaustively NO-GO), `e2b-campaign` (A1R −42%) |
| **shared-KV-GQA** (E4B) | `shared_kv_layers>0`; wider — `q_dim > local head_dim` (locals free), `q_dim < global head_dim` (**only globals** rank-bounded); GQA | TQ4/TQ4 pure | **TQ3 gated** — gate *partially* promotes only the rank-bounded globals (K adaptive / V tq3), 0.118 @ 26.5 MiB; *or* A1R@3.5 (0.129) | `production-ladder` (first non-all-or-nothing gate result) |
| **conv-hybrid** (LFM2.5 / LFM2-MoE) | per-layer `head_count_kv` array with 0s (short-conv layers); only a few attn layers own KV; high `kv_bypass`; `q_dim = head_dim` (no rank slack) | TQ4/TQ4 pure on the attn tensors | TQ3/TQ3 usable (0.060 / 0.155). **TQ2 REJECTED** — the 6 attn layers carry all retrieval | `entropy-lfm2-campaign` (TQ2 dilution rejected), `adaptive-kv-compiler` |
| **SSM-hybrid** (Qwen3.5 / mamba) | `arch.ssm.*` keys + mamba/deltanet layers that bypass KV; only a few attn layers own KV; very high `kv_bypass` | TQ4/TQ4 pure on the attn tensors | TQ3/TQ3; **TQ2 = the one documented 2-bit survivor** (+2.6%, extreme dilution) — measure-before-ship (see caveat) | `adaptive-kv-compiler`, `production-ladder` (TQ2 dead everywhere *except* the Qwen-hybrid exception) |
| **UNKNOWN** | arch not in the measured set | TQ4/TQ4 pure (safe default) | **none — measure first** (`UNVALIDATED` flag) | ladder rule #1; honesty: no faked confidence |
| **no-kv-cache** | encoder (bert etc.), no autoregressive cache | n/a | n/a | nothing to quantize |

**Two rules are architecture-independent and always emitted** (`scale-scheme` +
`kv-importance-ablation`):
- **scale = `mse_opt` @ 2-bit / `amax` @ ≥3-bit** — free ~½-gap of 2-bit quality on
  every non-rank-bounded model; neutral at ≥3-bit.
- **`--norm robust`** (rank-normalize + exclude the attention-sink layer) and the
  **Q-dim gate ON** (per-layer; auto-promotes an unsafe TQ3/TQ2 request to TQ4 on
  rank-bounded layers — measured-correct 3×).

---

## How the prober works (probe → features → policy)

### 1. PROBE — one GGUF header read, no inference
Reuses `kit-v2/gguf_extract.py`'s minimal reader (`_R`). Reads metadata KVs +
tensor-info names only (bounded 64 MiB; the multi-GB weight body is never touched).
Extracts per model:

- `arch`, `n_layer`, `n_embd`, `n_head`, `n_head_kv` (scalar **or** per-layer array
  — `0` marks a conv/SSM layer on lfm2 / lfm2moe).
- **KV-owning layers** via `blk.<il>.attn_k` tensor presence, then — for the shared-KV
  Gemma family — filtered by the engine's ownership rule `il < n_layer - shared_kv_layers`
  (`has_kv(il)`, src/llama-hparams.cpp). On gemma4 *every* layer has `attn_k` but only the
  first `n_layer - shared_kv_layers` allocate a cache (the rest reuse an earlier owner's,
  `k_l/v_l = nullptr`), so E2B = **15/35**, E4B = **24/42** owners — matching the kvimp
  collector and `kv_bit_allocator.py`. LFM2/Qwen omit `attn_k` on conv/SSM layers, so the
  tensor signal is already correct there.
- **Per-layer head_dim** — gemma4 splits `key_length` 512 (global) vs
  `key_length_swa` 256 (local); global layers are where `sliding_window_pattern[il]`
  is `False`. Other arches: `key_length` if present, else `n_embd/n_head`.
- `shared_kv_layers`, `rope.freq_base(_swa)`, `ssm.*` / `shortconv` presence.

### 2. FEATURES — derived geometry
`q_dim = n_embd/n_head`; `rank_bounded[il] = q_dim < head_dim[il]`;
`mqa_ratio = n_head/max(n_head_kv,1)`; `kv_bypass_fraction = non-KV layers / n_layer`;
global-vs-local layer list; **arch_family** = one of the seven above.

### 3. POLICY — the decision function
Emits `{family, features, ship_config, aggressive_config, chosen_tier, flags,
rationale[]}`. `--target-bpw N` selects the tier (≥3.8 → TQ4 ship, 2.8–3.8 → TQ3,
<2.8 → TQ2) and the policy **refuses** unsafe tiers per family (e.g. TQ2 on a
conv-hybrid emits the TQ4 floor + a loud flag). `--emit-plan FILE` writes a
`LEANKV_KV_PLAN` `.types` file for the chosen tier, one `<il> <ktype> <vtype>` line
per KV-owning layer.

```
python3 kit-v2/kv_policy.py <model.gguf> [--target-bpw N] [--emit-plan plan.types] [--json]
python3 kit-v2/kv_policy.py --validate [--models-dir DIR]
```

---

## Validation — the prober reproduces the DISCRIMINATING fields, not just the ship

Every measured model ships TQ4/TQ4 pure (ladder rule #1), so matching the ship string
alone is trivial — a prober that always says "TQ4" would pass. `--validate` therefore
asserts the fields that actually encode knowledge, from geometry alone, against the
measured ground truth: **arch family, KV-owning-layer count, rank-bounded pattern, the
TQ2 refusal at `--target-bpw 2`, and the aggressive-menu tier.** Any miscoding (wrong
family, wrong owner count, failing to refuse TQ2 on conv-hybrid, …) makes it FAIL.

| model | family | kv_own | rb | TQ2✗ | checks (family/kv_own/rb/refuse/aggr) |
|---|---|---|---|---|---|
| LFM2.5-1.2B | conv-hybrid | 6/16 | no | y | PASS |
| LFM2.5-8B-A1B | conv-hybrid | 6/24 | no | y | PASS |
| Qwen3.5-2B | SSM-hybrid | 6/24 | no | **n** | PASS |
| Qwen3.5-4B | SSM-hybrid | 8/32 | all | **n** | PASS |
| Qwen3.5-9B | SSM-hybrid | 8/32 | no | **n** | PASS |
| gemma-3-4b | dense-GQA | 34/34 | no | y | PASS |
| gemma-4-E2B | shared-KV-MQA | **15/35** | all | y | PASS |
| gemma-4-E4B | shared-KV-GQA | **24/42** | globals | y | PASS |

**8/8 on every discriminating field.** The TQ2✗ column shows the prober correctly refuses
TQ2 on conv-hybrid/dense/shared-KV (measured dead) while *allowing* it on SSM-hybrid (the
Qwen dilution exception) — a distinction a trivial always-TQ4 prober cannot make. Owner
counts are engine-true (E2B 15, E4B 24 — the shared-KV filter, not the 35/42 attn_k
count). Verified by an independent adversarial pass (re-parsed GGUFs + engine-source
cross-check; it caught the original 35/42 over-count and the trivial ship-only validator,
both fixed here).

Geometry cross-checks that fell out exactly as the docs recorded them:
- **E2B** globals at il `[4,9,14,19,24,29,34]`, `q_dim 192 < 256/512` on all owners,
  `shared_kv 20/35`. Matches the campaign doc.
- **E4B** globals at `[5,11,17,23,29,35,41]`, `q_dim 320 > 256` (locals free),
  `< 512` (only the 7 globals bounded), `shared_kv 18/42`. Matches "24 own / 18
  shared; only the globals remain bounded".
- **LFM2.5-1.2B** attn layers `[2,5,8,10,12,14]` (6/16), `q_dim 64 = head_dim` (no
  rank slack). **LFM2.5-8B-A1B** attn `[2,6,10,14,18,21]` (6/24). Matches campaigns 2–3.
- **Qwen3.5-4B** is the SSM-hybrid (`qwen35.ssm.*` present); attn layers
  `[3,7,11,15,19,23,27,31]` (8/32), the rest mamba/deltanet — matches the doc's
  "8/N layers have KV, mamba layers bypass".

---

## Honest scope — validated vs defaulted

- **Directly validated (ship = TQ4/TQ4 measured):** gemma-4 E2B, gemma-4 E4B,
  LFM2.5-1.2B, LFM2.5-8B-A1B — the four production-ladder decision-gate rows.
- **Policy-inferred, family-consistent:**
  - **gemma-3-4b** ships TQ4/TQ4 by the dense rule + ladder rule #1. Its doc number
    is the *legacy* A1@3.0 mixed plan (0.392/77%), which rule #1 explicitly
    supersedes ("every ship pick clears it by a wide margin"). The prober says so in
    the rationale.
  - **Qwen3.5-2B/4B/9B** are the SSM-hybrid *family* the docs measured, but **not the
    identical checkpoint** (the doc measured an 8/36-layer Qwen; these are
    8/32-and-similar). TQ4 pure is the safe floor; the **TQ2 exception is
    dilution-driven, and on Qwen3.5-4B the attn layers are *also* rank-bounded**
    (`q_dim 160 < 256`), so the Q-dim gate would promote TQ2→TQ4 unless
    `LEANKV_NO_QDIM_GATE=1`. The prober flags this tension and says *measure the exact
    checkpoint before trusting sub-TQ4*. It does not claim the doc's +2.6% for a
    checkpoint it wasn't measured on.
- **UNKNOWN arches** get TQ4/TQ4 + an `UNVALIDATED: measure before trusting` flag and
  **no** aggressive tier. Encoders (bert / bge) are reported `no-kv-cache` (nothing
  to quantize) rather than force-fit.
- **What the prober cannot see (needs the runtime kvimp pass):** the exact
  cross-layer *reuse graph* (which owner feeds how many consumers), per-layer
  variance/sink positions, and empirical rank fill. For shared-KV budget plans it
  therefore *recommends* the `kv_bit_allocator.py --arm A1R --norm robust` command
  rather than emitting the reuse-weighted bits itself. Ship (TQ4 pure) needs none of
  that; only sub-4bpw budget plans do.

---

## Roadmap — wiring into the engine

Today the prober is **advisory**: it emits a plan file + flags for the operator to
apply (`LEANKV_KV_PLAN=plan.types`, plus the scale/gate env already in the codec).
The follow-on is auto-config-at-load. Cheapest first:

1. **Emit-and-apply loop (now):** `kv_policy.py --emit-plan` → `LEANKV_KV_PLAN`.
   Ship path is complete for all six families.
2. **Fold `mse_opt`/`amax` gate + `--norm robust` into per-tier defaults** so the
   operator doesn't set three env vars — the scale-scheme doc's shippable win.
3. **Reuse-weighted budget on demand:** when `--target-bpw < 4` on a shared-KV
   family, chain a kvimp pass → allocator `--arm A1R` automatically instead of
   printing the command.
4. **Auto-config-at-load:** have the loader call the prober on the GGUF it just
   opened and set the KV cache types itself (TQ4 floor, gate on) with an operator
   override — the compiler's endgame: *the model tells the engine how to quantize
   its own cache.*
5. **New-arch intake:** on an `UNKNOWN`/`measure-first` result, the prober's flag is
   the trigger to run the kvimp + KLD ladder (README loop) and add a menu row — the
   process that produced every row above.

Provenance: `leankv-tq-production-ladder-2026-07.md`,
`leankv-adaptive-kv-compiler.md`, `leankv-scale-scheme-study-2026-07.md`,
`leankv-perchannel-e2b-study-2026-07.md`, `leankv-e2b-campaign-2026-07.md`,
`leankv-kv-importance-ablation-2026-07.md`,
`leankv-entropy-lfm2-campaign-2026-07.md`. Prober: `kit-v2/kv_policy.py`
(reuses `kit-v2/gguf_extract.py`; recommends `kit-v2/kv_bit_allocator.py --arm A1R`).
