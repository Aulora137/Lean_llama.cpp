# LeanKV ablation kit v2 — 2026-07-17

Two `git am`-able patches on main @ 6293da0f (order matters):
1. `0001-...kvimp...` — in-tree KV-importance collector (LEANKV_KVIMP=1)
2. `0002-...bit-plan-consumer...` — LEANKV_KV_PLAN consumer + per-layer-type
   Q-dim rank gate (replaces the old layer-0 global gate)

`kv_bit_allocator.py` here supersedes `KV Quant/hardened/` (adds --emit-types).

## The complete loop (any model, one machine)

```bash
# 0. apply once
git am 0001-*.patch 0002-*.patch && cmake --build build -j$(nproc)

# 1. CALIBRATE (CPU so activations are f32; same protocol as Phase 7)
LEANKV_KVIMP=1 LEANKV_KVIMP_PATH=kv_stats.json \
  ./build/bin/llama-cli -m model.gguf -ngl 0 -c 1024 -n 4 -f calib_corpus.txt

# 2. ALLOCATE (arms A1..A6; --bmax 4 keeps bits on the exact TQ ladder)
python3 kv_bit_allocator.py kv_stats.json --arm A1 --bpw 3.0 --bmax 4 --emit-types plan_A1.types
python3 kv_bit_allocator.py kv_stats.json --arm A3 --bpw 3.0 --bmax 4 --emit-types plan_A3.types

# 3. MEASURE — KL-to-F16 via llama-perplexity's KLD mode
#    (verify dataset first: md5 wiki.test.raw == 7c0137fc034ddbc56a296bce31b4f7fb)
./build/bin/llama-perplexity -m model.gguf -f wiki.test.raw -c 2048 \
    --kl-divergence-base base_f16.kld          # once: save F16 logits
LEANKV_KV_PLAN=plan_A1.types ./build/bin/llama-perplexity -m model.gguf \
    -f wiki.test.raw -c 2048 --kl-divergence-base base_f16.kld --kl-divergence
LEANKV_KV_PLAN=plan_A3.types ./build/bin/llama-perplexity -m model.gguf \
    -f wiki.test.raw -c 2048 --kl-divergence-base base_f16.kld --kl-divergence
# -> compare mean KLD (and same-top %). A3 < A1 at matched bpw = GO.
```

## Semantics worth knowing
- Plan lines: `<layer> <ktype> <vtype>`; `#` comments. Unlisted layers keep
  the -ctk/-ctv defaults. Plan overrides outlier auto-detect.
- The Q-dim rank gate (q_dim = n_embd/n_head < head_dim → TQ3/TQ2* unsafe)
  now judges EACH layer — on Gemma-4 it can pass locals (256) and flag
  globals (512). Under a plan it warns instead of rewriting;
  LEANKV_NO_QDIM_GATE=1 silences it for deliberate stress tests.
- Matched-bpw note: allocator bits {2,3,4} are nominal; TQ stores
  {2.5,3.5,4.5} bits/elem (block scale). The +0.5 offset is identical
  across arms, so arm comparisons stay fair.
- GQA models: --emit-types reduces per layer via max-bits (conservative);
  on MQA (Gemma-4 E2B/E4B) per-layer IS per-head — exact.
- Smoke evidence (cloud, tiny GQA model): plan tq2/tq2 → tq4/tq4 ramp
  applied 4/4 layers, mixed-type decode clean, KV 0.25→0.05 MiB.
- Cosmetic: the "KV self size" log line labels V as its global type even
  when per-layer V types are active (K says "adaptive"); numbers are right.
