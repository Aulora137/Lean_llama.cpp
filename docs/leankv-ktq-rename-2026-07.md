# KV TurboQuant type rename: `tq*_0` → `ktq*_0` (2026-07-21)

**Why.** Mainline llama.cpp uses `GGML_TYPE_TQ1_0 = 34` (ternary weights) and
`GGML_TYPE_TQ2_0 = 35` (2-bit weights) for BitNet/TriLM-style **weight** quantization.
Our fork used the same C identifier `GGML_TYPE_TQ2_0` (= 44) for a **KV-cache** type — a
name collision that (a) blocks porting mainline's ternary/1-bit weight support (needed to
run Bonsai-ternary, Hunyuan Hy3-1bit, HY-1.8B-2Bit, …) and (b) would break on the next
upstream merge. The *enum values* never clashed (ours are 42–45, mainline's 34–35); only
the C symbol names and the `type_name` strings did.

**What changed.** All four fork KV types were renamed, C symbol + block struct + kernels +
`type_name`:

| was | now | enum value (unchanged) |
|---|---|---|
| `GGML_TYPE_TQ3_0` | `GGML_TYPE_KTQ3_0` | 42 |
| `GGML_TYPE_TQ4_0` | `GGML_TYPE_KTQ4_0` | 43 |
| `GGML_TYPE_TQ2_0` | `GGML_TYPE_KTQ2_0` | 44 |
| `GGML_TYPE_TQ2_1` | `GGML_TYPE_KTQ2_1` | 45 |

`block_tq*_0 → block_ktq*_0`, `quantize_row_tq*_0 → quantize_row_ktq*_0` (+ dequantize,
vec_dot), and `ggml_type_name` now returns `"ktq4_0"` etc. Renamed across ggml (CPU + the
CUDA/Metal/IQK kernels — CUDA/Metal renamed uniformly but only the CPU path was compiled
and verified here), `src/`, `common/`, `examples/`, `tests/`.

**Backward compatibility — nothing operational breaks.** Our KV TQ types are *runtime
cache types* selected by `-ctk/-ctv`, never written into a saved GGUF, so the rename has
zero file-format impact. Both spellings are accepted:
- CLI (`kv_cache_type_from_str`): `-ctk ktq4_0` **and** legacy `-ctk tq4_0` → `KTQ4_0`.
- Plan files (`leankv_type_from_name`): a `tq*_0` token is mapped forward to `ktq*_0`
  before lookup, so old `LEANKV_KV_PLAN` files still resolve.
- The node's current `-ctk tq4_0 -ctv tq4_0` launch flags keep working unchanged.

The tools (`kv_policy.py`, `kv_bit_allocator.py`) now emit the canonical `ktq*_0`; the
`ktq/tq` aliases mean the docs' `tq4_0` examples remain valid.

**Verified.** Full CPU build clean; `-ctk tq4_0` and `-ctk ktq4_0` produce identical
output (same KV size, same PPL); `ggml_type_name` reports `ktq4_0`; `kv_policy --validate`
still 8/8.

**Porting mainline ternary/1-bit weights now.** With the `TQ1_0`/`TQ2_0` names freed, add
mainline's `GGML_TYPE_TQ1_0 = 34` / `TQ2_0 = 35` at their canonical values + names (block
structs, kernels, ftype). The fork already carries adjacent pieces: `I2_S = 36` (MS
BitNet), `Q1_0_G128 = 41` (Bonsai 1-bit). This rename was the specific blocker.
