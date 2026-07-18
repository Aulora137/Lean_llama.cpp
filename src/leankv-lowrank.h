#pragma once

// LeanKV Phase A: low-rank K projection (reconstruction-first)
//
// When enabled via LEANKV_KV_LOWRANK=<path>, the graph builder replaces the
// post-RoPE K tensor of selected layers with its projection onto a fixed
// orthonormal rank-r subspace BEFORE the KV-cache write:
//
//     k_cur  <-  P @ (P^T @ k_cur)        (per K vector / column)
//
// P [head_dim, rank] is fitted offline per layer with kit-v2/kv_lowrank_fit.py
// (SVD of a leankv-calib K dump). Attention kernels are untouched — they see
// an ordinary F32 K tensor that simply lives in a rank-r subspace, so this
// carries zero attention-kernel risk. V is untouched.
//
// Env vars:
//   LEANKV_KV_LOWRANK   path to a projection file (unset -> hard no-op:
//                       no file access, one cached-bool branch per layer)
//
// File format (little-endian; producer: kit-v2/kv_lowrank_fit.py — keep the
// two in sync):
//   file header: u32 magic = 'LKLR' (0x524C4B4C)
//                u32 version = 1
//                u32 n_entries
//   each entry:  u32 layer_idx
//                u32 head_dim
//                u32 rank                       (0 < rank <= head_dim)
//                f32 P[head_dim * rank]         row-major [head_dim][rank],
//                                               P[d*rank + j] = component d of
//                                               the j-th basis vector; columns
//                                               of P are orthonormal
//
// The two per-layer ggml tensors handed to the graph builder follow the
// "materialize a computed constant" pattern of llm_prepare_* in llama.cpp
// (metadata in a private ggml_context, storage in a backend buffer allocated
// with ggml_backend_alloc_ctx_tensors_from_buft, data uploaded once with
// ggml_backend_tensor_set, buffer usage = WEIGHTS). They are plain leaf
// constants from the scheduler's point of view, exactly like model weights:
//
//   p_down  ne = [head_dim, rank]  computes c = P^T k   (ggml_mul_mat
//   p_up    ne = [rank, head_dim]  computes P c          contracts ne[0])
//
// mirroring the LoRA down/up pair in llm_build_lora_mm (llama-build-context.cpp):
//   ggml_mul_mat(ctx, lora->b, ggml_mul_mat(ctx, lora->a, cur))
// with a = [n_embd, r] (down) and b = [r, n_out] (up).
//
// Scope / limits (Phase A):
//   * applied only in llm_build_kv_store, i.e. the layer-mode attention path.
//     The tensor-parallel split paths (-sm graph/attn) write the K cache
//     directly and are NOT projected.
//   * init runs once per process, on the first llama context whose model it
//     validates against; buffers live for the process lifetime (same policy
//     as the leankv-calib runtime LUT globals).
//   * layers whose head_dim does not match the file entry are skipped with a
//     warning.

struct ggml_tensor;
struct llama_model;

// Per-layer projection pair, ready to reference from a compute graph.
struct leankv_lowrank_entry {
    struct ggml_tensor * p_down = nullptr; // ne = [head_dim, rank]: c = P^T k
    struct ggml_tensor * p_up   = nullptr; // ne = [rank, head_dim]: k~ = P c
    int rank     = 0;
    int head_dim = 0;
};

// Load LEANKV_KV_LOWRANK (if set) and materialize the per-layer projection
// tensors against `model`'s buffer types. Idempotent; cheap no-op when the
// env var is unset. Logs one line per affected layer:
//   leankv-lowrank: layer <il> rank <r>/<head_dim>
void leankv_lowrank_init(const struct llama_model & model);

// Lookup for the graph builder: projection pair for layer `il`, or nullptr
// when disabled / not fitted for this layer. First branch is a cached bool,
// so the disabled path costs nothing on the hot path.
const leankv_lowrank_entry * leankv_lowrank_get(int il);
