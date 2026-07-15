#pragma once

// LeanKV: KV-importance calibration collector ("kvimp")
//
// Companion to the offline bit-allocator (kv_bit_allocator.py v2). During a
// normal forward pass over a calibration corpus it accumulates, per
// (layer, kv-head), the statistics the allocator's ablation arms A1..A6 need:
//
//   v_importance   downstream sensitivity of V-cache channels, read off the
//                  attn_output (W_o) matmul INPUT (== attention output ==
//                  weighted sum of V vectors), reduced per kv head.
//   k_importance   per-head Q energy Sum E[q_i^2], read off the attn_q matmul
//                  OUTPUT. Head-level energy is RoPE-invariant, so pre-RoPE
//                  capture is exact at this granularity.
//   k_var / v_var  second moments of the cached K/V (attn_k / attn_v matmul
//                  OUTPUTS). Per-head energy is Hadamard-invariant, so the
//                  pre-rotation capture already equals the rotated-basis
//                  statistic at head granularity (verified numerically).
//   k_*_rope/pass  per-partition K stats on p-RoPE layers (rope_fraction<1),
//                  feeding the allocator's A5 arm. Partition energies are
//                  RoPE-invariant (RoPE rotates pairs inside the partition).
//
// Geometry (incl. Gemma-4 cross-layer KV sharing) is populated automatically
// from the model's hparams at context creation:
//   owns_kv       = hparams.has_kv(il)        (n_layer_kv_from_start)
//   owner         = backward scan through owned layers for the same layer
//                   type (sliding vs global), mirroring the rule in
//                   gemma4_mtp_target_kv_layer()  [VERIFY on first real E2B
//                   run against the shared-layer attention path]
//   rope_fraction = hparams.rope_n_rot(il) / hparams.n_embd_head_k(il)
//   is_global     = swa_layers[il] == 0
//
// Consumer layers (owns_kv == false) contribute their attn_q / attn_output
// statistics into the OWNER's accumulators — measured downstream sensitivity
// summed over all consumers, the principled version of the reuse-count proxy.
// reuse_count is emitted as well; do not multiply both in the allocator
// (double-counts): compare use_reuse=True vs owner-accumulated importance as
// its own mini-ablation.
//
// Env vars:
//   LEANKV_KVIMP        "1" to enable (any non-"0" value)
//   LEANKV_KVIMP_PATH   output JSON path (default: kv_stats.json)
//
// Usage: any binary works as the driver. Calibrate on CPU so activations are
// captured in native f32 (same protocol as Phase 7):
//   LEANKV_KVIMP=1 LEANKV_KVIMP_PATH=/tmp/kv_stats.json \
//     ./build/bin/llama-cli -m model.gguf -ngl 0 -c 1024 -n 4 -f calib.txt
// The JSON is written when the context is destroyed.
//
// Known caveats (documented, deliberate):
//   * Q energy is read at the attn_q matmul output — i.e. BEFORE Gemma-style
//     Q/K RMS-norm. On QK-norm archs (Gemma-4) this is a proxy; if A3 is
//     borderline on E2B, add a Qcur_normed capture and re-check. [VERIFY-6]
//   * Rope'd channels are assumed to be the LEADING rope_n_rot dims of each
//     head (llama.cpp convention). [VERIFY-4]
//   * Layers with n_head_kv == 0 (pure recurrent / DeltaNet) are skipped.

#include <cstdint>

struct ggml_tensor;
struct llama_model;

struct leankv_kvimp_state;  // opaque

// Reads LEANKV_KVIMP; returns non-null only when enabled.
leankv_kvimp_state * leankv_kvimp_init(const llama_model & model);

// Scheduler eval-callback body. ask==true -> return 1 if the node's data is
// wanted; ask==false -> node computed, accumulate. Never modifies the graph.
int leankv_kvimp_cb(leankv_kvimp_state * st, struct ggml_tensor * t, bool ask);

// Writes the JSON (path from env) and frees. Safe on nullptr.
void leankv_kvimp_free(leankv_kvimp_state * st);
