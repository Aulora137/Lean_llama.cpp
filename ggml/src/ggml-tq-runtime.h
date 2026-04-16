#pragma once

// LeanKV Phase 7a Stage 4a: runtime-configurable TQ codebooks.
//
// Historically the TQ2_0 / TQ3_0 / TQ4_0 K-cache quantizers used the
// hardcoded Lloyd-Max-on-N(0,1) levels baked into `ggml-common.h`
// (tq{2,3,4}_values) and `ggml-tq.c` (TQ{N}_LEVELS*). Phase 7a fits an
// *empirical* codebook at first-load time (see src/leankv-calib.cpp) and
// stores it in ~/.cache/leankv/<fingerprint>_b<bits>.codebook.
//
// This module owns a tiny piece of mutable global state that the fitted
// codebook is installed into at context-init time. Every dequantization
// path then reads its LUT through the getters below, so a fitted codebook
// transparently replaces the Gaussian default without touching kernel
// inner loops.
//
// Thread safety:
//   - Setter is called once, from llama_init_from_model, before any
//     forward pass on the context. Hot-path getters are pure reads of
//     globals that are never re-written after install.
//   - Multiple contexts sharing the same model observe the same levels
//     (same fingerprint → same fit → same install).
//   - Loading a second, differently-fingerprinted model within the same
//     process re-writes the state. This is not thread-safe across in-
//     flight decodes on the old model, but mirrors existing llama.cpp
//     assumptions about single-model-per-process.

#include <stddef.h> // size_t
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// bits must be 2, 3, or 4. Passing any other value is a no-op (getter)
// or a warning (setter).
//
// `levels_i8` must point to a 16-byte buffer. Indices beyond 1<<bits are
// zero-padded by the caller so the byte table is safe for PSHUFB/VTBL.
//
// `levels_f` must point to a 16-float buffer, same padding convention.
//
// All four functions are safe to call before any setter — they just
// return the baked-in Gaussian defaults from ggml-common.h / ggml-tq.c.
void         ggml_tq_set_runtime_levels_i8(int bits, const int8_t * levels_i8);
void         ggml_tq_set_runtime_levels_f (int bits, const float  * levels_f);
const int8_t * ggml_tq_get_runtime_levels_i8(int bits);
const float  * ggml_tq_get_runtime_levels_f (int bits);

// Reset all three codebooks back to the hardcoded Gaussian defaults.
// Useful for tests and for cleanly switching models within one process.
// Also clears any per-layer overrides and the K-cache range registry.
void ggml_tq_reset_runtime_levels(void);

// Introspection: true iff a fitted codebook has been installed for `bits`
// (setter has been called at least once since process start or reset).
// Consumers can use this to log "fitted" vs "default" without comparing
// the LUT contents byte-for-byte.
int  ggml_tq_has_fitted_levels(int bits);

// ── Per-layer overrides (Stage 4b) ─────────────────────────────────
//
// Some models have outlier layers (usually L0) whose Hadamard-rotated K
// distribution differs enough from the rest of the model that a single
// global LUT leaves measurable SNR on the table. The Phase 7a calibration
// pipeline persists these as per-layer overrides in the codebook file.
//
// `il` must be in [0, GGML_TQ_MAX_LAYERS). Out-of-range or negative ils
// are ignored with a stderr warning.
//
// The getter returns the layer-specific LUT if one has been installed for
// `il`, else falls back to the global runtime LUT (same pointer as
// ggml_tq_get_runtime_levels_{i8,f}). Safe to call with il = -1 to force
// the global fallback.
#define GGML_TQ_MAX_LAYERS 128

void         ggml_tq_set_layer_override_i8(int bits, int il, const int8_t * levels_i8);
void         ggml_tq_set_layer_override_f (int bits, int il, const float  * levels_f);
const int8_t * ggml_tq_get_levels_for_layer_i8(int bits, int il);
const float  * ggml_tq_get_levels_for_layer_f (int bits, int il);

// ── K-cache pointer → layer lookup (Stage 4b) ──────────────────────
//
// The FA helpers receive only a raw `const char * data` row pointer with
// no layer context. We solve this with a tiny pointer-range registry:
// at KV-cache init time, each layer's K tensor registers its
// [base, base+size) range under its layer index. The FA helper then
// resolves its `data` argument via `ggml_tq_lookup_k_cache_layer`.
//
// The lookup is a linear scan over ≤ GGML_TQ_MAX_LAYERS entries — one
// per FA helper construction, outside the inner kernel loop. Amortized
// cost is negligible.
//
// Returns -1 if the pointer does not fall within any registered range
// (e.g. the K tensor is not TQ-quantized, or the registry was never
// populated — the caller should then use the global LUT).
void ggml_tq_register_k_cache_range(int il, const void * base, size_t nbytes);
void ggml_tq_clear_k_cache_ranges(void);
int  ggml_tq_lookup_k_cache_layer(const void * ptr);

#ifdef __cplusplus
}
#endif
