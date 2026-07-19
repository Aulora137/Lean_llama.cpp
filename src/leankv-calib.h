#pragma once

// LeanKV Phase 7: K-vector calibration dump
//
// When enabled via the LEANKV_CALIBRATION_DUMP=1 environment variable, the
// forward pass writes raw per-layer K tensors (after RoPE, before cache
// storage) to a binary file for offline SVD analysis. This is used to
// discover the effective rank of the K-subspace per layer, which in turn
// drives the rank-aware rotation matrices described in docs/experiment.md
// Phase 7.
//
// Env vars:
//   LEANKV_CALIBRATION_DUMP        "1" to enable (any non-"0" value)
//   LEANKV_CALIBRATION_DUMP_PATH   output file (default: leankv_k_calib.bin)
//   LEANKV_CALIBRATION_DUMP_MAX    max records to write (default: 0 = unlimited)
//   LEANKV_CALIBRATION_DUMP_Q_PATH when set (alongside LEANKV_CALIBRATION_DUMP=1),
//                                  ALSO capture the post-RoPE Q tensor of every
//                                  KV-owning layer into this separate file, same
//                                  KCAL record format. Q records have
//                                  ne = [head_dim, n_head, n_tokens] (n_head,
//                                  not n_head_kv).
//
// File format (little-endian):
//   u32 magic = 'KCAL' (0x4C41434B)
//   u32 version = 1
//   records...
//
// Each record:
//   u32 rec_magic = 'LKCR' (0x52434B4C)
//   u32 layer_idx
//   u32 dtype         (ggml_type: 0=f32, 1=f16, ...)
//   u32 n_dims
//   u32 ne[4]
//   u32 nb[4]
//   u32 n_bytes
//   u8  data[n_bytes]

#include <cstdio>
#include <cstdint>
#include <vector>

struct ggml_tensor;

namespace leankv { struct codebook; }

// ── K-vector dump hook (Phase 7 MVE) ─────────────────────────────────
//
// In dump mode the hook writes every captured K tensor out to a .bin file
// verbatim, for offline analysis. Used to collect the data that motivates
// the empirical codebook + rank-aware rotation.

struct leankv_calib_state {
    FILE * file        = nullptr;
    char   path[1024]  = {0};
    int    n_records   = 0;
    int    max_records = 0;

    // ── Phase 7a additions: in-memory accumulation ──
    //
    // When accumulate_in_memory is true, the hook *also* collects the raw
    // float values into `blocks` so the caller can run Lloyd-Max fitting
    // at the end of the warm-up pass without re-reading the dump file.
    // This is the mode used by first-load auto-calibration.
    bool               accumulate_in_memory = false;
    std::vector<float> blocks;                     // flat f32 buffer
    // Per-layer buffer lengths (so we can fit a per-layer override for L0
    // without re-scanning). `layer_blocks[il]` holds the float count dumped
    // for layer `il` so far. Matches the layout inside `blocks`.
    std::vector<size_t> layer_offsets;              // [n_layers+1], inclusive prefix-sum
    std::vector<size_t> layer_counts;               // [n_layers]
    int                 max_layer_seen = -1;
};

// Returns true if LEANKV_CALIBRATION_DUMP is set (cached after first call).
bool leankv_calib_enabled();

// Returns true if K-vector capture should be active in the current graph.
// This is leankv_calib_enabled() OR the runtime auto-calibration flag (set
// by leankv_autocalibrate around its warm-up pass). Callers in the graph
// builder use this to decide whether to rename k_cur to the dump prefix.
bool leankv_calib_capture_required();

// Runtime toggle used by auto-calibration. Safe to call from any thread.
void leankv_calib_set_runtime_capture(bool on);

// Returns true if post-RoPE Q capture should be active in the current graph:
// LEANKV_CALIBRATION_DUMP is enabled AND LEANKV_CALIBRATION_DUMP_Q_PATH is a
// non-empty path. Graph builders use this to rename q_cur on KV-owning layers.
bool leankv_calib_q_capture_required();

// Allocates a new calibration state, opens the output file, and writes the
// file header. Returns nullptr if calibration is disabled or fopen fails.
leankv_calib_state * leankv_calib_init();

// Allocates a calibration state configured for in-memory accumulation only
// (no file output). Used for first-load auto-calibration.
leankv_calib_state * leankv_calib_init_in_memory();

// Allocates a second calibration state that dumps post-RoPE Q tensors to
// LEANKV_CALIBRATION_DUMP_Q_PATH (same KCAL file format as the K dump).
// Returns nullptr if Q capture is not requested or fopen fails.
leankv_calib_state * leankv_calib_init_q();

// Closes the file and frees the state. Safe to call with nullptr.
void leankv_calib_free(leankv_calib_state * s);

// Reads tensor data from the backend and appends one record to the file.
// Returns true on success. Called from the scheduler eval callback.
bool leankv_calib_dump_tensor(leankv_calib_state * s, const struct ggml_tensor * t, int il);

// ── Fit ──────────────────────────────────────────────────────────────
//
// Given a calibration state that was run in in-memory accumulation mode
// (see leankv_calib_init_in_memory), fit an empirical Lloyd-Max codebook
// on the collected K-vector values and write it into `out`.
//
// `bits` is 2/3/4. `arch` and `n_layers` are used to populate the cache
// header. If the fit produces degenerate levels, `out` is populated with
// the "use Gaussian defaults" marker.
//
// Returns true if fitting succeeded (including the default-marker case).
// Returns false only if there is not enough data to attempt a fit.
bool leankv_calib_fit_codebook(const leankv_calib_state * s,
                               int                        bits,
                               int                        block_size,
                               const char *               arch,
                               int                        n_layers,
                               leankv::codebook *         out);

// ── Auto-calibration ─────────────────────────────────────────────────
//
// Top-level entry point wired into llama_init_from_model. If the model's
// K-cache was requested as a TQ type and no cached codebook exists for it,
// runs a short warm-up inference on the embedded calibration corpus with
// the K-vector capture hook active, fits a Lloyd-Max codebook, and writes
// it into ~/.cache/leankv/<fingerprint>_b<bits>.codebook. The KV cache is
// cleared afterwards so the calling context starts from a clean state.
//
// Gated on:
//   - params.type_k is one of GGML_TYPE_TQ{2,3,4}_0
//   - LEANKV_CALIBRATION_AUTO != "0"  (default: on)
//
// Safe to call from llama_init_from_model once `ctx` is fully initialized.
struct llama_context;
void leankv_autocalibrate(struct llama_context * lctx);

// Compute a stable 64-bit fingerprint of the loaded model, used as the
// cache key. Mixes the architecture string, corpus version, and a small
// sample of raw tensor data from stable tensors (tok_embd + a few wk).
// Returns 0 on failure.
struct llama_model;
uint64_t leankv_fingerprint_model(const struct llama_model * model);
