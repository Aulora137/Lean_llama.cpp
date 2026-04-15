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

struct ggml_tensor;

struct leankv_calib_state {
    FILE * file        = nullptr;
    char   path[1024]  = {0};
    int    n_records   = 0;
    int    max_records = 0;
};

// Returns true if LEANKV_CALIBRATION_DUMP is set (cached after first call).
bool leankv_calib_enabled();

// Allocates a new calibration state, opens the output file, and writes the
// file header. Returns nullptr if calibration is disabled or fopen fails.
leankv_calib_state * leankv_calib_init();

// Closes the file and frees the state. Safe to call with nullptr.
void leankv_calib_free(leankv_calib_state * s);

// Reads tensor data from the backend and appends one record to the file.
// Returns true on success. Called from the scheduler eval callback.
bool leankv_calib_dump_tensor(leankv_calib_state * s, const struct ggml_tensor * t, int il);
