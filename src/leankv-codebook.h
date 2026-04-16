#pragma once

// LeanKV Phase 7a: empirical codebook cache.
//
// Owns the on-disk format and fingerprinting for per-model Lloyd-Max
// codebooks fitted at first-load time. See docs/phase7-calibration/README.md
// for the empirical findings that motivate this feature.
//
// Lifecycle (once step 4 is wired):
//
//   1. User runs llama-cli with `-ctk tq3_0`.
//   2. Model loads. We compute a fingerprint over a stable sample of the
//      model's tensor data (the actual weights, not metadata).
//   3. We look for ~/.cache/leankv/<fingerprint>.codebook.
//      - Cache hit  → mmap / read the codebook, kernels use those levels.
//      - Cache miss → run a short warm-up inference on the embedded
//                     calibration corpus with the K-vector dump hook
//                     enabled, fit Lloyd-Max on the collected values,
//                     save the codebook, then proceed with real inference.
//   4. If fitting fails or produces degenerate levels, we persist a
//      "use gaussian defaults" marker so subsequent loads skip the attempt.
//
// This header only exposes the *offline* surface (cache file read/write,
// fingerprint). Step 5 (automatic warm-up trigger) lives in
// leankv-calib.cpp alongside the existing dump hook.
//
// ── On-disk format ─────────────────────────────────────────────────────
//
// All little-endian, no padding. Designed to be greppable with `xxd`
// and forward/backward compatible via the version field.
//
//   struct leankv_codebook_file_header {
//       u32 magic;          // 'LKCB' = 0x42434B4C
//       u32 version;        // 1
//       u64 fingerprint;    // FNV-1a over tensor sample + corpus_version
//       u32 n_levels;       // 4 / 8 / 16 (TQ2 / TQ3 / TQ4)
//       u32 n_overrides;    // number of per-layer overrides (e.g. L0)
//       u32 bits;           // 2 / 3 / 4
//       u32 flags;          // bit 0 = LEANKV_CODEBOOK_USE_DEFAULT (calibration punted)
//       char arch[32];      // architecture string (qwen3, llama, mistral, ...)
//       u32 n_layers;       // total layers in the model (for sanity check)
//       u32 reserved[3];    // zero-padded for future use
//   };
//
//   After the header:
//     float global_levels[n_levels];                  // symmetric, sorted asc, outer = ±1
//     for i in 0..n_overrides:
//         u32   layer_idx;
//         float layer_levels[n_levels];
//
// Total size with 8 global levels + 1 override: 128 bytes header + 32 B global
// + 36 B override = 196 B. Small enough that we always read the whole file.

#include <cstdint>
#include <cstddef>
#include <string>

namespace leankv {

inline constexpr uint32_t CODEBOOK_FILE_MAGIC = 0x42434B4Cu; // 'LKCB'
inline constexpr uint32_t CODEBOOK_FILE_VERSION = 1u;

// Flags (bitfield on `flags` in the header).
inline constexpr uint32_t CODEBOOK_FLAG_USE_DEFAULT = 1u << 0; // fitting was punted; use Gaussian

inline constexpr int CODEBOOK_MAX_LEVELS    = 16;  // TQ4
inline constexpr int CODEBOOK_MAX_OVERRIDES = 8;   // we only ever expect L0 in practice

struct codebook_layer_override {
    uint32_t layer_idx = 0;
    float    levels[CODEBOOK_MAX_LEVELS] = {0};
};

struct codebook {
    uint32_t version     = CODEBOOK_FILE_VERSION;
    uint64_t fingerprint = 0;
    int      bits        = 0;                // 2/3/4
    int      n_levels    = 0;                // 4/8/16
    int      n_layers    = 0;
    char     arch[32]    = {0};
    uint32_t flags       = 0;

    float    global_levels[CODEBOOK_MAX_LEVELS] = {0};

    int                     n_overrides = 0;
    codebook_layer_override overrides[CODEBOOK_MAX_OVERRIDES];

    // Convenience: returns overrides[il].levels if present, else global_levels.
    const float * levels_for_layer(int il) const;

    // True if the codebook was populated (either a real fit or a default marker).
    bool valid() const { return n_levels > 0; }

    // True if this codebook is the "use Gaussian defaults" marker.
    bool is_default_marker() const { return (flags & CODEBOOK_FLAG_USE_DEFAULT) != 0; }
};

// ── Fingerprint ─────────────────────────────────────────────────────

// 64-bit FNV-1a. Stable, no external deps, good enough for cache keying.
uint64_t fnv1a_64(const void * data, size_t n);

// Combine two 64-bit hashes (Boost-style mix).
uint64_t hash_mix(uint64_t a, uint64_t b);

// ── Cache path resolution ───────────────────────────────────────────

// Returns ~/.cache/leankv (respecting XDG_CACHE_HOME on Linux/macOS).
// Creates the directory if it does not exist.
std::string codebook_cache_dir();

// Returns <cache_dir>/<hex_fingerprint>_<bits>.codebook.
std::string codebook_cache_path(uint64_t fingerprint, int bits);

// ── Read / write ────────────────────────────────────────────────────

// Attempts to read `path` into `out`. Returns true on success.
// Validates magic, version, expected fingerprint (if nonzero), and sanity.
// Errors are logged to stderr via the leankv-calib: prefix.
bool codebook_load(const std::string & path,
                   uint64_t             expected_fingerprint,
                   codebook *           out);

// Writes `cb` to `path` atomically (temp file + rename).
bool codebook_save(const std::string & path,
                   const codebook &    cb);

// ── Default-marker helper ───────────────────────────────────────────

// Populates a codebook as the "use Gaussian defaults" marker so we can
// persist a negative result instead of retrying fitting on every load.
void codebook_make_default(codebook * cb,
                           uint64_t   fingerprint,
                           int        bits,
                           const char *arch,
                           int        n_layers);

// ── Level post-fit validation ───────────────────────────────────────

// Checks: symmetry, monotonicity, outermost ±1, no NaN, minimum spread.
// Returns true if `levels` is usable as a runtime codebook.
bool codebook_levels_ok(const float * levels, int n_levels);

// ── Runtime-LUT conversion helpers ──────────────────────────────────
//
// The kernel-side runtime tables (ggml-tq-runtime.{h,c}) take a 16-entry
// padded int8 LUT or a 16-entry padded float LUT. We always scale floats
// by 127 (the canonical block-scale convention shared across quantize,
// dequantize, and vec_dot paths) and round to the nearest int8.
//
// `out_i8`   — 16-byte buffer. Unused slots zero-padded.
// `out_f`    — 16-float buffer. Unused slots zero-padded.
// `in_f`     — normalized levels (outer = ±1), length `n_levels`.
// `n_levels` — 4 / 8 / 16 for bits 2 / 3 / 4.
//
// Returns false on bad inputs (null, bad n_levels, or a level outside
// [-1, 1]).
bool codebook_levels_to_i8(const float * in_f, int n_levels, int8_t * out_i8);
bool codebook_levels_to_f (const float * in_f, int n_levels, float  * out_f);

} // namespace leankv
