// LeanKV Phase 7a Stage 4a: runtime-configurable TQ codebooks.
// See ggml-tq-runtime.h for the surface and lifecycle.

#include "ggml-tq-runtime.h"
#include "ggml-common.h" // for the Gaussian defaults (tq{2,3,4}_values)

#include <stdio.h>
#include <string.h>

// ── Default (Gaussian-on-N(0,1), Lloyd-Max, scaled by 127) tables ────
//
// These mirror the hardcoded constants that used to live in
// iqk_gemm_legacy_quants.cpp / iqk_fa_templates.h / ggml-tq.c. Keeping
// them here means any site that wants "the default" can just call the
// getter with no codebook installed, and the behavior is identical to
// the pre-7a build.

static const int8_t k_default_tq2_i8[16] = {
    -127, -38, +38, +127, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
};
static const int8_t k_default_tq3_i8[16] = {
    -127, -79, -45, -14, +14, +45, +79, +127, 0, 0, 0, 0, 0, 0, 0, 0,
};
static const int8_t k_default_tq4_i8[16] = {
    -127, -96, -75, -58, -44, -31, -18, -6,
      +6, +18, +31, +44, +58, +75, +96, +127,
};

static const float k_default_tq2_f[16] = {
    -1.0000000f, -0.2997714f, +0.2997714f, +1.0000000f,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
};
static const float k_default_tq3_f[16] = {
    -1.0000000f, -0.6245203f, -0.3513239f, -0.1138989f,
    +0.1138989f, +0.3513239f, +0.6245203f, +1.0000000f,
    0, 0, 0, 0, 0, 0, 0, 0,
};
static const float k_default_tq4_f[16] = {
    -1.0000000f, -0.7573038f, -0.5923403f, -0.4599576f,
    -0.3450764f, -0.2405254f, -0.1421261f, -0.0470277f,
    +0.0470277f, +0.1421261f, +0.2405254f, +0.3450764f,
    +0.4599576f, +0.5923403f, +0.7573038f, +1.0000000f,
};

// ── Mutable state ────────────────────────────────────────────────────

static int8_t g_tq2_i8[16];
static int8_t g_tq3_i8[16];
static int8_t g_tq4_i8[16];

static float  g_tq2_f [16];
static float  g_tq3_f [16];
static float  g_tq4_f [16];

static int    g_tq2_fitted;
static int    g_tq3_fitted;
static int    g_tq4_fitted;

static int    g_initialized;

// ── Per-layer overrides (Stage 4b) ──────────────────────────────────
//
// We pre-allocate the full override tables statically. 3 bit-widths ×
// 128 layers × 16 entries × (1 byte int8 + 4 byte float) = 24 KiB total.
// Small enough to leave in BSS; large enough to cover any current model.

static int8_t g_layer_i8[3][GGML_TQ_MAX_LAYERS][16];
static float  g_layer_f [3][GGML_TQ_MAX_LAYERS][16];
static uint8_t g_layer_present[3][GGML_TQ_MAX_LAYERS]; // 1 iff override installed

static inline int bits_to_slot(int bits) {
    switch (bits) {
        case 2: return 0;
        case 3: return 1;
        case 4: return 2;
        default: return -1;
    }
}

// ── K-cache pointer → layer registry (Stage 4b) ────────────────────

struct kr_entry {
    int           il;
    const uint8_t * base;
    size_t        nbytes;
};
static struct kr_entry g_kr[GGML_TQ_MAX_LAYERS];
static int g_kr_count;

static void ensure_init(void) {
    if (g_initialized) return;
    memcpy(g_tq2_i8, k_default_tq2_i8, sizeof(g_tq2_i8));
    memcpy(g_tq3_i8, k_default_tq3_i8, sizeof(g_tq3_i8));
    memcpy(g_tq4_i8, k_default_tq4_i8, sizeof(g_tq4_i8));
    memcpy(g_tq2_f,  k_default_tq2_f,  sizeof(g_tq2_f));
    memcpy(g_tq3_f,  k_default_tq3_f,  sizeof(g_tq3_f));
    memcpy(g_tq4_f,  k_default_tq4_f,  sizeof(g_tq4_f));
    g_tq2_fitted = g_tq3_fitted = g_tq4_fitted = 0;
    g_initialized = 1;
}

// ── Setters / getters ────────────────────────────────────────────────

void ggml_tq_set_runtime_levels_i8(int bits, const int8_t * levels_i8) {
    ensure_init();
    if (!levels_i8) return;
    switch (bits) {
        case 2: memcpy(g_tq2_i8, levels_i8, sizeof(g_tq2_i8)); g_tq2_fitted = 1; return;
        case 3: memcpy(g_tq3_i8, levels_i8, sizeof(g_tq3_i8)); g_tq3_fitted = 1; return;
        case 4: memcpy(g_tq4_i8, levels_i8, sizeof(g_tq4_i8)); g_tq4_fitted = 1; return;
        default:
            fprintf(stderr, "ggml-tq-runtime: set_levels_i8: ignoring bits=%d\n", bits);
            return;
    }
}

void ggml_tq_set_runtime_levels_f(int bits, const float * levels_f) {
    ensure_init();
    if (!levels_f) return;
    switch (bits) {
        case 2: memcpy(g_tq2_f, levels_f, sizeof(g_tq2_f)); g_tq2_fitted = 1; return;
        case 3: memcpy(g_tq3_f, levels_f, sizeof(g_tq3_f)); g_tq3_fitted = 1; return;
        case 4: memcpy(g_tq4_f, levels_f, sizeof(g_tq4_f)); g_tq4_fitted = 1; return;
        default:
            fprintf(stderr, "ggml-tq-runtime: set_levels_f: ignoring bits=%d\n", bits);
            return;
    }
}

const int8_t * ggml_tq_get_runtime_levels_i8(int bits) {
    ensure_init();
    switch (bits) {
        case 2: return g_tq2_i8;
        case 3: return g_tq3_i8;
        case 4: return g_tq4_i8;
        default: return NULL;
    }
}

const float * ggml_tq_get_runtime_levels_f(int bits) {
    ensure_init();
    switch (bits) {
        case 2: return g_tq2_f;
        case 3: return g_tq3_f;
        case 4: return g_tq4_f;
        default: return NULL;
    }
}

void ggml_tq_reset_runtime_levels(void) {
    memcpy(g_tq2_i8, k_default_tq2_i8, sizeof(g_tq2_i8));
    memcpy(g_tq3_i8, k_default_tq3_i8, sizeof(g_tq3_i8));
    memcpy(g_tq4_i8, k_default_tq4_i8, sizeof(g_tq4_i8));
    memcpy(g_tq2_f,  k_default_tq2_f,  sizeof(g_tq2_f));
    memcpy(g_tq3_f,  k_default_tq3_f,  sizeof(g_tq3_f));
    memcpy(g_tq4_f,  k_default_tq4_f,  sizeof(g_tq4_f));
    g_tq2_fitted = g_tq3_fitted = g_tq4_fitted = 0;
    memset(g_layer_present, 0, sizeof(g_layer_present));
    memset(g_layer_i8, 0, sizeof(g_layer_i8));
    memset(g_layer_f,  0, sizeof(g_layer_f));
    memset(g_kr, 0, sizeof(g_kr));
    g_kr_count = 0;
    g_initialized = 1;
}

int ggml_tq_has_fitted_levels(int bits) {
    ensure_init();
    switch (bits) {
        case 2: return g_tq2_fitted;
        case 3: return g_tq3_fitted;
        case 4: return g_tq4_fitted;
        default: return 0;
    }
}

// ── Per-layer overrides ─────────────────────────────────────────────

void ggml_tq_set_layer_override_i8(int bits, int il, const int8_t * levels_i8) {
    ensure_init();
    if (!levels_i8) return;
    const int slot = bits_to_slot(bits);
    if (slot < 0) {
        fprintf(stderr, "ggml-tq-runtime: set_layer_override_i8: ignoring bits=%d\n", bits);
        return;
    }
    if (il < 0 || il >= GGML_TQ_MAX_LAYERS) {
        fprintf(stderr, "ggml-tq-runtime: set_layer_override_i8: il=%d out of range [0,%d)\n",
                il, GGML_TQ_MAX_LAYERS);
        return;
    }
    memcpy(g_layer_i8[slot][il], levels_i8, 16);
    g_layer_present[slot][il] = 1;
}

void ggml_tq_set_layer_override_f(int bits, int il, const float * levels_f) {
    ensure_init();
    if (!levels_f) return;
    const int slot = bits_to_slot(bits);
    if (slot < 0) {
        fprintf(stderr, "ggml-tq-runtime: set_layer_override_f: ignoring bits=%d\n", bits);
        return;
    }
    if (il < 0 || il >= GGML_TQ_MAX_LAYERS) {
        fprintf(stderr, "ggml-tq-runtime: set_layer_override_f: il=%d out of range [0,%d)\n",
                il, GGML_TQ_MAX_LAYERS);
        return;
    }
    memcpy(g_layer_f[slot][il], levels_f, sizeof(float) * 16);
    g_layer_present[slot][il] = 1;
}

const int8_t * ggml_tq_get_levels_for_layer_i8(int bits, int il) {
    ensure_init();
    const int slot = bits_to_slot(bits);
    if (slot < 0) return NULL;
    if (il >= 0 && il < GGML_TQ_MAX_LAYERS && g_layer_present[slot][il]) {
        return g_layer_i8[slot][il];
    }
    // Fall back to global runtime LUT.
    return ggml_tq_get_runtime_levels_i8(bits);
}

const float * ggml_tq_get_levels_for_layer_f(int bits, int il) {
    ensure_init();
    const int slot = bits_to_slot(bits);
    if (slot < 0) return NULL;
    if (il >= 0 && il < GGML_TQ_MAX_LAYERS && g_layer_present[slot][il]) {
        return g_layer_f[slot][il];
    }
    return ggml_tq_get_runtime_levels_f(bits);
}

// ── K-cache pointer → layer registry ────────────────────────────────

void ggml_tq_register_k_cache_range(int il, const void * base, size_t nbytes) {
    ensure_init();
    if (!base || nbytes == 0) return;
    if (il < 0 || il >= GGML_TQ_MAX_LAYERS) {
        fprintf(stderr, "ggml-tq-runtime: register_k_cache_range: il=%d out of range\n", il);
        return;
    }
    if (g_kr_count >= GGML_TQ_MAX_LAYERS) {
        fprintf(stderr, "ggml-tq-runtime: register_k_cache_range: registry full (%d)\n",
                g_kr_count);
        return;
    }
    g_kr[g_kr_count].il     = il;
    g_kr[g_kr_count].base   = (const uint8_t *)base;
    g_kr[g_kr_count].nbytes = nbytes;
    g_kr_count++;
}

void ggml_tq_clear_k_cache_ranges(void) {
    ensure_init();
    memset(g_kr, 0, sizeof(g_kr));
    g_kr_count = 0;
}

int ggml_tq_lookup_k_cache_layer(const void * ptr) {
    ensure_init();
    if (!ptr || g_kr_count == 0) return -1;
    const uint8_t * p = (const uint8_t *)ptr;
    for (int i = 0; i < g_kr_count; i++) {
        if (p >= g_kr[i].base && p < g_kr[i].base + g_kr[i].nbytes) {
            return g_kr[i].il;
        }
    }
    return -1;
}
