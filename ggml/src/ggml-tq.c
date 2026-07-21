/**
 * ggml-tq.c — TurboQuant TQ2_0/TQ3_0/TQ4_0 quantize/dequantize for ggml
 *
 * Lloyd-Max optimal scalar quantization with pre-computed codebooks.
 * Designed for use with Hadamard pre-rotation (k_cache_hadamard=true)
 * which makes the input distribution approximately Gaussian.
 *
 * Reference: Zandieh et al. "TurboQuant" (2025), Google Research
 */

#include "ggml-common.h"
#include "ggml-quants.h"
#include "ggml-impl.h"
#include "ggml.h"
#include "ggml-tq-runtime.h" // LeanKV 7a Stage 4a: runtime TQ codebook (read paths)

#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ── SIMD helpers ──────────────────────────────────────────────────── */

#if defined(__AVX2__)
#include <immintrin.h>

#define MM256_SET_M128I(a, b) _mm256_insertf128_si256(_mm256_castsi128_si256(b), (a), 1)

/* Signed i8 × i8 → i16 via maddubs (copied from ggml-quants.c) */
static inline __m256i mul_add_epi8(const __m256i x, const __m256i y) {
    const __m256i ax = _mm256_sign_epi8(x, x);
    const __m256i sy = _mm256_sign_epi8(y, x);
    return _mm256_maddubs_epi16(ax, sy);
}

/* Horizontal sum of 8 floats in __m256 (copied from ggml-quants.c) */
static inline float hsum_float_8(const __m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}
#endif

/* ── Lloyd-Max codebooks for N(0,1), normalized to [-1, 1] ─────────
 *
 * Computed via Lloyd-Max algorithm on N(0,1) with support [-6, 6].
 * Divided by max_level so outer levels = ±1.0.
 * Per-block scale d = max(|block|) maps data to this range.
 * ──────────────────────────────────────────────────────────────────── */

/* 2-bit: 4 levels, 3 boundaries */
static const float TQ2_LEVELS[4] = {
    -1.0000000f, -0.2997714f, +0.2997714f, +1.0000000f,
};

static const float TQ2_BOUNDARIES[3] = {
    -0.6498857f, +0.0000000f, +0.6498857f,
};

static const float TQ3_LEVELS[8] = {
    -1.0000000f, -0.6245203f, -0.3513239f, -0.1138989f,
    +0.1138989f, +0.3513239f, +0.6245203f, +1.0000000f,
};

static const float TQ3_BOUNDARIES[7] = {
    -0.8122602f, -0.4879221f, -0.2326114f, +0.0000000f,
    +0.2326114f, +0.4879221f, +0.8122602f,
};

static const float TQ4_LEVELS[16] = {
    -1.0000000f, -0.7573038f, -0.5923403f, -0.4599576f,
    -0.3450764f, -0.2405254f, -0.1421261f, -0.0470277f,
    +0.0470277f, +0.1421261f, +0.2405254f, +0.3450764f,
    +0.4599576f, +0.5923403f, +0.7573038f, +1.0000000f,
};

static const float TQ4_BOUNDARIES[15] = {
    -0.8786519f, -0.6748221f, -0.5261490f, -0.4025170f,
    -0.2928009f, -0.1913257f, -0.0945769f, +0.0000000f,
    +0.0945769f, +0.1913257f, +0.2928009f, +0.4025170f,
    +0.5261490f, +0.6748221f, +0.8786519f,
};

/* Int8-scaled codebooks for SIMD table lookup (scale factor = 127)
 * round(LEVELS[i] * 127). Divide by 127 after integer dot product. */
static const int8_t TQ2_LEVELS_I8[16] = {
    -127, -38, +38, +127,
       0,   0,   0,    0,   0,   0,   0,   0,   0,   0,   0,   0,  /* padding for PSHUFB */
};
static const int8_t TQ3_LEVELS_I8[16] = {
    -127, -79, -45, -14, +14, +45, +79, +127,
       0,   0,   0,   0,   0,   0,   0,    0,  /* padding for PSHUFB */
};
static const int8_t TQ4_LEVELS_I8[16] = {
    -127, -96, -75, -58, -44, -31, -18, -6,
      +6, +18, +31, +44, +58, +75, +96, +127,
};

/* ── 2-bit packing: 4 values (2-bit each) → 1 byte ───────────────── */

static inline void pack_2bit_32(const uint8_t indices[32], uint8_t qs[8]) {
    for (int i = 0; i < 8; i++) {
        qs[i] = (uint8_t)(
            (indices[4*i+0] & 3)        |
            ((indices[4*i+1] & 3) << 2) |
            ((indices[4*i+2] & 3) << 4) |
            ((indices[4*i+3] & 3) << 6)
        );
    }
}

static inline void unpack_2bit_32(const uint8_t qs[8], uint8_t indices[32]) {
    for (int i = 0; i < 8; i++) {
        indices[4*i+0] =  qs[i]       & 3;
        indices[4*i+1] = (qs[i] >> 2) & 3;
        indices[4*i+2] = (qs[i] >> 4) & 3;
        indices[4*i+3] = (qs[i] >> 6) & 3;
    }
}

/* ── 3-bit packing: 8 values (3-bit each) → 3 bytes ───────────────── */

static inline void pack_3bit_group(const uint8_t idx[8], uint8_t out[3]) {
    out[0] = (uint8_t)(
        (idx[0] & 7)        |
        ((idx[1] & 7) << 3) |
        ((idx[2] & 3) << 6)
    );
    out[1] = (uint8_t)(
        ((idx[2] >> 2) & 1)  |
        ((idx[3] & 7) << 1)  |
        ((idx[4] & 7) << 4)  |
        ((idx[5] & 1) << 7)
    );
    out[2] = (uint8_t)(
        ((idx[5] >> 1) & 3)  |
        ((idx[6] & 7) << 2)  |
        ((idx[7] & 7) << 5)
    );
}

static inline void unpack_3bit_group(const uint8_t in[3], uint8_t idx[8]) {
    idx[0] =  in[0]       & 7;
    idx[1] = (in[0] >> 3) & 7;
    idx[2] = ((in[0] >> 6) & 3) | ((in[1] & 1) << 2);
    idx[3] = (in[1] >> 1) & 7;
    idx[4] = (in[1] >> 4) & 7;
    idx[5] = ((in[1] >> 7) & 1) | ((in[2] & 3) << 1);
    idx[6] = (in[2] >> 2) & 7;
    idx[7] = (in[2] >> 5) & 7;
}

/* ── Nearest-level lookup ──────────────────────────────────────────── */

static inline uint8_t find_nearest_tq2(float xn) {
    uint8_t idx = 0;
    for (int b = 0; b < 3; b++) {
        if (xn > TQ2_BOUNDARIES[b]) idx = (uint8_t)(b + 1);
    }
    return idx;
}

static inline uint8_t find_nearest_tq3(float xn) {
    uint8_t idx = 0;
    for (int b = 0; b < 7; b++) {
        if (xn > TQ3_BOUNDARIES[b]) idx = (uint8_t)(b + 1);
    }
    return idx;
}

static inline uint8_t find_nearest_tq4(float xn) {
    // Binary search: 4 comparisons instead of 15
    // TQ4_BOUNDARIES is sorted, symmetric around 0
    uint8_t idx;
    if (xn <= TQ4_BOUNDARIES[7]) {             // <= 0.0
        if (xn <= TQ4_BOUNDARIES[3]) {         // <= -0.4025
            if (xn <= TQ4_BOUNDARIES[1]) {     // <= -0.6748
                idx = (xn <= TQ4_BOUNDARIES[0]) ? 0 : 1;
            } else {
                idx = (xn <= TQ4_BOUNDARIES[2]) ? 2 : 3;
            }
        } else {
            if (xn <= TQ4_BOUNDARIES[5]) {     // <= -0.1913
                idx = (xn <= TQ4_BOUNDARIES[4]) ? 4 : 5;
            } else {
                idx = (xn <= TQ4_BOUNDARIES[6]) ? 6 : 7;
            }
        }
    } else {
        if (xn <= TQ4_BOUNDARIES[11]) {        // <= +0.4025
            if (xn <= TQ4_BOUNDARIES[9]) {     // <= +0.1913
                idx = (xn <= TQ4_BOUNDARIES[8]) ? 8 : 9;
            } else {
                idx = (xn <= TQ4_BOUNDARIES[10]) ? 10 : 11;
            }
        } else {
            if (xn <= TQ4_BOUNDARIES[13]) {    // <= +0.6748
                idx = (xn <= TQ4_BOUNDARIES[12]) ? 12 : 13;
            } else {
                idx = (xn <= TQ4_BOUNDARIES[14]) ? 14 : 15;
            }
        }
    }
    return idx;
}

/* ── LeanKV: block scale statistic (LEANKV_TQ_SCALE) ───────────────────
 *
 * The shipping scale is `d = max|block|`, an extreme-order statistic that a
 * single outlier inside a 32-element block can hijack — wasting the 4/8/16
 * level codebook on empty range.  This selector lets a robust statistic be
 * chosen instead.  It is READ-path neutral: `d` is still stored fp16 in the
 * block exactly as today, so there is NO format change and dequantization is
 * untouched.  Default (env unset) is `amax` = current behaviour, and the
 * dispatch is a single predicted branch on a cached int.
 *
 *   LEANKV_TQ_SCALE = amax | absmean | rms | mse_opt
 *
 * See docs/leankv-scale-scheme-study-2026-07.md for the constants below.
 * ──────────────────────────────────────────────────────────────────── */

enum { GGML_TQ_SCALE_AMAX = 0, GGML_TQ_SCALE_ABSMEAN, GGML_TQ_SCALE_RMS, GGML_TQ_SCALE_MSEOPT };

/* c constants for d = c * mean|x| and d = c * sqrt(mean x^2), indexed by bits
 * (2,3,4).  Selected by minimizing calib-half reconstruction SSE, cross-model. */
static const float TQ_ABSMEAN_C[5] = { 0.0f, 0.0f, 1.8f, 2.6f, 3.4f };
static const float TQ_RMS_C    [5] = { 0.0f, 0.0f, 1.4f, 2.0f, 2.6f };

static int ggml_tq_scale_mode(void) {
    static int mode = -1;
    if (mode < 0) {
        const char * s = getenv("LEANKV_TQ_SCALE");
        int m = GGML_TQ_SCALE_AMAX;
        if (s) {
            if      (strcmp(s, "absmean") == 0) m = GGML_TQ_SCALE_ABSMEAN;
            else if (strcmp(s, "rms")     == 0) m = GGML_TQ_SCALE_RMS;
            else if (strcmp(s, "mse_opt") == 0) m = GGML_TQ_SCALE_MSEOPT;
        }
        mode = m;  /* benign race: idempotent */
    }
    return mode;
}

/* Nearest level for the given bit-width (shared by the mse_opt search). */
static inline uint8_t tq_find_nearest(float xn, int bits) {
    if (bits == 2) return find_nearest_tq2(xn);
    if (bits == 3) return find_nearest_tq3(xn);
    return find_nearest_tq4(xn);
}

static inline const float * tq_levels(int bits) {
    if (bits == 2) return TQ2_LEVELS;
    if (bits == 3) return TQ3_LEVELS;
    return TQ4_LEVELS;
}

/* Per-block grid search for the MSE-optimal scale: d = t * amax, 32 candidate
 * t in [0.24, 1.01].  This is the per-block optimum for a fixed level set —
 * the upper bound any fixed scale statistic can approach. */
static float tq_scale_mse_opt(const float * block, int n, int bits, float amax) {
    const float * L = tq_levels(bits);
    float best_d = amax, best_sse = INFINITY;
    for (int g = 0; g < 32; g++) {
        const float t  = 0.24f + 0.0248387097f * (float)g;   /* -> 1.01 */
        const float d  = amax * t;
        const float id = 1.0f / d;
        float sse = 0.0f;
        for (int j = 0; j < n; j++) {
            float xn = block[j] * id;
            if (xn < -1.0f) xn = -1.0f;
            if (xn >  1.0f) xn =  1.0f;
            const float e = block[j] - L[tq_find_nearest(xn, bits)] * d;
            sse += e * e;
        }
        if (sse < best_sse) { best_sse = sse; best_d = d; }
    }
    return best_d;
}

/* The one place the block scale is decided.  amax is passed in because every
 * caller already computed it (and mse_opt searches relative to it). */
static inline float tq_block_scale(const float * block, int n, int bits, float amax) {
    switch (ggml_tq_scale_mode()) {
        case GGML_TQ_SCALE_ABSMEAN: {
            float s = 0.0f;
            for (int j = 0; j < n; j++) s += fabsf(block[j]);
            return TQ_ABSMEAN_C[bits] * s / (float)n;
        }
        case GGML_TQ_SCALE_RMS: {
            float s = 0.0f;
            for (int j = 0; j < n; j++) s += block[j] * block[j];
            return TQ_RMS_C[bits] * sqrtf(s / (float)n);
        }
        case GGML_TQ_SCALE_MSEOPT:
            return tq_scale_mse_opt(block, n, bits, amax);
        default:
            return amax;
    }
}

/* ── TQ2_0 ─────────────────────────────────────────────────────────── */

void quantize_row_tq2_0_ref(const float * GGML_RESTRICT x, block_tq2_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TQ2 == 0);
    const int nb = (int)(k / QK_TQ2);

    for (int i = 0; i < nb; i++) {
        const float * block = x + i * QK_TQ2;

        /* Find max absolute value for block scale */
        float amax = 0.0f;
        for (int j = 0; j < QK_TQ2; j++) {
            float v = fabsf(block[j]);
            if (v > amax) amax = v;
        }

        const float d = (amax < 1e-10f) ? amax : tq_block_scale(block, QK_TQ2, 2, amax);
        y[i].d = GGML_FP32_TO_FP16(d);

        if (amax < 1e-10f) {
            memset(y[i].qs, 0, sizeof(y[i].qs));
            continue;
        }

        const float id = 1.0f / d;
        uint8_t indices[QK_TQ2];
        for (int j = 0; j < QK_TQ2; j++) {
            float xn = block[j] * id;  /* normalized to [-1, 1] */
            indices[j] = find_nearest_tq2(xn);
        }

        pack_2bit_32(indices, y[i].qs);
    }
}

void dequantize_row_tq2_0(const block_tq2_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TQ2 == 0);
    const int nb = (int)(k / QK_TQ2);

    /* LeanKV 7a Stage 4a: read from runtime LUT (fitted or Gaussian default). */
    const float * TQ2_LEVELS = ggml_tq_get_runtime_levels_f(2);

    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);
        uint8_t indices[QK_TQ2];
        unpack_2bit_32(x[i].qs, indices);
        for (int j = 0; j < QK_TQ2; j++) {
            y[i * QK_TQ2 + j] = TQ2_LEVELS[indices[j]] * d;
        }
    }
}

void quantize_row_tq2_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_tq2_0_ref(x, (block_tq2_0 *)y, k);
}

/* ── TQ3_0 ─────────────────────────────────────────────────────────── */

/* Least-squares optimal scale for given index assignment:
 *   d_opt = sum(x[j] * L[idx[j]]) / sum(L[idx[j]]^2)
 * Minimizes block MSE.  */
static inline float tq3_optimal_scale(const float * block, const uint8_t * indices, int n) {
    float num = 0.0f, den = 0.0f;
    for (int j = 0; j < n; j++) {
        float lev = TQ3_LEVELS[indices[j]];
        num += block[j] * lev;
        den += lev * lev;
    }
    return (den > 0.0f) ? num / den : 0.0f;
}

/* Block MSE for given indices + scale */
static inline float tq3_block_mse(const float * block, const uint8_t * indices, float d, int n) {
    float mse = 0.0f;
    for (int j = 0; j < n; j++) {
        float err = block[j] - TQ3_LEVELS[indices[j]] * d;
        mse += err * err;
    }
    return mse;
}

void quantize_row_tq3_0_ref(const float * GGML_RESTRICT x, block_tq3_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TQ3 == 0);
    const int nb = (int)(k / QK_TQ3);

    for (int i = 0; i < nb; i++) {
        const float * block = x + i * QK_TQ3;

        /* Find max absolute value for initial scale */
        float amax = 0.0f;
        for (int j = 0; j < QK_TQ3; j++) {
            float v = fabsf(block[j]);
            if (v > amax) amax = v;
        }

        if (amax == 0.0f) {
            y[i].d = ggml_fp32_to_fp16(0.0f);
            memset(y[i].qs, 0, 12);
            continue;
        }

        /* Initial nearest-level assignment.  Shipping behaviour uses the max|x|
         * scale here; LEANKV_TQ_SCALE swaps in a different initial statistic
         * and the least-squares + coordinate-descent refinement below then runs
         * unchanged on top of that (better init -> better local optimum). */
        const float d0 = tq_block_scale(block, QK_TQ3, 3, amax);
        const float id = 1.0f / ((d0 > 0.0f) ? d0 : amax);
        uint8_t indices[QK_TQ3];
        for (int j = 0; j < QK_TQ3; j++) {
            float xn = block[j] * id;
            if (xn < -1.0f) xn = -1.0f;
            if (xn >  1.0f) xn =  1.0f;
            indices[j] = find_nearest_tq3(xn);
        }

        /* Compute least-squares optimal scale */
        float d = tq3_optimal_scale(block, indices, QK_TQ3);

        /* Coordinate descent: try adjacent levels, keep if MSE improves.
         * 2 passes suffice (converges fast per Python prototype). */
        float best_mse = tq3_block_mse(block, indices, d, QK_TQ3);
        for (int pass = 0; pass < 2; pass++) {
            int improved = 0;
            for (int j = 0; j < QK_TQ3; j++) {
                uint8_t orig = indices[j];
                /* Try index - 1 */
                if (orig > 0) {
                    indices[j] = orig - 1;
                    float nd = tq3_optimal_scale(block, indices, QK_TQ3);
                    float nm = tq3_block_mse(block, indices, nd, QK_TQ3);
                    if (nm < best_mse) {
                        best_mse = nm;
                        d = nd;
                        improved = 1;
                        continue;
                    }
                    indices[j] = orig;
                }
                /* Try index + 1 */
                if (orig < 7) {
                    indices[j] = orig + 1;
                    float nd = tq3_optimal_scale(block, indices, QK_TQ3);
                    float nm = tq3_block_mse(block, indices, nd, QK_TQ3);
                    if (nm < best_mse) {
                        best_mse = nm;
                        d = nd;
                        improved = 1;
                        continue;
                    }
                    indices[j] = orig;
                }
            }
            if (!improved) break;
        }

        y[i].d = ggml_fp32_to_fp16(d);

        /* Pack 32 × 3-bit indices into 12 bytes (4 groups of 8) */
        for (int g = 0; g < 4; g++) {
            pack_3bit_group(indices + g * 8, y[i].qs + g * 3);
        }
    }
}

void dequantize_row_tq3_0(const block_tq3_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TQ3 == 0);
    const int nb = (int)(k / QK_TQ3);

    /* LeanKV 7a Stage 4a: read from runtime LUT. */
    const float * TQ3_LEVELS = ggml_tq_get_runtime_levels_f(3);

    for (int i = 0; i < nb; i++) {
        const float d = ggml_fp16_to_fp32(x[i].d);

        uint8_t indices[QK_TQ3];
        for (int g = 0; g < 4; g++) {
            unpack_3bit_group(x[i].qs + g * 3, indices + g * 8);
        }

        for (int j = 0; j < QK_TQ3; j++) {
            y[i * QK_TQ3 + j] = TQ3_LEVELS[indices[j]] * d;
        }
    }
}

void quantize_row_tq3_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_tq3_0_ref(x, (block_tq3_0 *)y, k);
}

/* ── TQ4_0 ─────────────────────────────────────────────────────────── */

void quantize_row_tq4_0_ref(const float * GGML_RESTRICT x, block_tq4_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TQ4 == 0);
    const int nb = (int)(k / QK_TQ4);

    for (int i = 0; i < nb; i++) {
        const float * block = x + i * QK_TQ4;

        float amax = 0.0f;
        for (int j = 0; j < QK_TQ4; j++) {
            float v = fabsf(block[j]);
            if (v > amax) amax = v;
        }

        const float d  = (amax <= 0.0f) ? amax : tq_block_scale(block, QK_TQ4, 4, amax);
        const float id = (d > 0.0f) ? 1.0f / d : 0.0f;
        y[i].d = ggml_fp32_to_fp16(d);

        for (int j = 0; j < QK_TQ4 / 2; j++) {
            float xn0 = block[j]              * id;
            float xn1 = block[j + QK_TQ4 / 2] * id;
            if (xn0 < -1.0f) xn0 = -1.0f;
            if (xn0 >  1.0f) xn0 =  1.0f;
            if (xn1 < -1.0f) xn1 = -1.0f;
            if (xn1 >  1.0f) xn1 =  1.0f;

            uint8_t i0 = find_nearest_tq4(xn0);
            uint8_t i1 = find_nearest_tq4(xn1);
            y[i].qs[j] = i0 | (i1 << 4);
        }
    }
}

void dequantize_row_tq4_0(const block_tq4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TQ4 == 0);
    const int nb = (int)(k / QK_TQ4);

    /* LeanKV 7a Stage 4a: read from runtime LUT. */
    const float * TQ4_LEVELS = ggml_tq_get_runtime_levels_f(4);

    for (int i = 0; i < nb; i++) {
        const float d = ggml_fp16_to_fp32(x[i].d);

        for (int j = 0; j < QK_TQ4 / 2; j++) {
            const uint8_t i0 = x[i].qs[j] & 0x0F;
            const uint8_t i1 = x[i].qs[j] >> 4;

            y[i * QK_TQ4 + j]              = TQ4_LEVELS[i0] * d;
            y[i * QK_TQ4 + j + QK_TQ4 / 2] = TQ4_LEVELS[i1] * d;
        }
    }
}

void quantize_row_tq4_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_tq4_0_ref(x, (block_tq4_0 *)y, k);
}

/* ── vec_dot: TQ3_0 · Q8_0 and TQ4_0 · Q8_0 ──────────────────────── */

void ggml_vec_dot_tq3_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs,
        const void * GGML_RESTRICT vx, size_t bx,
        const void * GGML_RESTRICT vy, size_t by, int nrc) {
    const int qk = QK_TQ3;
    const int nb = n / qk;

    assert(n % qk == 0);
    assert(nrc == 1);
    (void)bs; (void)bx; (void)by; (void)nrc;

    const block_tq3_0 * GGML_RESTRICT x = (const block_tq3_0 *) vx;
    const block_q8_0  * GGML_RESTRICT y = (const block_q8_0  *) vy;

    /* LeanKV 7a Stage 4a: shadow the file-scope defaults with runtime LUTs. */
    const int8_t * TQ3_LEVELS_I8 = ggml_tq_get_runtime_levels_i8(3);
    const float  * TQ3_LEVELS    = ggml_tq_get_runtime_levels_f (3);
    (void)TQ3_LEVELS_I8; (void)TQ3_LEVELS;

    float sumf = 0.0f;
    int ib = 0;

#if defined(__ARM_NEON)
    const int8x16_t values = vld1q_s8(TQ3_LEVELS_I8);
    int32x4_t prod_1, prod_2;

    for (; ib + 1 < nb; ib += 2) {
        /* Unpack 3-bit indices to bytes */
        uint8_t indices_0[32], indices_1[32];
        for (int g = 0; g < 4; g++) {
            unpack_3bit_group(x[ib + 0].qs + g * 3, indices_0 + g * 8);
            unpack_3bit_group(x[ib + 1].qs + g * 3, indices_1 + g * 8);
        }

        /* Table lookup */
        const int8x16_t q3b_0_lo = ggml_vqtbl1q_s8(values, vld1q_u8(indices_0));
        const int8x16_t q3b_0_hi = ggml_vqtbl1q_s8(values, vld1q_u8(indices_0 + 16));
        const int8x16_t q3b_1_lo = ggml_vqtbl1q_s8(values, vld1q_u8(indices_1));
        const int8x16_t q3b_1_hi = ggml_vqtbl1q_s8(values, vld1q_u8(indices_1 + 16));

        /* Load Q8 data */
        const int8x16_t q8b_0_lo = vld1q_s8(y[ib + 0].qs);
        const int8x16_t q8b_0_hi = vld1q_s8(y[ib + 0].qs + 16);
        const int8x16_t q8b_1_lo = vld1q_s8(y[ib + 1].qs);
        const int8x16_t q8b_1_hi = vld1q_s8(y[ib + 1].qs + 16);

        /* Dot products */
        prod_1 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), q3b_0_lo, q8b_0_lo), q3b_0_hi, q8b_0_hi);
        prod_2 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), q3b_1_lo, q8b_1_lo), q3b_1_hi, q8b_1_hi);

        sumf +=
            GGML_FP16_TO_FP32(x[ib+0].d) * GGML_FP16_TO_FP32(y[ib+0].d) / 127.0f * (float)vaddvq_s32(prod_1) +
            GGML_FP16_TO_FP32(x[ib+1].d) * GGML_FP16_TO_FP32(y[ib+1].d) / 127.0f * (float)vaddvq_s32(prod_2);
    }
#elif defined(__AVX2__)
    const __m128i values128 = _mm_loadu_si128((const __m128i *)TQ3_LEVELS_I8);
    const __m256i mone = _mm256_set1_epi16(1);
    __m256 accum1 = _mm256_setzero_ps();
    __m256 accum2 = _mm256_setzero_ps();

    for (; ib + 1 < nb; ib += 2) {
        /* Unpack 3-bit indices to bytes */
        uint8_t indices_0[32], indices_1[32];
        for (int g = 0; g < 4; g++) {
            unpack_3bit_group(x[ib + 0].qs + g * 3, indices_0 + g * 8);
            unpack_3bit_group(x[ib + 1].qs + g * 3, indices_1 + g * 8);
        }

        /* Table lookup via PSHUFB */
        const __m256i q3b_1 = MM256_SET_M128I(
            _mm_shuffle_epi8(values128, _mm_loadu_si128((const __m128i *)(indices_0 + 16))),
            _mm_shuffle_epi8(values128, _mm_loadu_si128((const __m128i *)indices_0)));
        const __m256i q3b_2 = MM256_SET_M128I(
            _mm_shuffle_epi8(values128, _mm_loadu_si128((const __m128i *)(indices_1 + 16))),
            _mm_shuffle_epi8(values128, _mm_loadu_si128((const __m128i *)indices_1)));

        /* Load Q8 data */
        const __m256i q8b_1 = _mm256_loadu_si256((const __m256i *)y[ib + 0].qs);
        const __m256i q8b_2 = _mm256_loadu_si256((const __m256i *)y[ib + 1].qs);

        /* Integer dot product */
        const __m256i p16_1 = mul_add_epi8(q3b_1, q8b_1);
        const __m256i p16_2 = mul_add_epi8(q3b_2, q8b_2);
        const __m256i p_1 = _mm256_madd_epi16(p16_1, mone);
        const __m256i p_2 = _mm256_madd_epi16(p16_2, mone);

        accum1 = _mm256_fmadd_ps(
            _mm256_set1_ps(GGML_FP16_TO_FP32(y[ib+0].d) * GGML_FP16_TO_FP32(x[ib+0].d) / 127.0f),
            _mm256_cvtepi32_ps(p_1), accum1);
        accum2 = _mm256_fmadd_ps(
            _mm256_set1_ps(GGML_FP16_TO_FP32(y[ib+1].d) * GGML_FP16_TO_FP32(x[ib+1].d) / 127.0f),
            _mm256_cvtepi32_ps(p_2), accum2);
    }
    sumf = hsum_float_8(_mm256_add_ps(accum1, accum2));
#endif

    /* Scalar fallback for remaining blocks */
    for (; ib < nb; ib++) {
        const float d_tq = ggml_fp16_to_fp32(x[ib].d);
        const float d_q8 = ggml_fp16_to_fp32(y[ib].d);

        uint8_t indices[QK_TQ3];
        for (int g = 0; g < 4; g++) {
            unpack_3bit_group(x[ib].qs + g * 3, indices + g * 8);
        }

        float block_sum = 0.0f;
        for (int j = 0; j < qk; j++) {
            block_sum += TQ3_LEVELS[indices[j]] * (float)y[ib].qs[j];
        }
        sumf += d_tq * d_q8 * block_sum;
    }

    *s = sumf;
}

void ggml_vec_dot_tq4_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs,
        const void * GGML_RESTRICT vx, size_t bx,
        const void * GGML_RESTRICT vy, size_t by, int nrc) {
    const int qk = QK_TQ4;
    const int nb = n / qk;

    assert(n % qk == 0);
    assert(nrc == 1);
    (void)bs; (void)bx; (void)by; (void)nrc;

    const block_tq4_0 * GGML_RESTRICT x = (const block_tq4_0 *) vx;
    const block_q8_0  * GGML_RESTRICT y = (const block_q8_0  *) vy;

    /* LeanKV 7a Stage 4a: shadow the file-scope defaults with runtime LUTs. */
    const int8_t * TQ4_LEVELS_I8 = ggml_tq_get_runtime_levels_i8(4);
    const float  * TQ4_LEVELS    = ggml_tq_get_runtime_levels_f (4);
    (void)TQ4_LEVELS_I8; (void)TQ4_LEVELS;

    float sumf = 0.0f;
    int ib = 0;

#if defined(__ARM_NEON)
    const int8x16_t values = vld1q_s8(TQ4_LEVELS_I8);
    const uint8x16_t m4b = vdupq_n_u8(0x0f);
    uint8x16x2_t q4bits;
    int8x16x4_t q4b;
    int8x16x4_t q8b;
    int32x4_t prod_1, prod_2;

    for (; ib + 1 < nb; ib += 2) {
        q4bits.val[0] = vld1q_u8(x[ib + 0].qs);
        q4bits.val[1] = vld1q_u8(x[ib + 1].qs);
        q8b.val[0]    = vld1q_s8(y[ib + 0].qs);
        q8b.val[1]    = vld1q_s8(y[ib + 0].qs + 16);
        q8b.val[2]    = vld1q_s8(y[ib + 1].qs);
        q8b.val[3]    = vld1q_s8(y[ib + 1].qs + 16);
        q4b.val[0] = ggml_vqtbl1q_s8(values, vandq_u8  (q4bits.val[0], m4b));
        q4b.val[1] = ggml_vqtbl1q_s8(values, vshrq_n_u8(q4bits.val[0], 4));
        q4b.val[2] = ggml_vqtbl1q_s8(values, vandq_u8  (q4bits.val[1], m4b));
        q4b.val[3] = ggml_vqtbl1q_s8(values, vshrq_n_u8(q4bits.val[1], 4));
        prod_1 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), q4b.val[0], q8b.val[0]), q4b.val[1], q8b.val[1]);
        prod_2 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), q4b.val[2], q8b.val[2]), q4b.val[3], q8b.val[3]);
        sumf +=
            GGML_FP16_TO_FP32(x[ib+0].d) * GGML_FP16_TO_FP32(y[ib+0].d) / 127.0f * (float)vaddvq_s32(prod_1) +
            GGML_FP16_TO_FP32(x[ib+1].d) * GGML_FP16_TO_FP32(y[ib+1].d) / 127.0f * (float)vaddvq_s32(prod_2);
    }
#elif defined(__AVX2__)
    const __m128i values128 = _mm_loadu_si128((const __m128i *)TQ4_LEVELS_I8);
    const __m128i m4b  = _mm_set1_epi8(0x0f);
    const __m256i mone = _mm256_set1_epi16(1);
    __m256 accum1 = _mm256_setzero_ps();
    __m256 accum2 = _mm256_setzero_ps();

    for (; ib + 1 < nb; ib += 2) {
        const __m128i q4bits_1 = _mm_loadu_si128((const __m128i *)x[ib + 0].qs);
        const __m128i q4bits_2 = _mm_loadu_si128((const __m128i *)x[ib + 1].qs);
        const __m256i q8b_1 = _mm256_loadu_si256((const __m256i *)y[ib + 0].qs);
        const __m256i q8b_2 = _mm256_loadu_si256((const __m256i *)y[ib + 1].qs);
        const __m256i q4b_1 = MM256_SET_M128I(
            _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_1, 4), m4b)),
            _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_1, m4b)));
        const __m256i q4b_2 = MM256_SET_M128I(
            _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_2, 4), m4b)),
            _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_2, m4b)));
        const __m256i p16_1 = mul_add_epi8(q4b_1, q8b_1);
        const __m256i p16_2 = mul_add_epi8(q4b_2, q8b_2);
        const __m256i p_1 = _mm256_madd_epi16(p16_1, mone);
        const __m256i p_2 = _mm256_madd_epi16(p16_2, mone);
        accum1 = _mm256_fmadd_ps(
            _mm256_set1_ps(GGML_FP16_TO_FP32(y[ib+0].d) * GGML_FP16_TO_FP32(x[ib+0].d) / 127.0f),
            _mm256_cvtepi32_ps(p_1), accum1);
        accum2 = _mm256_fmadd_ps(
            _mm256_set1_ps(GGML_FP16_TO_FP32(y[ib+1].d) * GGML_FP16_TO_FP32(x[ib+1].d) / 127.0f),
            _mm256_cvtepi32_ps(p_2), accum2);
    }
    sumf = hsum_float_8(_mm256_add_ps(accum1, accum2));
#endif

    /* Scalar fallback for remaining blocks */
    for (; ib < nb; ib++) {
        const float d_tq = ggml_fp16_to_fp32(x[ib].d);
        const float d_q8 = ggml_fp16_to_fp32(y[ib].d);

        float block_sum = 0.0f;
        for (int j = 0; j < qk / 2; j++) {
            const uint8_t i0 = x[ib].qs[j] & 0x0F;
            const uint8_t i1 = x[ib].qs[j] >> 4;
            block_sum += TQ4_LEVELS[i0] * (float)y[ib].qs[j];
            block_sum += TQ4_LEVELS[i1] * (float)y[ib].qs[j + qk / 2];
        }
        sumf += d_tq * d_q8 * block_sum;
    }

    *s = sumf;
}

/* ── vec_dot: TQ2_0 · Q8_0 ───────────────────────────────────────── */

void ggml_vec_dot_tq2_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs,
        const void * GGML_RESTRICT vx, size_t bx,
        const void * GGML_RESTRICT vy, size_t by, int nrc) {
    const int qk = QK_TQ2;
    const int nb = n / qk;

    assert(n % qk == 0);
    assert(nrc == 1);
    (void)bs; (void)bx; (void)by; (void)nrc;

    const block_tq2_0 * GGML_RESTRICT x = (const block_tq2_0 *) vx;
    const block_q8_0  * GGML_RESTRICT y = (const block_q8_0  *) vy;

    /* LeanKV 7a Stage 4a: shadow the file-scope defaults with runtime LUTs. */
    const int8_t * TQ2_LEVELS_I8 = ggml_tq_get_runtime_levels_i8(2);
    const float  * TQ2_LEVELS    = ggml_tq_get_runtime_levels_f (2);
    (void)TQ2_LEVELS_I8; (void)TQ2_LEVELS;

    float sumf = 0.0f;
    int ib = 0;

#if defined(__ARM_NEON)
    const int8x16_t values = vld1q_s8(TQ2_LEVELS_I8);
    const uint8x16_t m2b = vdupq_n_u8(0x03);
    int32x4_t prod_1, prod_2;

    for (; ib + 1 < nb; ib += 2) {
        /* Unpack 2-bit indices to bytes for two blocks */
        uint8_t indices_0[32], indices_1[32];
        unpack_2bit_32(x[ib + 0].qs, indices_0);
        unpack_2bit_32(x[ib + 1].qs, indices_1);

        /* Table lookup */
        const int8x16_t q2b_0_lo = ggml_vqtbl1q_s8(values, vld1q_u8(indices_0));
        const int8x16_t q2b_0_hi = ggml_vqtbl1q_s8(values, vld1q_u8(indices_0 + 16));
        const int8x16_t q2b_1_lo = ggml_vqtbl1q_s8(values, vld1q_u8(indices_1));
        const int8x16_t q2b_1_hi = ggml_vqtbl1q_s8(values, vld1q_u8(indices_1 + 16));

        const int8x16_t q8b_0_lo = vld1q_s8(y[ib + 0].qs);
        const int8x16_t q8b_0_hi = vld1q_s8(y[ib + 0].qs + 16);
        const int8x16_t q8b_1_lo = vld1q_s8(y[ib + 1].qs);
        const int8x16_t q8b_1_hi = vld1q_s8(y[ib + 1].qs + 16);

        prod_1 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), q2b_0_lo, q8b_0_lo), q2b_0_hi, q8b_0_hi);
        prod_2 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), q2b_1_lo, q8b_1_lo), q2b_1_hi, q8b_1_hi);
        sumf +=
            GGML_FP16_TO_FP32(x[ib+0].d) * GGML_FP16_TO_FP32(y[ib+0].d) / 127.0f * (float)vaddvq_s32(prod_1) +
            GGML_FP16_TO_FP32(x[ib+1].d) * GGML_FP16_TO_FP32(y[ib+1].d) / 127.0f * (float)vaddvq_s32(prod_2);
    }
#elif defined(__AVX2__)
    const __m128i values128 = _mm_loadu_si128((const __m128i *)TQ2_LEVELS_I8);
    const __m256i values256 = MM256_SET_M128I(values128, values128);
    const __m256i mone = _mm256_set1_epi16(1);
    __m256 accum1 = _mm256_setzero_ps();
    __m256 accum2 = _mm256_setzero_ps();

    for (; ib + 1 < nb; ib += 2) {
        /* Unpack 2-bit to 8-bit indices then PSHUFB lookup */
        uint8_t indices_0[32], indices_1[32];
        unpack_2bit_32(x[ib + 0].qs, indices_0);
        unpack_2bit_32(x[ib + 1].qs, indices_1);

        __m256i idx256_0 = _mm256_loadu_si256((const __m256i *)indices_0);
        __m256i idx256_1 = _mm256_loadu_si256((const __m256i *)indices_1);
        __m256i q2b_0 = _mm256_shuffle_epi8(values256, idx256_0);
        __m256i q2b_1 = _mm256_shuffle_epi8(values256, idx256_1);

        const __m256i q8b_0 = _mm256_loadu_si256((const __m256i *)y[ib + 0].qs);
        const __m256i q8b_1 = _mm256_loadu_si256((const __m256i *)y[ib + 1].qs);

        const __m256i p16_0 = mul_add_epi8(q2b_0, q8b_0);
        const __m256i p16_1 = mul_add_epi8(q2b_1, q8b_1);
        const __m256i p_0 = _mm256_madd_epi16(p16_0, mone);
        const __m256i p_1 = _mm256_madd_epi16(p16_1, mone);

        accum1 = _mm256_fmadd_ps(
            _mm256_set1_ps(GGML_FP16_TO_FP32(y[ib+0].d) * GGML_FP16_TO_FP32(x[ib+0].d) / 127.0f),
            _mm256_cvtepi32_ps(p_0), accum1);
        accum2 = _mm256_fmadd_ps(
            _mm256_set1_ps(GGML_FP16_TO_FP32(y[ib+1].d) * GGML_FP16_TO_FP32(x[ib+1].d) / 127.0f),
            _mm256_cvtepi32_ps(p_1), accum2);
    }
    sumf = hsum_float_8(_mm256_add_ps(accum1, accum2));
#endif

    /* Scalar fallback for remaining blocks */
    for (; ib < nb; ib++) {
        const float d_tq = GGML_FP16_TO_FP32(x[ib].d);
        const float d_q8 = GGML_FP16_TO_FP32(y[ib].d);

        uint8_t indices[QK_TQ2];
        unpack_2bit_32(x[ib].qs, indices);

        float block_sum = 0.0f;
        for (int j = 0; j < qk; j++) {
            block_sum += TQ2_LEVELS[indices[j]] * (float)y[ib].qs[j];
        }
        sumf += d_tq * d_q8 * block_sum;
    }

    *s = sumf;
}

/* ══════════════════════════════════════════════════════════════════════
 * TQ2_1: Mixed-precision TQ3 (outlier) + TQ2 (normal), 2.75 bits/elem
 * Block size: 128 elements (32 outlier as TQ3, 96 normal as 3×TQ2)
 * ══════════════════════════════════════════════════════════════════════ */

void dequantize_row_tq2_1(const block_tq2_1 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TQ2_1 == 0);
    const int nb = (int)(k / QK_TQ2_1);

    /* LeanKV 7a Stage 4a: runtime LUTs for the mixed-bit dequant. */
    const float * TQ2_LEVELS = ggml_tq_get_runtime_levels_f(2);
    const float * TQ3_LEVELS = ggml_tq_get_runtime_levels_f(3);

    for (int i = 0; i < nb; i++) {
        float * out = y + i * QK_TQ2_1;

        /* Outlier region: 32 elements, TQ3 encoding */
        {
            const float d = GGML_FP16_TO_FP32(x[i].d_out);
            uint8_t indices[32];
            unpack_3bit_group(x[i].qs_out +  0, indices +  0);
            unpack_3bit_group(x[i].qs_out +  3, indices +  8);
            unpack_3bit_group(x[i].qs_out +  6, indices + 16);
            unpack_3bit_group(x[i].qs_out +  9, indices + 24);
            for (int j = 0; j < 32; j++) {
                out[j] = TQ3_LEVELS[indices[j]] * d;
            }
        }

        /* Normal region: 3 × TQ2 blocks (96 elements) */
        const ggml_half * d_ptrs[3] = { &x[i].d_n0, &x[i].d_n1, &x[i].d_n2 };
        const uint8_t * qs_ptrs[3] = { x[i].qs_n0, x[i].qs_n1, x[i].qs_n2 };

        for (int b = 0; b < 3; b++) {
            const float d = GGML_FP16_TO_FP32(*d_ptrs[b]);
            uint8_t indices[32];
            unpack_2bit_32(qs_ptrs[b], indices);
            for (int j = 0; j < 32; j++) {
                out[32 + b * 32 + j] = TQ2_LEVELS[indices[j]] * d;
            }
        }
    }
}

void quantize_row_tq2_1_ref(const float * GGML_RESTRICT x, block_tq2_1 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TQ2_1 == 0);
    const int nb = (int)(k / QK_TQ2_1);

    for (int i = 0; i < nb; i++) {
        const float * block = x + i * QK_TQ2_1;

        /* Quantize outlier region (elements 0-31) as TQ3 */
        {
            block_tq3_0 tmp;
            quantize_row_tq3_0_ref(block, &tmp, 32);
            y[i].d_out = tmp.d;
            memcpy(y[i].qs_out, tmp.qs, 12);
        }

        /* Quantize normal region (elements 32-127) as 3 × TQ2 blocks */
        {
            block_tq2_0 tmp[3];
            quantize_row_tq2_0_ref(block + 32, tmp, 96);
            y[i].d_n0 = tmp[0].d;
            memcpy(y[i].qs_n0, tmp[0].qs, 8);
            y[i].d_n1 = tmp[1].d;
            memcpy(y[i].qs_n1, tmp[1].qs, 8);
            y[i].d_n2 = tmp[2].d;
            memcpy(y[i].qs_n2, tmp[2].qs, 8);
        }
    }
}

void quantize_row_tq2_1(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_tq2_1_ref(x, (block_tq2_1 *)y, k);
}

void ggml_vec_dot_tq2_1_q8_0(int n, float * GGML_RESTRICT s, size_t bs,
        const void * GGML_RESTRICT vx, size_t bx,
        const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_TQ2_1 == 0);
    assert(nrc == 1);
    (void)bs; (void)bx; (void)by; (void)nrc;

    const block_tq2_1 * GGML_RESTRICT x2 = (const block_tq2_1 *) vx;
    const block_q8_0  * GGML_RESTRICT y8 = (const block_q8_0  *) vy;

    /* LeanKV 7a Stage 4a: runtime LUTs for the mixed-bit vec_dot. */
    const int8_t * TQ2_LEVELS_I8 = ggml_tq_get_runtime_levels_i8(2);
    const int8_t * TQ3_LEVELS_I8 = ggml_tq_get_runtime_levels_i8(3);
    const float  * TQ2_LEVELS    = ggml_tq_get_runtime_levels_f (2);
    const float  * TQ3_LEVELS    = ggml_tq_get_runtime_levels_f (3);
    (void)TQ2_LEVELS_I8; (void)TQ3_LEVELS_I8;
    (void)TQ2_LEVELS; (void)TQ3_LEVELS;

    const int nb = n / QK_TQ2_1;
    float sumf = 0.0f;
    int i = 0;

#if defined(__ARM_NEON)
    const int8x16_t tq3_values_neon = vld1q_s8(TQ3_LEVELS_I8);
    const int8x16_t tq2_values_neon = vld1q_s8(TQ2_LEVELS_I8);

    for (; i < nb; i++) {
        const block_q8_0 * yb = y8 + i * 4;
        int32x4_t sum_i = vdupq_n_s32(0);

        /* Outlier TQ3 region: 32 elements -> yb[0] */
        uint8_t idx3[32];
        unpack_3bit_group(x2[i].qs_out + 0, idx3 + 0);
        unpack_3bit_group(x2[i].qs_out + 3, idx3 + 8);
        unpack_3bit_group(x2[i].qs_out + 6, idx3 + 16);
        unpack_3bit_group(x2[i].qs_out + 9, idx3 + 24);

        const int8x16_t q3_lo = ggml_vqtbl1q_s8(tq3_values_neon, vld1q_u8(idx3));
        const int8x16_t q3_hi = ggml_vqtbl1q_s8(tq3_values_neon, vld1q_u8(idx3 + 16));
        const int8x16_t q8_0_lo = vld1q_s8(yb[0].qs);
        const int8x16_t q8_0_hi = vld1q_s8(yb[0].qs + 16);
        int32x4_t prod_tq3 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), q3_lo, q8_0_lo), q3_hi, q8_0_hi);
        sumf += GGML_FP16_TO_FP32(x2[i].d_out) * GGML_FP16_TO_FP32(yb[0].d) / 127.0f * (float)vaddvq_s32(prod_tq3);

        /* Normal TQ2 regions: 3 x 32 elements -> yb[1..3] */
        const ggml_half * d_ptrs[3] = { &x2[i].d_n0, &x2[i].d_n1, &x2[i].d_n2 };
        const uint8_t * qs_ptrs[3] = { x2[i].qs_n0, x2[i].qs_n1, x2[i].qs_n2 };

        for (int b = 0; b < 3; b++) {
            uint8_t idx2[32];
            unpack_2bit_32(qs_ptrs[b], idx2);

            const int8x16_t q2_lo = ggml_vqtbl1q_s8(tq2_values_neon, vld1q_u8(idx2));
            const int8x16_t q2_hi = ggml_vqtbl1q_s8(tq2_values_neon, vld1q_u8(idx2 + 16));
            const int8x16_t q8_b_lo = vld1q_s8(yb[1 + b].qs);
            const int8x16_t q8_b_hi = vld1q_s8(yb[1 + b].qs + 16);
            int32x4_t prod_tq2 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), q2_lo, q8_b_lo), q2_hi, q8_b_hi);
            sumf += GGML_FP16_TO_FP32(*d_ptrs[b]) * GGML_FP16_TO_FP32(yb[1 + b].d) / 127.0f * (float)vaddvq_s32(prod_tq2);
        }
    }
#elif defined(__AVX2__)
    const __m128i tq3_values128 = _mm_loadu_si128((const __m128i *)TQ3_LEVELS_I8);
    const __m128i tq2_values128 = _mm_loadu_si128((const __m128i *)TQ2_LEVELS_I8);
    const __m256i tq2_values256 = MM256_SET_M128I(tq2_values128, tq2_values128);
    const __m256i mone = _mm256_set1_epi16(1);

    for (; i < nb; i++) {
        const block_q8_0 * yb = y8 + i * 4;
        __m256 accum = _mm256_setzero_ps();

        /* Outlier TQ3 region: 32 elements -> yb[0] */
        uint8_t idx3[32];
        unpack_3bit_group(x2[i].qs_out + 0, idx3 + 0);
        unpack_3bit_group(x2[i].qs_out + 3, idx3 + 8);
        unpack_3bit_group(x2[i].qs_out + 6, idx3 + 16);
        unpack_3bit_group(x2[i].qs_out + 9, idx3 + 24);

        const __m256i q3b = MM256_SET_M128I(
            _mm_shuffle_epi8(tq3_values128, _mm_loadu_si128((const __m128i *)(idx3 + 16))),
            _mm_shuffle_epi8(tq3_values128, _mm_loadu_si128((const __m128i *)idx3)));
        const __m256i q8b_0 = _mm256_loadu_si256((const __m256i *)yb[0].qs);
        const __m256i p16_tq3 = mul_add_epi8(q3b, q8b_0);
        const __m256i p_tq3 = _mm256_madd_epi16(p16_tq3, mone);
        accum = _mm256_fmadd_ps(
            _mm256_set1_ps(GGML_FP16_TO_FP32(yb[0].d) * GGML_FP16_TO_FP32(x2[i].d_out) / 127.0f),
            _mm256_cvtepi32_ps(p_tq3), accum);

        /* Normal TQ2 regions: 3 x 32 elements -> yb[1..3] */
        const ggml_half * d_ptrs[3] = { &x2[i].d_n0, &x2[i].d_n1, &x2[i].d_n2 };
        const uint8_t * qs_ptrs[3] = { x2[i].qs_n0, x2[i].qs_n1, x2[i].qs_n2 };

        for (int b = 0; b < 3; b++) {
            uint8_t idx2[32];
            unpack_2bit_32(qs_ptrs[b], idx2);

            const __m256i idx256 = _mm256_loadu_si256((const __m256i *)idx2);
            const __m256i q2b = _mm256_shuffle_epi8(tq2_values256, idx256);
            const __m256i q8b_b = _mm256_loadu_si256((const __m256i *)yb[1 + b].qs);
            const __m256i p16_tq2 = mul_add_epi8(q2b, q8b_b);
            const __m256i p_tq2 = _mm256_madd_epi16(p16_tq2, mone);
            accum = _mm256_fmadd_ps(
                _mm256_set1_ps(GGML_FP16_TO_FP32(yb[1 + b].d) * GGML_FP16_TO_FP32(*d_ptrs[b]) / 127.0f),
                _mm256_cvtepi32_ps(p_tq2), accum);
        }
        sumf += hsum_float_8(accum);
    }
#endif

    /* Scalar fallback for any remaining blocks (both SIMD paths consume all i = 0..nb) */
    for (; i < nb; i++) {
        const block_q8_0 * yb = y8 + i * 4;

        /* Outlier TQ3 region: 32 elements -> 1 Q8_0 block */
        {
            const float d_tq = GGML_FP16_TO_FP32(x2[i].d_out);
            const float d_q8 = GGML_FP16_TO_FP32(yb[0].d);
            uint8_t indices[32];
            unpack_3bit_group(x2[i].qs_out +  0, indices +  0);
            unpack_3bit_group(x2[i].qs_out +  3, indices +  8);
            unpack_3bit_group(x2[i].qs_out +  6, indices + 16);
            unpack_3bit_group(x2[i].qs_out +  9, indices + 24);
            float block_sum = 0.0f;
            for (int j = 0; j < 32; j++) {
                block_sum += TQ3_LEVELS[indices[j]] * (float)yb[0].qs[j];
            }
            sumf += d_tq * d_q8 * block_sum;
        }

        /* Normal TQ2 regions: 3 x 32 elements -> 3 Q8_0 blocks */
        const ggml_half * d_ptrs[3] = { &x2[i].d_n0, &x2[i].d_n1, &x2[i].d_n2 };
        const uint8_t * qs_ptrs[3] = { x2[i].qs_n0, x2[i].qs_n1, x2[i].qs_n2 };

        for (int b = 0; b < 3; b++) {
            const float d_tq = GGML_FP16_TO_FP32(*d_ptrs[b]);
            const float d_q8 = GGML_FP16_TO_FP32(yb[1 + b].d);
            uint8_t indices[32];
            unpack_2bit_32(qs_ptrs[b], indices);
            float block_sum = 0.0f;
            for (int j = 0; j < 32; j++) {
                block_sum += TQ2_LEVELS[indices[j]] * (float)yb[1 + b].qs[j];
            }
            sumf += d_tq * d_q8 * block_sum;
        }
    }

    *s = sumf;
}
