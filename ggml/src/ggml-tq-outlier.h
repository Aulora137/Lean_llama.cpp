/**
 * ggml-tq-outlier.h — Outlier channel treatment for TurboQuant KV cache
 *
 * Mixed-precision quantization: outlier channels get more bits (e.g. TQ3),
 * normal channels get fewer bits (e.g. TQ2). This is a runtime policy that
 * operates on existing TQ block types — no new GGML types needed.
 *
 * Key insight: the same channels are outliers across ALL tokens in a layer
 * (structural property of the model). So detection happens once at model load,
 * and the permutation table is fixed per layer with zero runtime detection cost.
 *
 * Effective bit-widths (head_dim=128, 25% outlier fraction):
 *   TQ3+TQ2: 32 @ 3.5 + 96 @ 2.5 = 2.75 bits/elem
 *   TQ4+TQ3: 32 @ 4.5 + 96 @ 3.5 = 3.75 bits/elem
 *   TQ4+TQ2: 32 @ 4.5 + 96 @ 2.5 = 3.00 bits/elem
 *
 * Reference: TurboQuant Section 4.3 (outlier treatment)
 */

#ifndef GGML_TQ_OUTLIER_H
#define GGML_TQ_OUTLIER_H

#include "ggml-common.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Outlier precision tiers ──────────────────────────────────────── */

typedef enum {
    TQ_TIER_TQ2 = 0,   /* 2.5 bits/elem — for normal channels  */
    TQ_TIER_TQ3 = 1,   /* 3.5 bits/elem — for outlier channels  */
    TQ_TIER_TQ4 = 2,   /* 4.5 bits/elem — for outlier channels  */
} tq_tier_t;

/* ── Outlier configuration (per-layer, fixed at model load) ───────── */

#define TQ_OUTLIER_MAX_DIM 256   /* max head_dim we support */

typedef struct {
    int     head_dim;               /* dimension of each head             */
    int     n_outlier;              /* number of outlier channels          */
    int     n_normal;               /* number of normal channels           */
    float   outlier_frac;           /* fraction (0.0–1.0) of outliers     */
    tq_tier_t outlier_tier;         /* quantization tier for outliers     */
    tq_tier_t normal_tier;          /* quantization tier for normal       */

    /* Channel permutation: perm[i] = original channel index for position i.
     * Outlier channels are placed first [0..n_outlier-1],
     * normal channels follow [n_outlier..head_dim-1].
     * This makes both regions contiguous for SIMD-friendly quantization. */
    int     perm[TQ_OUTLIER_MAX_DIM];

    /* Inverse permutation: inv_perm[original_idx] = position in permuted order.
     * Used for unpermuting after dequantize. */
    int     inv_perm[TQ_OUTLIER_MAX_DIM];

    /* Per-channel variance (from calibration). Higher = outlier. */
    float   channel_var[TQ_OUTLIER_MAX_DIM];
} tq_outlier_config;

/* ── Outlier detection ────────────────────────────────────────────── */

/**
 * Identify outlier channels from calibration data.
 *
 * Computes per-channel variance across n_tokens of KV data,
 * flags the top outlier_frac as outliers, builds permutation tables.
 *
 * @param config        Output: filled config struct
 * @param calib_data    Calibration KV data [n_tokens × head_dim], row-major
 * @param n_tokens      Number of calibration tokens
 * @param head_dim      Dimension per head
 * @param outlier_frac  Fraction of channels to treat as outliers (e.g. 0.25)
 * @param outlier_tier  TQ tier for outlier channels (TQ_TIER_TQ3 or TQ_TIER_TQ4)
 * @param normal_tier   TQ tier for normal channels (TQ_TIER_TQ2 or TQ_TIER_TQ3)
 */
void tq_identify_outliers(
    tq_outlier_config * config,
    const float * calib_data,
    int n_tokens,
    int head_dim,
    float outlier_frac,
    tq_tier_t outlier_tier,
    tq_tier_t normal_tier
);

/**
 * Initialize config with uniform treatment (no outlier split).
 * All channels treated as normal tier. Permutation is identity.
 */
void tq_outlier_config_init_uniform(
    tq_outlier_config * config,
    int head_dim,
    tq_tier_t tier
);

/* ── Mixed-precision quantize/dequantize ──────────────────────────── */

/**
 * Compute the buffer sizes needed for mixed-precision quantization.
 *
 * @param config        Outlier config
 * @param outlier_size  Output: bytes needed for outlier buffer
 * @param normal_size   Output: bytes needed for normal buffer
 */
void tq_mixed_buffer_sizes(
    const tq_outlier_config * config,
    size_t * outlier_size,
    size_t * normal_size
);

/**
 * Compute effective bits per element for the mixed-precision config.
 */
float tq_mixed_effective_bpe(const tq_outlier_config * config);

/**
 * Mixed-precision quantize: permute channels, split, quantize each tier.
 *
 * Input x has head_dim elements (one head's worth of KV for one token).
 * Outlier channels are quantized with outlier_tier, normal with normal_tier.
 *
 * @param x             Input data [head_dim]
 * @param config        Outlier config (determines split + permutation)
 * @param outlier_buf   Output: quantized outlier channels (TQ3 or TQ4 blocks)
 * @param normal_buf    Output: quantized normal channels (TQ2 or TQ3 blocks)
 */
void tq_mixed_quantize(
    const float * x,
    const tq_outlier_config * config,
    void * outlier_buf,
    void * normal_buf
);

/**
 * Mixed-precision dequantize: dequantize both tiers, unpermute channels.
 *
 * @param outlier_buf   Quantized outlier channels
 * @param normal_buf    Quantized normal channels
 * @param config        Outlier config
 * @param y             Output: dequantized data [head_dim]
 */
void tq_mixed_dequantize(
    const void * outlier_buf,
    const void * normal_buf,
    const tq_outlier_config * config,
    float * y
);

/* ── Utility ──────────────────────────────────────────────────────── */

/**
 * Apply channel permutation: y[i] = x[perm[i]] for i in [0, n).
 */
void tq_permute_channels(const float * x, float * y, const int * perm, int n);

/**
 * Apply inverse permutation: y[i] = x[inv_perm[i]] for i in [0, n).
 * Reverses the effect of tq_permute_channels.
 */
void tq_unpermute_channels(const float * x, float * y, const int * inv_perm, int n);

#ifdef __cplusplus
}
#endif

#endif /* GGML_TQ_OUTLIER_H */
