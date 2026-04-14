/**
 * ggml-tq-outlier.c — Outlier channel treatment for TurboQuant KV cache
 *
 * Implements mixed-precision quantization: outlier channels (high-variance)
 * get more bits, normal channels get fewer. Uses existing TQ2/TQ3/TQ4 block
 * types with channel permutation to keep regions contiguous for SIMD.
 *
 * Reference: TurboQuant Section 4.3
 */

#include "ggml-tq-outlier.h"
#include "ggml-quants.h"

#include <assert.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

/* ── Helper: block size for a tier ────────────────────────────────── */

static inline int tier_block_size(tq_tier_t tier) {
    switch (tier) {
        case TQ_TIER_TQ2: return QK_TQ2;  /* 32 */
        case TQ_TIER_TQ3: return QK_TQ3;
        case TQ_TIER_TQ4: return QK_TQ4;
    }
    return 32;
}

static inline size_t tier_block_bytes(tq_tier_t tier) {
    switch (tier) {
        case TQ_TIER_TQ2: return sizeof(block_tq2_0);  /* 10 */
        case TQ_TIER_TQ3: return sizeof(block_tq3_0);  /* 14 */
        case TQ_TIER_TQ4: return sizeof(block_tq4_0);  /* 18 */
    }
    return sizeof(block_tq2_0);
}

static inline float tier_bpe(tq_tier_t tier) {
    switch (tier) {
        case TQ_TIER_TQ2: return 2.5f;
        case TQ_TIER_TQ3: return 3.5f;
        case TQ_TIER_TQ4: return 4.5f;
    }
    return 2.5f;
}

/* ── Quantize/dequantize dispatch by tier ─────────────────────────── */

static void tier_quantize(const float * x, void * buf, int n, tq_tier_t tier) {
    switch (tier) {
        case TQ_TIER_TQ2:
            quantize_row_tq2_0_ref(x, (block_tq2_0 *)buf, n);
            break;
        case TQ_TIER_TQ3:
            quantize_row_tq3_0_ref(x, (block_tq3_0 *)buf, n);
            break;
        case TQ_TIER_TQ4:
            quantize_row_tq4_0_ref(x, (block_tq4_0 *)buf, n);
            break;
    }
}

static void tier_dequantize(const void * buf, float * y, int n, tq_tier_t tier) {
    switch (tier) {
        case TQ_TIER_TQ2:
            dequantize_row_tq2_0((const block_tq2_0 *)buf, y, n);
            break;
        case TQ_TIER_TQ3:
            dequantize_row_tq3_0((const block_tq3_0 *)buf, y, n);
            break;
        case TQ_TIER_TQ4:
            dequantize_row_tq4_0((const block_tq4_0 *)buf, y, n);
            break;
    }
}

/* ── Sort helper: argsort by variance (descending) ────────────────── */

typedef struct {
    int   idx;
    float var;
} var_entry;

static int var_cmp_desc(const void * a, const void * b) {
    float va = ((const var_entry *)a)->var;
    float vb = ((const var_entry *)b)->var;
    if (va > vb) return -1;
    if (va < vb) return  1;
    return 0;
}

/* ── Outlier detection ────────────────────────────────────────────── */

void tq_identify_outliers(
    tq_outlier_config * config,
    const float * calib_data,
    int n_tokens,
    int head_dim,
    float outlier_frac,
    tq_tier_t outlier_tier,
    tq_tier_t normal_tier)
{
    assert(head_dim > 0 && head_dim <= TQ_OUTLIER_MAX_DIM);
    assert(n_tokens > 0);
    assert(outlier_frac >= 0.0f && outlier_frac <= 1.0f);

    config->head_dim = head_dim;
    config->outlier_frac = outlier_frac;
    config->outlier_tier = outlier_tier;
    config->normal_tier  = normal_tier;

    /* Step 1: Compute per-channel mean */
    float mean[TQ_OUTLIER_MAX_DIM];
    memset(mean, 0, sizeof(float) * (size_t)head_dim);
    for (int t = 0; t < n_tokens; t++) {
        const float * row = calib_data + t * head_dim;
        for (int d = 0; d < head_dim; d++) {
            mean[d] += row[d];
        }
    }
    for (int d = 0; d < head_dim; d++) {
        mean[d] /= (float)n_tokens;
    }

    /* Step 2: Compute per-channel variance */
    memset(config->channel_var, 0, sizeof(float) * (size_t)head_dim);
    for (int t = 0; t < n_tokens; t++) {
        const float * row = calib_data + t * head_dim;
        for (int d = 0; d < head_dim; d++) {
            float diff = row[d] - mean[d];
            config->channel_var[d] += diff * diff;
        }
    }
    for (int d = 0; d < head_dim; d++) {
        config->channel_var[d] /= (float)n_tokens;
    }

    /* Step 3: Argsort channels by variance (descending) */
    var_entry entries[TQ_OUTLIER_MAX_DIM];
    for (int d = 0; d < head_dim; d++) {
        entries[d].idx = d;
        entries[d].var = config->channel_var[d];
    }
    qsort(entries, (size_t)head_dim, sizeof(var_entry), var_cmp_desc);

    /* Step 4: Top outlier_frac channels are outliers.
     * Round n_outlier to nearest multiple of 32 (block size) for SIMD alignment. */
    int n_raw = (int)(outlier_frac * (float)head_dim + 0.5f);
    /* Round to nearest multiple of 32, but at least 32 if any requested */
    int n_outlier;
    if (n_raw == 0) {
        n_outlier = 0;
    } else {
        n_outlier = ((n_raw + 31) / 32) * 32;
        if (n_outlier > head_dim) n_outlier = head_dim;
    }
    int n_normal = head_dim - n_outlier;

    config->n_outlier = n_outlier;
    config->n_normal  = n_normal;

    /* Step 5: Build permutation — outliers first, then normal.
     * perm[i] = which original channel goes to position i. */
    for (int i = 0; i < head_dim; i++) {
        config->perm[i] = entries[i].idx;
    }

    /* Step 6: Build inverse permutation */
    for (int i = 0; i < head_dim; i++) {
        config->inv_perm[config->perm[i]] = i;
    }
}

/* ── Auto-detect outlier fraction from variance spectrum ─────────── */

static int float_cmp_desc(const void * a, const void * b) {
    float fa = *(const float *)a;
    float fb = *(const float *)b;
    if (fa > fb) return -1;
    if (fa < fb) return  1;
    return 0;
}

float tq_auto_detect_outlier_frac(
    const float * channel_var,
    int head_dim,
    float * stats_out)
{
    assert(head_dim > 0 && head_dim <= TQ_OUTLIER_MAX_DIM);

    /* Sort variances descending to get the spectrum */
    float sorted[TQ_OUTLIER_MAX_DIM];
    for (int i = 0; i < head_dim; i++) sorted[i] = channel_var[i];
    qsort(sorted, (size_t)head_dim, sizeof(float), float_cmp_desc);

    /* Median is the center of the sorted array */
    float median = sorted[head_dim / 2];

    /* Degenerate case: uniform or zero variance → no outliers */
    if (median <= 1e-12f) {
        if (stats_out) {
            stats_out[0] = 1.0f;  /* max ratio = 1 */
            stats_out[1] = 0.0f;
            stats_out[2] = 0.0f;
        }
        return 0.0f;
    }

    /* Count channels above thresholds. sorted[0] is the maximum. */
    float max_ratio = sorted[0] / median;
    int n_moderate = 0;  /* > 2x median */
    int n_strong   = 0;  /* > 5x median */
    for (int i = 0; i < head_dim; i++) {
        if (sorted[i] > 5.0f * median) n_strong++;
        if (sorted[i] > 2.0f * median) n_moderate++;
    }

    if (stats_out) {
        stats_out[0] = max_ratio;
        stats_out[1] = (float)n_moderate;
        stats_out[2] = (float)n_strong;
    }

    /* Use n_moderate as the primary signal: captures channels that
     * meaningfully exceed the typical variance level. Map to block-aligned
     * fraction choices. The downstream tq_identify_outliers() will round
     * the actual n_outlier to multiples of 32 for SIMD alignment.
     *
     * Thresholds chosen so that:
     * - Near-Gaussian distributions (few outliers) → 0% or 12.5%
     * - Moderate heavy tails → 25% (the TQ2_1 default)
     * - Very heavy tails (rare after Hadamard) → 50%
     */
    float raw_frac = (float)n_moderate / (float)head_dim;

    if (raw_frac < 0.0625f) return 0.0f;     /* < 6.25% moderate outliers */
    if (raw_frac < 0.1875f) return 0.125f;   /* 6.25% - 18.75%            */
    if (raw_frac < 0.375f)  return 0.25f;    /* 18.75% - 37.5%            */
    return 0.5f;                             /* > 37.5% (heavy-tailed)    */
}

/* ── Experimental: parameterized auto-detect for threshold tuning ── */

/* Map raw_frac to block-aligned fraction choices. Shared by all metrics
 * that produce a "fraction of channels to protect" signal. */
static inline float map_raw_frac_to_choice(float raw_frac) {
    if (raw_frac < 0.0625f) return 0.0f;
    if (raw_frac < 0.1875f) return 0.125f;
    if (raw_frac < 0.375f)  return 0.25f;
    return 0.5f;
}

float tq_auto_detect_outlier_frac_ex(
    const float * channel_var,
    int head_dim,
    float total_variance,
    float median_total_var,
    int metric,
    float threshold,
    float * stats_out)
{
    assert(head_dim > 0 && head_dim <= TQ_OUTLIER_MAX_DIM);

    /* Sort variances descending to get the spectrum */
    float sorted[TQ_OUTLIER_MAX_DIM];
    for (int i = 0; i < head_dim; i++) sorted[i] = channel_var[i];
    qsort(sorted, (size_t)head_dim, sizeof(float), float_cmp_desc);

    float median = sorted[head_dim / 2];

    if (median <= 1e-12f) {
        if (stats_out) {
            stats_out[0] = 1.0f;
            stats_out[1] = 0.0f;
            stats_out[2] = 0.0f;
        }
        return 0.0f;
    }

    /* Always compute the baseline signals so stats_out is meaningful
     * regardless of metric. */
    float max_ratio = sorted[0] / median;
    int n_moderate = 0;
    int n_strong   = 0;
    for (int i = 0; i < head_dim; i++) {
        if (sorted[i] > 5.0f * median) n_strong++;
        if (sorted[i] > threshold * median) n_moderate++;
    }

    if (stats_out) {
        stats_out[0] = max_ratio;
        stats_out[1] = (float)n_moderate;
        stats_out[2] = (float)n_strong;
    }

    switch (metric) {
        case TQ_OUTLIER_METRIC_MAX_RATIO: {
            /* Layer is "outlier-heavy" if any channel stands out sharply.
             * Returns a flat 0.25 (promote to TQ2_1) or 0.0 (flat). */
            return (max_ratio > threshold) ? 0.25f : 0.0f;
        }

        case TQ_OUTLIER_METRIC_TOTAL_VAR: {
            /* Per-layer total variance predicts "information density."
             * Layers with more total variance than the cross-layer median
             * need more bits regardless of per-channel outlier structure. */
            if (median_total_var <= 1e-12f) return 0.0f;
            float ratio = total_variance / median_total_var;
            return (ratio > threshold) ? 0.25f : 0.0f;
        }

        case TQ_OUTLIER_METRIC_HYBRID: {
            /* Promote if EITHER signal fires: n_moderate raw_frac > 6.25%
             * OR max_ratio > threshold. */
            float raw_frac = (float)n_moderate / (float)head_dim;
            float from_nmod = map_raw_frac_to_choice(raw_frac);
            float from_max  = (max_ratio > threshold) ? 0.25f : 0.0f;
            return (from_nmod > from_max) ? from_nmod : from_max;
        }

        case TQ_OUTLIER_METRIC_N_MODERATE:
        default: {
            /* Current default: n_moderate threshold.
             * Note: the threshold argument controls the "2× median" multiplier
             * so lowering it (e.g. to 1.5) promotes more layers. */
            float raw_frac = (float)n_moderate / (float)head_dim;
            return map_raw_frac_to_choice(raw_frac);
        }
    }
}

/* ── Uniform init (no outlier split) ──────────────────────────────── */

void tq_outlier_config_init_uniform(
    tq_outlier_config * config,
    int head_dim,
    tq_tier_t tier)
{
    assert(head_dim > 0 && head_dim <= TQ_OUTLIER_MAX_DIM);

    config->head_dim = head_dim;
    config->n_outlier = 0;
    config->n_normal  = head_dim;
    config->outlier_frac = 0.0f;
    config->outlier_tier = tier;
    config->normal_tier  = tier;

    /* Identity permutation */
    for (int i = 0; i < head_dim; i++) {
        config->perm[i] = i;
        config->inv_perm[i] = i;
        config->channel_var[i] = 0.0f;
    }
}

/* ── Buffer sizes ─────────────────────────────────────────────────── */

void tq_mixed_buffer_sizes(
    const tq_outlier_config * config,
    size_t * outlier_size,
    size_t * normal_size)
{
    int obs = tier_block_size(config->outlier_tier);
    int nbs = tier_block_size(config->normal_tier);

    /* Number of blocks for each region */
    int n_outlier_blocks = (config->n_outlier + obs - 1) / obs;
    int n_normal_blocks  = (config->n_normal  + nbs - 1) / nbs;

    *outlier_size = (size_t)n_outlier_blocks * tier_block_bytes(config->outlier_tier);
    *normal_size  = (size_t)n_normal_blocks  * tier_block_bytes(config->normal_tier);
}

float tq_mixed_effective_bpe(const tq_outlier_config * config) {
    if (config->head_dim == 0) return 0.0f;
    float outlier_bits = (float)config->n_outlier * tier_bpe(config->outlier_tier);
    float normal_bits  = (float)config->n_normal  * tier_bpe(config->normal_tier);
    return (outlier_bits + normal_bits) / (float)config->head_dim;
}

/* ── Channel permutation ──────────────────────────────────────────── */

void tq_permute_channels(const float * x, float * y, const int * perm, int n) {
    for (int i = 0; i < n; i++) {
        y[i] = x[perm[i]];
    }
}

void tq_unpermute_channels(const float * x, float * y, const int * inv_perm, int n) {
    for (int i = 0; i < n; i++) {
        y[i] = x[inv_perm[i]];
    }
}

/* ── Mixed-precision quantize ─────────────────────────────────────── */

void tq_mixed_quantize(
    const float * x,
    const tq_outlier_config * config,
    void * outlier_buf,
    void * normal_buf)
{
    const int hd = config->head_dim;
    const int no = config->n_outlier;
    const int nn = config->n_normal;

    /* Step 1: Permute channels (outliers first) */
    float permuted[TQ_OUTLIER_MAX_DIM];
    tq_permute_channels(x, permuted, config->perm, hd);

    /* Step 2: Quantize outlier channels with higher-precision tier */
    if (no > 0) {
        tier_quantize(permuted, outlier_buf, no, config->outlier_tier);
    }

    /* Step 3: Quantize normal channels with lower-precision tier */
    if (nn > 0) {
        tier_quantize(permuted + no, normal_buf, nn, config->normal_tier);
    }
}

/* ── Mixed-precision dequantize ───────────────────────────────────── */

void tq_mixed_dequantize(
    const void * outlier_buf,
    const void * normal_buf,
    const tq_outlier_config * config,
    float * y)
{
    const int hd = config->head_dim;
    const int no = config->n_outlier;
    const int nn = config->n_normal;

    /* Step 1: Dequantize both tiers into permuted order */
    float permuted[TQ_OUTLIER_MAX_DIM];
    if (no > 0) {
        tier_dequantize(outlier_buf, permuted, no, config->outlier_tier);
    }
    if (nn > 0) {
        tier_dequantize(normal_buf, permuted + no, nn, config->normal_tier);
    }

    /* Step 2: Unpermute back to original channel order */
    tq_unpermute_channels(permuted, y, config->inv_perm, hd);
}
