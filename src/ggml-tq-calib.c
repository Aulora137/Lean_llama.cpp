/**
 * ggml-tq-calib.c — Empirical Lloyd-Max codebook fitting for LeanKV Phase 7a
 *
 * Ports scripts/analyze_empirical_codebook.py:lloyd_max_symmetric to C so
 * first-load calibration can run without a Python dependency.
 *
 * Algorithm (symmetric codebook):
 *   1. Take absolute values of the (already block-normalized) input.
 *   2. Drop near-zero values (|x| <= 1e-6) so the fit is not dominated by
 *      elements that quantize to 0 under any reasonable codebook.
 *   3. Quantile-initialize n_half levels at np.linspace(0.05, 0.98, n_half).
 *   4. Lloyd iterations on the positive half:
 *        edges  = [0, (L[0]+L[1])/2, ..., (L[n-2]+L[n-1])/2, +INF]
 *        L[i]   = mean{ |x| : edges[i] <= |x| < edges[i+1] }
 *      Convergence when max change < 1e-6 or n_iter reached.
 *   5. Rescale so L[-1] = 1.
 *   6. Mirror: out = sort([-L[::-1], +L]).
 *
 * Matches Python reference bit-for-bit within 1e-6 (verified against
 * qwen3_4b_big_calib.bin in the unit tests).
 */

#include "ggml-tq-calib.h"

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ── Small internal helpers ──────────────────────────────────────── */

static int cmp_float_asc(const void * a, const void * b) {
    float fa = *(const float *)a;
    float fb = *(const float *)b;
    if (fa < fb) return -1;
    if (fa > fb) return +1;
    return 0;
}

/* np.quantile with linear interpolation (method="linear"), on a *sorted*
 * ascending array. q in [0, 1]. */
static float quantile_sorted(const float * sorted, size_t n, double q) {
    if (n == 0) return 0.0f;
    if (n == 1) return sorted[0];
    if (q <= 0.0) return sorted[0];
    if (q >= 1.0) return sorted[n - 1];
    double pos = q * (double)(n - 1);
    size_t lo = (size_t)pos;
    size_t hi = lo + 1;
    if (hi >= n) return sorted[n - 1];
    double frac = pos - (double)lo;
    return (float)((1.0 - frac) * sorted[lo] + frac * sorted[hi]);
}

/* ── Block normalization ──────────────────────────────────────────── */

void leankv_block_normalize(const float * in,
                            float * out,
                            size_t n_blocks,
                            int block_size) {
    assert(block_size > 0);
    for (size_t b = 0; b < n_blocks; b++) {
        const float * src = in  + b * (size_t)block_size;
        float       * dst = out + b * (size_t)block_size;
        float amax = 0.0f;
        for (int i = 0; i < block_size; i++) {
            float a = fabsf(src[i]);
            if (a > amax) amax = a;
        }
        if (amax < 1e-12f) {
            for (int i = 0; i < block_size; i++) dst[i] = 0.0f;
            continue;
        }
        float inv = 1.0f / amax;
        for (int i = 0; i < block_size; i++) dst[i] = src[i] * inv;
    }
}

/* ── Symmetric Lloyd-Max fit ──────────────────────────────────────── */

int leankv_lloyd_max_symmetric(const float * data,
                               size_t n,
                               int n_levels,
                               int n_iter,
                               float * out_levels) {
    assert(n_levels >= 2 && (n_levels & 1) == 0);
    assert(out_levels != NULL);

    const int n_half = n_levels / 2;

    /* 1. Collect |data|, drop near-zero. */
    float * absv = (float *)malloc(n * sizeof(float));
    if (!absv) return 0;
    size_t m = 0;
    for (size_t i = 0; i < n; i++) {
        float a = fabsf(data[i]);
        if (a > 1e-6f) absv[m++] = a;
    }

    /* Degenerate cases — fall back to a reasonable default so the caller
     * still gets a valid (if uninteresting) codebook. */
    if (m < (size_t)n_levels) {
        free(absv);
        /* Emit uniform levels in [-1, 1] as a safe default. */
        for (int i = 0; i < n_half; i++) {
            float lv = (float)(i + 1) / (float)n_half;
            out_levels[n_half + i]       = lv;
            out_levels[n_half - 1 - i]   = -lv;
        }
        return 0;
    }

    /* 2. Sort for quantile-based init. */
    qsort(absv, m, sizeof(float), cmp_float_asc);

    /* 3. Quantile init at linspace(0.05, 0.98, n_half). */
    float * levels = (float *)malloc(n_half * sizeof(float));
    float * newlev = (float *)malloc(n_half * sizeof(float));
    if (!levels || !newlev) { free(absv); free(levels); free(newlev); return 0; }

    for (int i = 0; i < n_half; i++) {
        double q;
        if (n_half == 1) {
            q = 0.05;
        } else {
            q = 0.05 + (0.98 - 0.05) * (double)i / (double)(n_half - 1);
        }
        levels[i] = quantile_sorted(absv, m, q);
    }

    /* Ensure the init is strictly increasing so edges are well-defined. */
    for (int i = 1; i < n_half; i++) {
        if (levels[i] <= levels[i - 1]) {
            levels[i] = levels[i - 1] + 1e-6f;
        }
    }

    /* 4. Lloyd iterations on the sorted absolute values.
     *
     * Because absv is sorted, each bin is a contiguous slice and we can
     * compute the bin mean with two searches instead of n_half full
     * passes over the data. O(n_half * log m) per iteration.
     */
    int iter;
    for (iter = 0; iter < n_iter; iter++) {
        /* Build edges: [0, (L0+L1)/2, (L1+L2)/2, ..., +INF]. */
        /* Walk absv with two pointers; bin i covers [edges[i], edges[i+1]). */
        double prev_edge = 0.0;
        size_t lo = 0;
        for (int i = 0; i < n_half; i++) {
            double next_edge;
            if (i == n_half - 1) {
                next_edge = DBL_MAX;
            } else {
                next_edge = 0.5 * ((double)levels[i] + (double)levels[i + 1]);
            }

            /* Skip forward while absv[lo] < prev_edge (should already be
             * satisfied by monotonicity of the sweep, but guard anyway). */
            while (lo < m && (double)absv[lo] < prev_edge) lo++;

            /* Advance hi until we leave the bin. */
            size_t hi = lo;
            double sum = 0.0;
            size_t cnt = 0;
            while (hi < m && (double)absv[hi] < next_edge) {
                sum += (double)absv[hi];
                cnt++;
                hi++;
            }

            if (cnt > 0) {
                newlev[i] = (float)(sum / (double)cnt);
            } else {
                /* Empty bin: keep the old level so the codebook stays well-defined. */
                newlev[i] = levels[i];
            }

            lo = hi;
            prev_edge = next_edge;
        }

        /* Convergence check (matches Python's np.allclose atol=1e-6). */
        float max_delta = 0.0f;
        for (int i = 0; i < n_half; i++) {
            float d = fabsf(newlev[i] - levels[i]);
            if (d > max_delta) max_delta = d;
        }
        memcpy(levels, newlev, (size_t)n_half * sizeof(float));
        if (max_delta < 1e-6f) { iter++; break; }
    }

    /* 5. Normalize so the outermost level is exactly 1. */
    float max_lvl = levels[n_half - 1];
    if (max_lvl > 0.0f) {
        float inv = 1.0f / max_lvl;
        for (int i = 0; i < n_half; i++) levels[i] *= inv;
    }

    /* 6. Mirror into the full symmetric codebook. */
    for (int i = 0; i < n_half; i++) {
        out_levels[n_half + i]         = levels[i];
        out_levels[n_half - 1 - i]     = -levels[i];
    }

    free(absv);
    free(levels);
    free(newlev);
    return iter;
}

/* ── Convenience wrapper ──────────────────────────────────────────── */

int leankv_fit_empirical_codebook(const float * blocks,
                                  size_t n_blocks,
                                  int block_size,
                                  int n_levels,
                                  int n_iter,
                                  float * out_levels) {
    size_t total = n_blocks * (size_t)block_size;
    float * norm = (float *)malloc(total * sizeof(float));
    if (!norm) return 0;
    leankv_block_normalize(blocks, norm, n_blocks, block_size);
    int iter = leankv_lloyd_max_symmetric(norm, total, n_levels, n_iter, out_levels);
    free(norm);
    return iter;
}

/* ── Quality metrics ──────────────────────────────────────────────── */

/* For nearest-level lookup we use boundaries = midpoints (same rule the
 * fit uses). This is a linear scan — fine for n_levels <= 16. */
void leankv_codebook_mse(const float * values,
                         size_t n,
                         const float * levels,
                         int n_levels,
                         double * mse_out,
                         double * pwr_out) {
    double mse = 0.0;
    double pwr = 0.0;
    for (size_t i = 0; i < n; i++) {
        float x = values[i];
        /* Find nearest level by linear scan. */
        float best = levels[0];
        float best_d = fabsf(x - best);
        for (int k = 1; k < n_levels; k++) {
            float d = fabsf(x - levels[k]);
            if (d < best_d) { best_d = d; best = levels[k]; }
        }
        double err = (double)x - (double)best;
        mse += err * err;
        pwr += (double)x * (double)x;
    }
    if (n > 0) {
        mse /= (double)n;
        pwr /= (double)n;
    }
    if (mse_out) *mse_out = mse;
    if (pwr_out) *pwr_out = pwr;
}
