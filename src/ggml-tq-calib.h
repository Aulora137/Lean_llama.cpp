/**
 * ggml-tq-calib.h — Empirical Lloyd-Max codebook fitting for LeanKV Phase 7a
 *
 * Given a set of real K-vector values (e.g. dumped by Lean_llama.cpp's
 * calibration hook), block-normalize them and fit a symmetric Lloyd-Max
 * codebook to the empirical distribution.
 *
 * The output is a drop-in replacement for the Gaussian TQ{2,3,4}_LEVELS
 * constants in ggml-tq.h — same shape, same block format, better levels
 * for the actual model's K-vector distribution.
 *
 * This is a pure offline routine. Runtime quant/dequant kernels are
 * unchanged — they just read from a different LEVELS array when an
 * empirical codebook is active.
 *
 * Reference: the Python prototype lives in scripts/analyze_empirical_codebook.py.
 * This header ports the symmetric Lloyd-Max fit to C so it can run during
 * first-load calibration without a Python dependency.
 */

#ifndef GGML_TQ_CALIB_H
#define GGML_TQ_CALIB_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Block normalization ───────────────────────────────────────────── */

/**
 * Divide each block of `block_size` elements by its absolute maximum.
 *
 *   in  : [n_blocks * block_size] floats
 *   out : [n_blocks * block_size] floats, each block scaled to lie in [-1, 1]
 *
 * Blocks with amax < 1e-12 are written as zero. Output may alias input.
 */
void leankv_block_normalize(const float * in,
                            float * out,
                            size_t n_blocks,
                            int block_size);

/* ── Symmetric Lloyd-Max fit ──────────────────────────────────────── */

/**
 * Fit a symmetric Lloyd-Max codebook on `data` (values already in [-1, 1]
 * after block normalization).
 *
 *   data      : [n] normalized values (zeros are skipped internally)
 *   n         : number of samples
 *   n_levels  : must be even (4, 8, or 16)
 *   n_iter    : Lloyd iteration cap (40 is plenty — usually converges by ~10)
 *   out_levels: [n_levels] output codebook (sorted ascending, symmetric
 *               around 0, outermost = ±1.0)
 *
 * Returns the number of iterations actually run (<= n_iter).
 *
 * Uses quantile-based initialization on |data| so convergence is stable
 * and repeatable (no RNG).
 *
 * Matches the Python reference `lloyd_max_symmetric` in
 * scripts/analyze_empirical_codebook.py bit-for-bit (within 1e-6).
 */
int leankv_lloyd_max_symmetric(const float * data,
                               size_t n,
                               int n_levels,
                               int n_iter,
                               float * out_levels);

/* ── Convenience: fit directly from raw blocks ────────────────────── */

/**
 * One-shot: normalize `n_blocks` blocks of `block_size` floats, then fit
 * `n_levels` symmetric Lloyd-Max levels on the concatenation.
 *
 * Allocates a temporary normalization buffer of size n_blocks*block_size.
 *
 * Returns the number of Lloyd iterations run.
 */
int leankv_fit_empirical_codebook(const float * blocks,
                                  size_t n_blocks,
                                  int block_size,
                                  int n_levels,
                                  int n_iter,
                                  float * out_levels);

/* ── Quality metrics (optional, for validation / logging) ─────────── */

/**
 * Compute MSE and signal power of `values` when quantized to the nearest
 * level in `levels` (nearest-neighbor in the 1D embedding).
 *
 *   mse_out : mean squared error
 *   pwr_out : mean squared signal
 *
 * SNR in dB = 10 * log10(pwr / mse).
 */
void leankv_codebook_mse(const float * values,
                         size_t n,
                         const float * levels,
                         int n_levels,
                         double * mse_out,
                         double * pwr_out);

#ifdef __cplusplus
}
#endif

#endif /* GGML_TQ_CALIB_H */
