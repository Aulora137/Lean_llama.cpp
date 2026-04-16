/**
 * test-tq3-attn-scale.c — Attention-aware scaling experiment for TQ3_0
 *
 * Tests whether optimizing the TQ3 block scale factor for dot-product
 * fidelity (attention-aware) improves on MSE-optimal scaling.
 *
 * Theory: for attention score s = q · k, error Δs = q · e.
 * If q_i are i.i.d., E[Δs²] = σ²_q · Σ e²_i = σ²_q · n · MSE.
 * So MSE-optimal = dot-product-optimal when Q is i.i.d.
 * Post-Hadamard Q is near-i.i.d., so the gap should be small.
 * This test quantifies "small."
 *
 * Build: cc -O2 -o test-tq3-attn-scale test-tq3-attn-scale.c -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#define QK 32      /* TQ3 block size */
#define NLEVELS 8

/* Lloyd-Max codebook for N(0,1), normalized to [-1, 1] */
static const float TQ3_LEVELS[8] = {
    -1.0000000f, -0.6245203f, -0.3513239f, -0.1138989f,
    +0.1138989f, +0.3513239f, +0.6245203f, +1.0000000f,
};

static const float TQ3_BOUNDARIES[7] = {
    -0.8122602f, -0.4879221f, -0.2326114f, +0.0000000f,
    +0.2326114f, +0.4879221f, +0.8122602f,
};

/* ── RNG: Box-Muller ──────────────────────────────────────────────── */

static float randn(void) {
    float u1 = (float)(rand() + 1) / ((float)RAND_MAX + 2.0f);
    float u2 = (float)(rand() + 1) / ((float)RAND_MAX + 2.0f);
    return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
}

/* ── Quantization helpers ─────────────────────────────────────────── */

static int find_nearest(float xn) {
    for (int i = 0; i < 7; i++) {
        if (xn < TQ3_BOUNDARIES[i]) return i;
    }
    return 7;
}

/* Quantize block with a given scale d, return indices */
static void quantize_block(const float *block, float d, uint8_t *indices) {
    if (d == 0.0f) {
        memset(indices, 4, QK); /* map to +0.114 (near zero) */
        return;
    }
    float id = 1.0f / d;
    for (int j = 0; j < QK; j++) {
        float xn = block[j] * id;
        if (xn < -1.0f) xn = -1.0f;
        if (xn >  1.0f) xn =  1.0f;
        indices[j] = find_nearest(xn);
    }
}

/* Reconstruct block from indices + scale */
static void dequantize_block(const uint8_t *indices, float d, float *out) {
    for (int j = 0; j < QK; j++) {
        out[j] = TQ3_LEVELS[indices[j]] * d;
    }
}

/* Least-squares optimal scale for given index assignment */
static float optimal_scale(const float *block, const uint8_t *indices) {
    float num = 0.0f, den = 0.0f;
    for (int j = 0; j < QK; j++) {
        float lev = TQ3_LEVELS[indices[j]];
        num += block[j] * lev;
        den += lev * lev;
    }
    return (den > 0.0f) ? num / den : 0.0f;
}

/* Block MSE */
static float block_mse(const float *block, const uint8_t *indices, float d) {
    float mse = 0.0f;
    for (int j = 0; j < QK; j++) {
        float err = block[j] - TQ3_LEVELS[indices[j]] * d;
        mse += err * err;
    }
    return mse / QK;
}

/* ── Strategy 1: Current MSE-optimal (with coordinate descent) ──── */

static float quantize_mse_optimal(const float *block, uint8_t *indices, float *out_d) {
    float amax = 0.0f;
    for (int j = 0; j < QK; j++) {
        float v = fabsf(block[j]);
        if (v > amax) amax = v;
    }
    if (amax == 0.0f) {
        memset(indices, 4, QK);
        *out_d = 0.0f;
        return 0.0f;
    }

    /* Initial assignment */
    quantize_block(block, amax, indices);
    float d = optimal_scale(block, indices);

    /* Coordinate descent (2 passes, matching production code) */
    float best = block_mse(block, indices, d);
    for (int pass = 0; pass < 2; pass++) {
        int improved = 0;
        for (int j = 0; j < QK; j++) {
            uint8_t orig = indices[j];
            if (orig > 0) {
                indices[j] = orig - 1;
                float nd = optimal_scale(block, indices);
                float nm = block_mse(block, indices, nd);
                if (nm < best) { best = nm; d = nd; improved = 1; continue; }
                indices[j] = orig;
            }
            if (orig < 7) {
                indices[j] = orig + 1;
                float nd = optimal_scale(block, indices);
                float nm = block_mse(block, indices, nd);
                if (nm < best) { best = nm; d = nd; improved = 1; continue; }
                indices[j] = orig;
            }
        }
        if (!improved) break;
    }

    *out_d = d;
    return best;
}

/* ── Strategy 2: Shrunk scale (α * d_mse) ────────────────────────── */

static float quantize_shrunk(const float *block, float alpha, uint8_t *indices, float *out_d) {
    float d_mse;
    quantize_mse_optimal(block, indices, &d_mse);
    /* Re-quantize with shrunk scale */
    float d = d_mse * alpha;
    quantize_block(block, d, indices);
    d = optimal_scale(block, indices);
    *out_d = d;
    return block_mse(block, indices, d);
}

/* ── Strategy 3: Percentile-based scale ───────────────────────────── */

static int cmp_float(const void *a, const void *b) {
    float fa = *(const float *)a, fb = *(const float *)b;
    return (fa > fb) - (fa < fb);
}

static float quantize_percentile(const float *block, float pctile, uint8_t *indices, float *out_d) {
    /* Sort absolute values, pick percentile as scale */
    float absvals[QK];
    for (int j = 0; j < QK; j++) absvals[j] = fabsf(block[j]);
    qsort(absvals, QK, sizeof(float), cmp_float);

    int idx = (int)(pctile * (QK - 1));
    if (idx < 1) idx = 1;
    if (idx >= QK) idx = QK - 1;
    float d = absvals[idx];
    if (d == 0.0f) d = absvals[QK - 1];

    quantize_block(block, d, indices);
    d = optimal_scale(block, indices);

    /* Still do coordinate descent */
    float best = block_mse(block, indices, d);
    for (int pass = 0; pass < 2; pass++) {
        int improved = 0;
        for (int j = 0; j < QK; j++) {
            uint8_t orig = indices[j];
            if (orig > 0) {
                indices[j] = orig - 1;
                float nd = optimal_scale(block, indices);
                float nm = block_mse(block, indices, nd);
                if (nm < best) { best = nm; d = nd; improved = 1; continue; }
                indices[j] = orig;
            }
            if (orig < 7) {
                indices[j] = orig + 1;
                float nd = optimal_scale(block, indices);
                float nm = block_mse(block, indices, nd);
                if (nm < best) { best = nm; d = nd; improved = 1; continue; }
                indices[j] = orig;
            }
        }
        if (!improved) break;
    }
    *out_d = d;
    return best;
}

/* ── Strategy 4: Grid search for best dot-product fidelity ────────
 *
 * Generate N_Q random query vectors, for each candidate scale d,
 * measure E[ (q·k - q·k̂)² ] and pick the d that minimizes it.
 * This is the "gold standard" attention-aware approach.
 * ──────────────────────────────────────────────────────────────── */

#define N_Q_SAMPLES 64  /* random Q vectors per block */

static float quantize_dotprod_optimal(const float *block, float *q_samples,
                                       uint8_t *best_indices, float *out_d) {
    float amax = 0.0f;
    for (int j = 0; j < QK; j++) {
        float v = fabsf(block[j]);
        if (v > amax) amax = v;
    }
    if (amax == 0.0f) {
        memset(best_indices, 4, QK);
        *out_d = 0.0f;
        return 0.0f;
    }

    /* Try scale factors from 0.7*amax to 1.1*amax in 41 steps */
    float best_dp_err = 1e30f;
    float best_d = amax;
    uint8_t trial_indices[QK];

    for (int si = 0; si <= 40; si++) {
        float alpha = 0.70f + 0.01f * si;
        float d_try = amax * alpha;

        quantize_block(block, d_try, trial_indices);
        float d = optimal_scale(block, trial_indices);

        /* Coordinate descent on MSE (still reasonable inner loop) */
        float mse = block_mse(block, trial_indices, d);
        for (int pass = 0; pass < 2; pass++) {
            int improved = 0;
            for (int j = 0; j < QK; j++) {
                uint8_t orig = trial_indices[j];
                if (orig > 0) {
                    trial_indices[j] = orig - 1;
                    float nd = optimal_scale(block, trial_indices);
                    float nm = block_mse(block, trial_indices, nd);
                    if (nm < mse) { mse = nm; d = nd; improved = 1; continue; }
                    trial_indices[j] = orig;
                }
                if (orig < 7) {
                    trial_indices[j] = orig + 1;
                    float nd = optimal_scale(block, trial_indices);
                    float nm = block_mse(block, trial_indices, nd);
                    if (nm < mse) { mse = nm; d = nd; improved = 1; continue; }
                    trial_indices[j] = orig;
                }
            }
            if (!improved) break;
        }

        /* Measure dot-product error against Q samples */
        float recon[QK];
        dequantize_block(trial_indices, d, recon);

        float dp_err = 0.0f;
        for (int qi = 0; qi < N_Q_SAMPLES; qi++) {
            float *q = q_samples + qi * QK;
            float dot_orig = 0.0f, dot_recon = 0.0f;
            for (int j = 0; j < QK; j++) {
                dot_orig  += q[j] * block[j];
                dot_recon += q[j] * recon[j];
            }
            float de = dot_orig - dot_recon;
            dp_err += de * de;
        }
        dp_err /= N_Q_SAMPLES;

        if (dp_err < best_dp_err) {
            best_dp_err = dp_err;
            best_d = d;
            memcpy(best_indices, trial_indices, QK);
        }
    }

    *out_d = best_d;
    return best_dp_err;
}

/* ── Measurement ──────────────────────────────────────────────────── */

typedef struct {
    double sum_dp_err2;     /* Σ (q·k - q·k̂)² */
    double sum_dp_orig2;    /* Σ (q·k)² */
    double sum_mse;
    double sum_cosine;
    int    n_blocks;
} metrics_t;

static void measure_block(const float *block, const uint8_t *indices, float d,
                          float *q_pool, int n_q, metrics_t *m) {
    float recon[QK];
    dequantize_block(indices, d, recon);

    /* MSE */
    float mse = 0.0f;
    for (int j = 0; j < QK; j++) {
        float e = block[j] - recon[j];
        mse += e * e;
    }
    m->sum_mse += mse / QK;

    /* Cosine similarity */
    float dot_kr = 0.0f, norm_k = 0.0f, norm_r = 0.0f;
    for (int j = 0; j < QK; j++) {
        dot_kr += block[j] * recon[j];
        norm_k += block[j] * block[j];
        norm_r += recon[j] * recon[j];
    }
    float cos_sim = dot_kr / (sqrtf(norm_k) * sqrtf(norm_r) + 1e-30f);
    m->sum_cosine += cos_sim;

    /* Dot-product error against Q pool */
    for (int qi = 0; qi < n_q; qi++) {
        float *q = q_pool + qi * QK;
        float dot_orig = 0.0f, dot_recon = 0.0f;
        for (int j = 0; j < QK; j++) {
            dot_orig  += q[j] * block[j];
            dot_recon += q[j] * recon[j];
        }
        float de = dot_orig - dot_recon;
        m->sum_dp_err2  += de * de;
        m->sum_dp_orig2 += dot_orig * dot_orig;
    }

    m->n_blocks++;
}

/* ── Main ─────────────────────────────────────────────────────────── */

#define N_BLOCKS    10000
#define N_Q_POOL    256    /* Q vectors for evaluation */

int main(void) {
    srand((unsigned)time(NULL));

    printf("=== TQ3 Attention-Aware Scaling Experiment ===\n");
    printf("Blocks: %d, Block size: %d, Q vectors: %d\n\n", N_BLOCKS, QK, N_Q_POOL);

    /* Generate Q pool (post-Hadamard: near-Gaussian) */
    float *q_pool = (float *)malloc(N_Q_POOL * QK * sizeof(float));
    for (int i = 0; i < N_Q_POOL * QK; i++) q_pool[i] = randn();

    /* Q samples for dot-product optimization (subset) */
    float *q_samples = (float *)malloc(N_Q_SAMPLES * QK * sizeof(float));
    for (int i = 0; i < N_Q_SAMPLES * QK; i++) q_samples[i] = randn();

    /* Generate K blocks (post-Hadamard: near-Gaussian) */
    float *blocks = (float *)malloc(N_BLOCKS * QK * sizeof(float));
    for (int i = 0; i < N_BLOCKS * QK; i++) blocks[i] = randn();

    /* ── Run each strategy ──────────────────────────────────────── */

    metrics_t m_mse     = {0};  /* Strategy 1: MSE-optimal (current) */
    metrics_t m_shrunk  = {0};  /* Strategy 2: shrunk scale */
    metrics_t m_pct95   = {0};  /* Strategy 3: 95th percentile */
    metrics_t m_pct90   = {0};  /* Strategy 3b: 90th percentile */
    metrics_t m_dpopt   = {0};  /* Strategy 4: dot-product grid search */

    uint8_t indices[QK];
    float d;

    printf("Running %d blocks...\n", N_BLOCKS);

    for (int b = 0; b < N_BLOCKS; b++) {
        const float *block = blocks + b * QK;

        /* Strategy 1: MSE-optimal (current production code) */
        quantize_mse_optimal(block, indices, &d);
        measure_block(block, indices, d, q_pool, N_Q_POOL, &m_mse);

        /* Strategy 2: shrunk (α=0.92) */
        quantize_shrunk(block, 0.92f, indices, &d);
        measure_block(block, indices, d, q_pool, N_Q_POOL, &m_shrunk);

        /* Strategy 3: 95th percentile */
        quantize_percentile(block, 0.95f, indices, &d);
        measure_block(block, indices, d, q_pool, N_Q_POOL, &m_pct95);

        /* Strategy 3b: 90th percentile */
        quantize_percentile(block, 0.90f, indices, &d);
        measure_block(block, indices, d, q_pool, N_Q_POOL, &m_pct90);

        /* Strategy 4: dot-product optimal (expensive, every 10th block) */
        if (b % 10 == 0) {
            quantize_dotprod_optimal(block, q_samples, indices, &d);
            measure_block(block, indices, d, q_pool, N_Q_POOL, &m_dpopt);
        }

        if (b % 2000 == 0 && b > 0) printf("  %d/%d blocks done\n", b, N_BLOCKS);
    }

    /* ── Report ─────────────────────────────────────────────────── */

    printf("\n%-25s %10s %10s %10s %10s\n",
           "Strategy", "MSE", "Cosine", "DP_SNR_dB", "DP_relErr%");
    printf("%-25s %10s %10s %10s %10s\n",
           "------------------------", "----------", "----------", "----------", "----------");

    struct {
        const char *name;
        metrics_t *m;
    } strats[] = {
        {"MSE-optimal (current)",  &m_mse},
        {"Shrunk (alpha=0.92)",    &m_shrunk},
        {"Percentile 95th",        &m_pct95},
        {"Percentile 90th",        &m_pct90},
        {"DotProd grid search",    &m_dpopt},
    };

    for (int i = 0; i < 5; i++) {
        metrics_t *m = strats[i].m;
        int nb = m->n_blocks;
        if (nb == 0) continue;
        int nq_total = nb * N_Q_POOL;

        float avg_mse    = (float)(m->sum_mse / nb);
        float avg_cosine = (float)(m->sum_cosine / nb);
        float dp_snr     = (m->sum_dp_err2 > 0)
                         ? 10.0f * log10f((float)(m->sum_dp_orig2 / m->sum_dp_err2))
                         : 999.0f;
        float dp_rel_err = 100.0f * sqrtf((float)(m->sum_dp_err2 / (m->sum_dp_orig2 + 1e-30)));

        printf("%-25s %10.6f %10.6f %10.2f %10.2f\n",
               strats[i].name, avg_mse, avg_cosine, dp_snr, dp_rel_err);
    }

    printf("\n");
    printf("DP_SNR_dB = 10·log10(Σ(q·k)² / Σ(q·k - q·k̂)²)  — higher is better\n");
    printf("DP_relErr = 100·√(Σerr²/Σsig²)                   — lower is better\n");
    printf("\nIf MSE-optimal ≈ DotProd-optimal, then attention-aware scaling\n");
    printf("provides no benefit (post-Hadamard Q is sufficiently i.i.d.).\n");

    free(q_pool);
    free(q_samples);
    free(blocks);

    return 0;
}
