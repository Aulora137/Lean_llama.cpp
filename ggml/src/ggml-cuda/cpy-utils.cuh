#pragma once

#include "ggml-common.h"

template<typename src_t, typename dst_t>
static __device__ __forceinline__ void convert_flt(const src_t * src, dst_t * dst) {
    if constexpr (std::is_same_v<src_t, dst_t>) {
        *dst = *src;
    } else {
        *dst = float(*src);
    }
}

static __device__ __forceinline__ int best_index_int8(int n, const int8_t * val, float x) {
    if (x <= val[0]) return 0;
    if (x >= val[n-1]) return n-1;
    int ml = 0, mu = n-1;
    while (mu-ml > 1) {
        int mav = (ml+mu)/2;
        if (x < val[mav]) mu = mav; else ml = mav;
    }
    return x - val[mu-1] < val[mu] - x ? mu-1 : mu;
}

static __device__ void quantize_f32_q4_0_block(const float * __restrict__ x, block_q4_0 * __restrict__ y) {
    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK4_0; ++j) {
        const float v = x[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }

    const float d  = vmax / -8;
    const float id = d ? 1.0f/d : 0.0f;

    y->d = d;

    float sumqx = 0, sumq2 = 0;
    for (int j = 0; j < QK4_0/2; ++j) {
        const float v0 = x[0       + j];
        const float v1 = x[QK4_0/2 + j];
        const float x0 = v0*id;
        const float x1 = v1*id;

        const uint8_t xi0 = min(15, (int8_t)(x0 + 8.5f));
        const uint8_t xi1 = min(15, (int8_t)(x1 + 8.5f));
        float q0 = xi0 - 8;
        float q1 = xi1 - 8;
        float w0 = v0*v0;
        float w1 = v1*v1;
        sumqx += w0*q0*v0 + w1*q1*v1;
        sumq2 += w0*q0*q0 + w1*q1*q1;

        y->qs[j]  = xi0;
        y->qs[j] |= xi1 << 4;
    }
    if (sumq2 > 0) {
        y->d = sumqx/sumq2;
    }
}

static __device__ void quantize_f32_q4_1_block(const float * __restrict__ x, block_q4_1 * __restrict__ y) {
    float vmin = FLT_MAX;
    float vmax = -FLT_MAX;

    for (int j = 0; j < QK4_1; ++j) {
        const float v = x[j];
        if (v < vmin) vmin = v;
        if (v > vmax) vmax = v;
    }

    const float d  = (vmax - vmin) / ((1 << 4) - 1);
    const float id = d ? 1.0f/d : 0.0f;

    y->dm.x = d;
    y->dm.y = vmin;

    for (int j = 0; j < QK4_1/2; ++j) {
        const float x0 = (x[0       + j] - vmin)*id;
        const float x1 = (x[QK4_1/2 + j] - vmin)*id;

        const uint8_t xi0 = min(15, (int8_t)(x0 + 0.5f));
        const uint8_t xi1 = min(15, (int8_t)(x1 + 0.5f));

        y->qs[j]  = xi0;
        y->qs[j] |= xi1 << 4;
    }
}

static __device__ void quantize_f32_q5_0_block(const float * __restrict__ x, block_q5_0 * __restrict__ y) {
    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK5_0; ++j) {
        const float v = x[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }

    const float d  = vmax / -16;
    const float id = d ? 1.0f/d : 0.0f;

    y->d = d;

    uint32_t qh = 0;
    for (int j = 0; j < QK5_0/2; ++j) {
        const float x0 = x[0       + j]*id;
        const float x1 = x[QK5_0/2 + j]*id;

        const uint8_t xi0 = min(31, (int8_t)(x0 + 16.5f));
        const uint8_t xi1 = min(31, (int8_t)(x1 + 16.5f));

        y->qs[j]  = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_0/2);
    }
    memcpy(y->qh, &qh, sizeof(qh));
}

static __device__ void quantize_f32_q5_1_block(const float * __restrict__ x, block_q5_1 * __restrict__ y) {
    float min = x[0];
    float max = x[0];

    for (int j = 1; j < QK5_1; ++j) {
        const float v = x[j];
        min = v < min ? v : min;
        max = v > max ? v : max;
    }

    const float d  = (max - min) / 31;
    const float id = d ? 1.0f/d : 0.0f;

    y->dm.x = d;
    y->dm.y = min;

    uint32_t qh = 0;
    for (int j = 0; j < QK5_1/2; ++j) {
        const float x0 = (x[0       + j] - min)*id;
        const float x1 = (x[QK5_1/2 + j] - min)*id;

        const uint8_t xi0 = (uint8_t)(x0 + 0.5f);
        const uint8_t xi1 = (uint8_t)(x1 + 0.5f);

        y->qs[j]  = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_1/2);
    }
    memcpy(y->qh, &qh, sizeof(qh));
}

static __device__ void quantize_f32_q8_0_block(const float * __restrict__ x, block_q8_0 * __restrict__ y) {
    float amax = 0.0f; // absolute max

    for (int j = 0; j < QK8_0; j++) {
        const float v = x[j];
        amax = fmaxf(amax, fabsf(v));
    }

    const float d = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f/d : 0.0f;

    y->d = d;

    for (int j = 0; j < QK8_0; ++j) {
        const float x0 = x[j]*id;
        y->qs[j] = roundf(x0);
    }
}

static __device__ void quantize_f32_iq4_nl_block(const float * __restrict__ x, block_iq4_nl * __restrict__ y) {
    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK4_NL; ++j) {
        const float v = x[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }

    float d = vmax / kvalues_iq4nl[0];
    const float id = d ? 1.0f/d : 0.0f;

    float sumqx = 0, sumq2 = 0;
    for (int j = 0; j < QK4_NL/2; ++j) {
        const float x0 = x[0        + j]*id;
        const float x1 = x[QK4_NL/2 + j]*id;
        const uint8_t xi0 = best_index_int8(16, kvalues_iq4nl, x0);
        const uint8_t xi1 = best_index_int8(16, kvalues_iq4nl, x1);
        y->qs[j] = xi0 | (xi1 << 4);
        const float v0 = kvalues_iq4nl[xi0];
        const float v1 = kvalues_iq4nl[xi1];
        const float w0 = x[0        + j]*x[0        + j];
        const float w1 = x[QK4_NL/2 + j]*x[QK4_NL/2 + j];
        sumqx += w0*v0*x[j] + w1*v1*x[QK4_NL/2 + j];
        sumq2 += w0*v0*v0 + w1*v1*v1;
    }

    //y->d = d;
    y->d = sumq2 > 0 ? sumqx/sumq2 : d;
}

static __device__ void quantize_f32_q6_0_block(const float * __restrict__ xi, block_q6_0 * __restrict__ y) {

    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK6_0; ++j) {
        const float v  = xi[j];
        const float av = fabsf(xi[j]);
        if (amax < av) {
            amax = av;
            vmax = v;
        }
    }

    const float d  = vmax / -32;
    const float id = d ? 1.0f/d : 0.0f;

    y->d = d;
    memset(y->qh, 0, QK6_0/4);

    for (int j = 0; j < QK6_0/2; ++j) {
        const float x0 = xi[0       + j]*id;
        const float x1 = xi[QK6_0/2 + j]*id;

        const uint8_t xi0 = min(63, (int8_t)(x0 + 32.5f));
        const uint8_t xi1 = min(63, (int8_t)(x1 + 32.5f));

        y->qs[j]  = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        const uint8_t h = (xi0 >> 4) | ((xi1 >> 4) << 2);
        y->qh[j%(QK6_0/4)] |= (h << 4*(j/(QK6_0/4)));
    }
}

// TurboQuant codebook tables (int8-scaled, scale factor = 127)
static constexpr __device__ int8_t tq2_levels_i8[4]  = { -127, -38, +38, +127 };
static constexpr __device__ int8_t tq3_levels_i8[8]  = { -127, -79, -45, -14, +14, +45, +79, +127 };
static constexpr __device__ int8_t tq4_levels_i8[16] = {
    -127, -96, -75, -58, -44, -31, -18, -6,
      +6, +18, +31, +44, +58, +75, +96, +127,
};

static __device__ void quantize_f32_tq4_0_block(const float * __restrict__ x, block_ktq4_0 * __restrict__ y) {
    float amax = 0.0f;

    for (int j = 0; j < QK_TQ4; ++j) {
        const float v = x[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
        }
    }

    float d = amax / 127.0f;
    const float id = d ? 127.0f / amax : 0.0f;

    float sumqx = 0, sumq2 = 0;
    for (int j = 0; j < QK_TQ4/2; ++j) {
        const float x0 = x[0          + j] * id;
        const float x1 = x[QK_TQ4/2   + j] * id;
        const uint8_t xi0 = best_index_int8(16, tq4_levels_i8, x0);
        const uint8_t xi1 = best_index_int8(16, tq4_levels_i8, x1);
        y->qs[j] = xi0 | (xi1 << 4);
        const float v0 = tq4_levels_i8[xi0];
        const float v1 = tq4_levels_i8[xi1];
        const float w0 = x[0          + j] * x[0          + j];
        const float w1 = x[QK_TQ4/2   + j] * x[QK_TQ4/2   + j];
        sumqx += w0 * v0 * x[j] + w1 * v1 * x[QK_TQ4/2 + j];
        sumq2 += w0 * v0 * v0   + w1 * v1 * v1;
    }

    // Scale d by 127 to match normalized dequant LUT convention:
    // dequant uses (codebook_i8/127) * d, so d must be in "amax" scale, not "amax/127"
    const float d_final = sumq2 > 0 ? sumqx / sumq2 : d;
    y->d = d_final * 127.0f;
}

static __device__ void quantize_f32_tq2_0_block(const float * __restrict__ x, block_ktq2_0 * __restrict__ y) {
    float amax = 0.0f;

    for (int j = 0; j < QK_TQ2; ++j) {
        const float v = x[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
        }
    }

    float d = amax / 127.0f;
    const float id = d ? 127.0f / amax : 0.0f;

    uint8_t indices[QK_TQ2];
    float sumqx = 0, sumq2 = 0;
    for (int j = 0; j < QK_TQ2; ++j) {
        const float xn = x[j] * id;
        const uint8_t idx = best_index_int8(4, tq2_levels_i8, xn);
        indices[j] = idx;
        const float v = tq2_levels_i8[idx];
        const float w = x[j] * x[j];
        sumqx += w * v * x[j];
        sumq2 += w * v * v;
    }

    const float d_final = sumq2 > 0 ? sumqx / sumq2 : d;
    y->d = d_final * 127.0f;

    // Pack 4 indices per byte (2-bit each)
    for (int byte = 0; byte < QK_TQ2/4; ++byte) {
        y->qs[byte] = indices[4*byte]
                    | (indices[4*byte + 1] << 2)
                    | (indices[4*byte + 2] << 4)
                    | (indices[4*byte + 3] << 6);
    }
}

static __device__ void quantize_f32_tq3_0_block(const float * __restrict__ x, block_ktq3_0 * __restrict__ y) {
    float amax = 0.0f;

    for (int j = 0; j < QK_TQ3; ++j) {
        const float v = x[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
        }
    }

    float d = amax / 127.0f;
    const float id = d ? 127.0f / amax : 0.0f;

    uint8_t indices[QK_TQ3];
    float sumqx = 0, sumq2 = 0;
    for (int j = 0; j < QK_TQ3; ++j) {
        const float xn = x[j] * id;
        const uint8_t idx = best_index_int8(8, tq3_levels_i8, xn);
        indices[j] = idx;
        const float v = tq3_levels_i8[idx];
        const float w = x[j] * x[j];
        sumqx += w * v * x[j];
        sumq2 += w * v * v;
    }

    const float d_final = sumq2 > 0 ? sumqx / sumq2 : d;
    y->d = d_final * 127.0f;

    // Pack 8 values per 3 bytes (3-bit each), 4 groups of 8 = 32 values, 12 bytes
    for (int g = 0; g < 4; ++g) {
        const uint8_t * idx = indices + g * 8;
        uint8_t * out = y->qs + g * 3;
        out[0] = (idx[0])       | (idx[1] << 3) | (idx[2] << 6);
        out[1] = (idx[2] >> 2)  | (idx[3] << 1) | (idx[4] << 4) | (idx[5] << 7);
        out[2] = (idx[5] >> 1)  | (idx[6] << 2) | (idx[7] << 5);
    }
}

// Helper: quantize 32 floats into TQ3 (3-bit) encoding, storing scale and packed qs
static __device__ void quantize_f32_tq3_sub(const float * __restrict__ x, ggml_half * d_out, uint8_t * qs_out) {
    float amax = 0.0f;
    for (int j = 0; j < 32; ++j) {
        if (amax < fabsf(x[j])) {
            amax = fabsf(x[j]);
        }
    }

    float d = amax / 127.0f;
    const float id = d ? 127.0f / amax : 0.0f;

    uint8_t indices[32];
    float sumqx = 0, sumq2 = 0;
    for (int j = 0; j < 32; ++j) {
        const float xn = x[j] * id;
        const uint8_t idx = best_index_int8(8, tq3_levels_i8, xn);
        indices[j] = idx;
        const float v = tq3_levels_i8[idx];
        const float w = x[j] * x[j];
        sumqx += w * v * x[j];
        sumq2 += w * v * v;
    }

    const float d_final_tq3 = sumq2 > 0 ? sumqx / sumq2 : d;
    *d_out = d_final_tq3 * 127.0f;

    for (int g = 0; g < 4; ++g) {
        const uint8_t * idx = indices + g * 8;
        uint8_t * out = qs_out + g * 3;
        out[0] = (idx[0])       | (idx[1] << 3) | (idx[2] << 6);
        out[1] = (idx[2] >> 2)  | (idx[3] << 1) | (idx[4] << 4) | (idx[5] << 7);
        out[2] = (idx[5] >> 1)  | (idx[6] << 2) | (idx[7] << 5);
    }
}

// Helper: quantize 32 floats into TQ2 (2-bit) encoding, storing scale and packed qs
static __device__ void quantize_f32_tq2_sub(const float * __restrict__ x, ggml_half * d_out, uint8_t * qs_out) {
    float amax = 0.0f;
    for (int j = 0; j < 32; ++j) {
        if (amax < fabsf(x[j])) {
            amax = fabsf(x[j]);
        }
    }

    float d = amax / 127.0f;
    const float id = d ? 127.0f / amax : 0.0f;

    uint8_t indices[32];
    float sumqx = 0, sumq2 = 0;
    for (int j = 0; j < 32; ++j) {
        const float xn = x[j] * id;
        const uint8_t idx = best_index_int8(4, tq2_levels_i8, xn);
        indices[j] = idx;
        const float v = tq2_levels_i8[idx];
        const float w = x[j] * x[j];
        sumqx += w * v * x[j];
        sumq2 += w * v * v;
    }

    const float d_final_tq2 = sumq2 > 0 ? sumqx / sumq2 : d;
    *d_out = d_final_tq2 * 127.0f;

    for (int byte = 0; byte < 8; ++byte) {
        qs_out[byte] = indices[4*byte]
                     | (indices[4*byte + 1] << 2)
                     | (indices[4*byte + 2] << 4)
                     | (indices[4*byte + 3] << 6);
    }
}

static __device__ void quantize_f32_tq2_1_block(const float * __restrict__ x, block_ktq2_1 * __restrict__ y) {
    // First 32 elements: outlier channels, TQ3 encoding
    quantize_f32_tq3_sub(x,      &y->d_out, y->qs_out);
    // Next 3x32 elements: normal channels, TQ2 encoding
    quantize_f32_tq2_sub(x + 32, &y->d_n0,  y->qs_n0);
    quantize_f32_tq2_sub(x + 64, &y->d_n1,  y->qs_n1);
    quantize_f32_tq2_sub(x + 96, &y->d_n2,  y->qs_n2);
}

// Wrapper functions for cpy.cu compatibility
static __device__ void cpy_blck_f32_q4_0(const char * cxi, char * cdsti) {
    quantize_f32_q4_0_block((const float *)cxi, (block_q4_0 *)cdsti);
}

static __device__ void cpy_blck_f32_q4_1(const char * cxi, char * cdsti) {
    quantize_f32_q4_1_block((const float *)cxi, (block_q4_1 *)cdsti);
}

static __device__ void cpy_blck_f32_q5_0(const char * cxi, char * cdsti) {
    quantize_f32_q5_0_block((const float *)cxi, (block_q5_0 *)cdsti);
}

static __device__ void cpy_blck_f32_q5_1(const char * cxi, char * cdsti) {
    quantize_f32_q5_1_block((const float *)cxi, (block_q5_1 *)cdsti);
}

static __device__ void cpy_blck_f32_q6_0(const char * cxi, char * cdsti) {
    quantize_f32_q6_0_block((const float *)cxi, (block_q6_0 *)cdsti);
}

static __device__ void cpy_blck_f32_q8_0(const char * cxi, char * cdsti) {
    quantize_f32_q8_0_block((const float *)cxi, (block_q8_0 *)cdsti);
}

static __device__ void cpy_blck_f32_iq4_nl(const char * cxi, char * cdsti) {
    quantize_f32_iq4_nl_block((const float *)cxi, (block_iq4_nl *)cdsti);
}

static __device__ void cpy_blck_f32_tq4_0(const char * cxi, char * cdsti) {
    quantize_f32_tq4_0_block((const float *)cxi, (block_ktq4_0 *)cdsti);
}

static __device__ void cpy_blck_f32_tq2_0(const char * cxi, char * cdsti) {
    quantize_f32_tq2_0_block((const float *)cxi, (block_ktq2_0 *)cdsti);
}

static __device__ void cpy_blck_f32_tq3_0(const char * cxi, char * cdsti) {
    quantize_f32_tq3_0_block((const float *)cxi, (block_ktq3_0 *)cdsti);
}

static __device__ void cpy_blck_f32_tq2_1(const char * cxi, char * cdsti) {
    quantize_f32_tq2_1_block((const float *)cxi, (block_ktq2_1 *)cdsti);
}

template<typename src_t, typename dst_t>
static __device__ void cpy_1_flt(const char * cxi, char * cdsti) {
    convert_flt((const src_t *)cxi, (dst_t *)cdsti);
}
