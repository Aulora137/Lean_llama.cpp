#include "common.cuh"

static __device__ __forceinline__ void dequantize_q4_0(const void * vx, const int64_t ib, const int iqs, dfloat2 & v){
    const block_q4_0 * x = (const block_q4_0 *) vx;

    const dfloat d = x[ib].d;

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

#ifdef GGML_CUDA_F16
    v = __hsub2(v, {8.0f, 8.0f});
    v = __hmul2(v, {d, d});
#else
    v.x = (v.x - 8.0f) * d;
    v.y = (v.y - 8.0f) * d;
#endif // GGML_CUDA_F16
}

static __device__ __forceinline__ void dequantize_iq4_nl(const void * vx, const int64_t ib, const int iqs, dfloat2 & v){
    const block_q4_0 * x = (const block_q4_0 *) vx;

    const dfloat d = x[ib].d;

    const int vui = x[ib].qs[iqs];

    v.x = kvalues_iq4nl[vui & 0xF];
    v.y = kvalues_iq4nl[vui >>  4];

#ifdef GGML_CUDA_F16
    v = __hmul2(v, {d, d});
#else
    v.x = v.x * d;
    v.y = v.y * d;
#endif // GGML_CUDA_F16
}

static __device__ __forceinline__ void dequantize_q4_1(const void * vx, const int64_t ib, const int iqs, dfloat2 & v){
    const block_q4_1 * x = (const block_q4_1 *) vx;

    const dfloat d = __low2half(x[ib].dm);
    const dfloat m = __high2half(x[ib].dm);

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

#ifdef GGML_CUDA_F16
    v = __hmul2(v, {d, d});
    v = __hadd2(v, {m, m});
#else
    v.x = (v.x * d) + m;
    v.y = (v.y * d) + m;
#endif // GGML_CUDA_F16
}

static __device__ __forceinline__ void dequantize_q5_0(const void * vx, const int64_t ib, const int iqs, dfloat2 & v){
    const block_q5_0 * x = (const block_q5_0 *) vx;

    const dfloat d = x[ib].d;

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);

#ifdef GGML_CUDA_F16
    v = __hsub2(v, {16.0f, 16.0f});
    v = __hmul2(v, {d, d});
#else
    v.x = (v.x - 16.0f) * d;
    v.y = (v.y - 16.0f) * d;
#endif // GGML_CUDA_F16
}

static __device__ __forceinline__ void dequantize_q5_1(const void * vx, const int64_t ib, const int iqs, dfloat2 & v){
    const block_q5_1 * x = (const block_q5_1 *) vx;

    const dfloat d = __low2half(x[ib].dm);
    const dfloat m = __high2half(x[ib].dm);

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);

#ifdef GGML_CUDA_F16
    v = __hmul2(v, {d, d});
    v = __hadd2(v, {m, m});
#else
    v.x = (v.x * d) + m;
    v.y = (v.y * d) + m;
#endif // GGML_CUDA_F16
}

static __device__ __forceinline__ void dequantize_q6_0(const void * vx, const int64_t ib, const int iqs, dfloat2 & v){
    const block_q6_0 * x = (const block_q6_0 *) vx;

    const dfloat d = x[ib].d;

    const uint8_t h = x[ib].qh[iqs%8] >> 4*(iqs/8);
    v.x = ((x[ib].qs[iqs] & 0xf) | ((h & 0x3) << 4));
    v.y = ((x[ib].qs[iqs] >>  4) | ((h & 0xc) << 2));

#ifdef GGML_CUDA_F16
    v = __hsub2(v, {32.0f, 32.0f});
    v = __hmul2(v, {d, d});
#else
    v.x = (v.x - 32.0f) * d;
    v.y = (v.y - 32.0f) * d;
#endif // GGML_CUDA_F16
}

static __device__ __forceinline__ void dequantize_q8_0(const void * vx, const int64_t ib, const int iqs, dfloat2 & v){
    const block_q8_0 * x = (const block_q8_0 *) vx;

    const dfloat d = x[ib].d;

    v.x = x[ib].qs[iqs + 0];
    v.y = x[ib].qs[iqs + 1];

#ifdef GGML_CUDA_F16
    v = __hmul2(v, {d, d});
#else
    v.x *= d;
    v.y *= d;
#endif // GGML_CUDA_F16
}

// ── LeanKV TurboQuant dequantization ──────────────────────────────────

// Lloyd-Max codebook values scaled by 1/127 for direct float multiply
__constant__ static const float tq2_lut[4]  = { -1.0f, -0.29921260f, 0.29921260f, 1.0f };
__constant__ static const float tq3_lut[8]  = { -1.0f, -0.62204724f, -0.35433071f, -0.11023622f,
                                                  0.11023622f, 0.35433071f, 0.62204724f, 1.0f };
__constant__ static const float tq4_lut[16] = { -1.0f, -0.75590551f, -0.59055118f, -0.45669291f,
                                                 -0.34645669f, -0.24409449f, -0.14173228f, -0.04724409f,
                                                  0.04724409f, 0.14173228f, 0.24409449f, 0.34645669f,
                                                  0.45669291f, 0.59055118f, 0.75590551f, 1.0f };

// Helper: unpack a single TQ3 index from a 3-byte group
static __device__ __forceinline__ int tq3_unpack_idx(const uint8_t * in, int pos) {
    const uint8_t b0 = in[0], b1 = in[1], b2 = in[2];
    switch (pos) {
        case 0: return  b0       & 7;
        case 1: return (b0 >> 3) & 7;
        case 2: return ((b0 >> 6) & 3) | ((b1 & 1) << 2);
        case 3: return (b1 >> 1) & 7;
        case 4: return (b1 >> 4) & 7;
        case 5: return ((b1 >> 7) & 1) | ((b2 & 3) << 1);
        case 6: return (b2 >> 2) & 7;
        case 7: return (b2 >> 5) & 7;
        default: return 0;
    }
}

// Helper: unpack a single TQ2 index from a byte array
static __device__ __forceinline__ int tq2_unpack_idx(const uint8_t * qs, int elem) {
    return (qs[elem / 4] >> ((elem % 4) * 2)) & 3;
}

// TQ4_0: 32 elements, 4-bit indices packed as nibbles, QK=32, QR=2
// Convention: iqs=0..15, v.x=elem[iqs], v.y=elem[iqs+16]
// Packing: qs[i] low nibble = elem[i], high nibble = elem[i+16]
static __device__ __forceinline__ void dequantize_tq4_0(const void * vx, const int64_t ib, const int iqs, dfloat2 & v){
    const block_ktq4_0 * x = (const block_ktq4_0 *) vx;
    const float d = __half2float(x[ib].d);

    const uint8_t byte = x[ib].qs[iqs];
    v.x = tq4_lut[byte & 0x0F] * d;
    v.y = tq4_lut[byte >> 4]   * d;
}

// TQ2_0: 32 elements, 2-bit indices packed 4 per byte, QK=32, QR=2
// Convention: iqs=0..15, v.x=elem[iqs], v.y=elem[iqs+16]
// Packing: qs[8 bytes], element i is at qs[i/4] bits (i%4)*2..(i%4)*2+1
static __device__ __forceinline__ void dequantize_tq2_0(const void * vx, const int64_t ib, const int iqs, dfloat2 & v){
    const block_ktq2_0 * x = (const block_ktq2_0 *) vx;
    const float d = __half2float(x[ib].d);

    const int idx0 = tq2_unpack_idx(x[ib].qs, iqs);
    const int idx1 = tq2_unpack_idx(x[ib].qs, iqs + 16);

    v.x = tq2_lut[idx0] * d;
    v.y = tq2_lut[idx1] * d;
}

// TQ3_0: 32 elements, 3-bit indices packed 8 per 3 bytes, QK=32, QR=2
// Convention: iqs=0..15, v.x=elem[iqs], v.y=elem[iqs+16]
// Packing: 4 groups of 8 elements, each group = 3 bytes
static __device__ __forceinline__ void dequantize_tq3_0(const void * vx, const int64_t ib, const int iqs, dfloat2 & v){
    const block_ktq3_0 * x = (const block_ktq3_0 *) vx;
    const float d = __half2float(x[ib].d);

    // First value: element iqs (0..15)
    const int group0  = iqs / 8;
    const int within0 = iqs % 8;
    v.x = tq3_lut[tq3_unpack_idx(x[ib].qs + group0 * 3, within0)] * d;

    // Second value: element iqs+16 (16..31)
    const int elem1   = iqs + 16;
    const int group1   = elem1 / 8;
    const int within1  = elem1 % 8;
    v.y = tq3_lut[tq3_unpack_idx(x[ib].qs + group1 * 3, within1)] * d;
}

// TQ2_1: 128 elements, mixed TQ3 (32 outlier) + TQ2 (96 normal), QK=128, QR=2
// Convention: iqs=0..63, v.x=elem[iqs], v.y=elem[iqs+64]
static __device__ __forceinline__ void dequantize_tq2_1(const void * vx, const int64_t ib, const int iqs, dfloat2 & v){
    const block_ktq2_1 * x = (const block_ktq2_1 *) vx;

    // Decode element at position `pos` within the 128-element block
    auto decode_elem = [&](int pos) -> float {
        if (pos < 32) {
            // Outlier region: TQ3 encoding
            const float d = __half2float(x[ib].d_out);
            return tq3_lut[tq3_unpack_idx(x[ib].qs_out + (pos / 8) * 3, pos % 8)] * d;
        } else {
            // Normal region: 3 × TQ2 sub-blocks of 32 elements each
            const int norm = pos - 32;    // 0..95
            const int blk  = norm / 32;   // sub-block 0,1,2
            const int local = norm % 32;

            float d;
            const uint8_t * qs;
            switch (blk) {
                case 0:  d = __half2float(x[ib].d_n0); qs = x[ib].qs_n0; break;
                case 1:  d = __half2float(x[ib].d_n1); qs = x[ib].qs_n1; break;
                default: d = __half2float(x[ib].d_n2); qs = x[ib].qs_n2; break;
            }
            return tq2_lut[tq2_unpack_idx(qs, local)] * d;
        }
    };

    v.x = decode_elem(iqs);
    v.y = decode_elem(iqs + 64);
}
