// Cross-architecture determinism probe for GGML_OP_DELTA_NET.
//
// Fills q/k/v/g/beta/state with a fixed LCG sequence, runs the op, and
// prints output samples + aggregates in hex-float so results can be
// diffed exactly between builds (ARM NEON fused vs x86 AVX2 fused vs
// generic C via LEANKV_NO_FUSED_DELTA_NET=1).
//
// Dims mirror a Qwen3.5-9B deltanet layer at prompt time.

#include "ggml.h"
#include "ggml-backend.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static uint32_t lcg_state = 0x12345678u;
static float lcg_float(void) { // deterministic, platform-independent
    lcg_state = lcg_state * 1664525u + 1013904223u;
    return (float)(lcg_state >> 8) / (float)(1u << 24); // [0,1)
}

static void fill_tensor(struct ggml_tensor * t, float lo, float hi) {
    float * d = (float *) t->data;
    const int64_t n = ggml_nelements(t);
    for (int64_t i = 0; i < n; i++) {
        d[i] = lo + (hi - lo) * lcg_float();
    }
}

int main(int argc, char ** argv) {
    const int n_threads = argc > 1 ? atoi(argv[1]) : 1;

    const int64_t S_k = 128, H_k = 16;   // key head dim / heads
    const int64_t S_v = 128, H_v = 32;   // value head dim / heads (gqa 2)
    const int64_t n_tokens = 64, n_seqs = 1;

    struct ggml_init_params ip = {
        .mem_size   = 512*1024*1024,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };
    struct ggml_context * ctx = ggml_init(ip);

    struct ggml_tensor * q     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_k, n_tokens, H_k, n_seqs);
    struct ggml_tensor * k     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_k, n_tokens, H_k, n_seqs);
    struct ggml_tensor * v     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v, n_tokens, H_v, n_seqs);
    struct ggml_tensor * g     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, n_tokens, 1, H_v, n_seqs);
    struct ggml_tensor * beta  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, n_tokens, H_v, n_seqs);
    struct ggml_tensor * state = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v, S_v*H_v, 1, n_seqs);

    // realistic magnitudes: unit-ish qkv, mild decay, beta in (0,1)
    fill_tensor(q,     -1.0f,  1.0f);
    fill_tensor(k,     -1.0f,  1.0f);
    fill_tensor(v,     -1.0f,  1.0f);
    fill_tensor(g,     -0.2f,  0.0f);
    fill_tensor(beta,   0.05f, 0.95f);
    fill_tensor(state, -0.5f,  0.5f);

    struct ggml_tensor * out = ggml_delta_net(ctx, q, k, v, g, beta, state, NULL);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(ctx, gf, n_threads);

    const float * o = (const float *) out->data;
    const int64_t output_size = S_v * H_v * n_tokens * n_seqs;
    const int64_t state_size  = S_v * S_v * H_v * n_seqs;

    double sum_o = 0, sum_s = 0;
    float max_o = -INFINITY, max_s = -INFINITY;
    for (int64_t i = 0; i < output_size; i++) { sum_o += o[i]; if (fabsf(o[i]) > max_o) max_o = fabsf(o[i]); }
    for (int64_t i = 0; i < state_size;  i++) { const float x = o[output_size + i]; sum_s += x; if (fabsf(x) > max_s) max_s = fabsf(x); }

    printf("threads=%d\n", n_threads);
    printf("out  first8:"); for (int i = 0; i < 8; i++) printf(" %a", o[i]); printf("\n");
    printf("out  last8: "); for (int i = 0; i < 8; i++) printf(" %a", o[output_size-8+i]); printf("\n");
    printf("stat first8:"); for (int i = 0; i < 8; i++) printf(" %a", o[output_size+i]); printf("\n");
    printf("out  sum=%.10e absmax=%.10e\n", sum_o, (double)max_o);
    printf("stat sum=%.10e absmax=%.10e\n", sum_s, (double)max_s);

    ggml_free(ctx);
    return 0;
}
