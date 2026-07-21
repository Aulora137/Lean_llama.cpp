// Standalone test: TQ3 quantize+dequantize roundtrip quality
// Compile: cc -O2 -I../ggml/include -I../ggml/src -o test-tq-roundtrip test-tq-roundtrip.c ../ggml/src/ggml-tq.c ../ggml/src/ggml-tq-runtime.c -lm
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "ggml.h"
#include "ggml-common.h"
#include "ggml-quants.h"

int main(void) {
    // Generate 128 random Gaussian-like values (simulating post-Hadamard K)
    const int N = 128;
    float input[128], output[128], backup[128];

    srand(42);
    for (int i = 0; i < N; i++) {
        // Box-Muller for Gaussian
        float u1 = (float)(rand() + 1) / (float)(RAND_MAX + 1u);
        float u2 = (float)(rand() + 1) / (float)(RAND_MAX + 1u);
        input[i] = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
        input[i] *= 10.0f;  // Scale to typical post-Hadamard range
    }

    // Print first 8 input values
    printf("Input[0:8]:  ");
    for (int i = 0; i < 8; i++) printf("%8.4f ", input[i]);
    printf("\n");

    // TQ3 roundtrip: quantize then dequantize
    memcpy(backup, input, sizeof(input));

    // Process in 32-element blocks (same as noise op)
    for (int b = 0; b < N / 32; b++) {
        block_ktq3_0 blk;
        quantize_row_ktq3_0_ref(input + b * 32, &blk, 32);

        // no block-level debug

        dequantize_row_ktq3_0(&blk, input + b * 32, 32);
    }

    // Print first 8 output values
    printf("Output[0:8]: ");
    for (int i = 0; i < 8; i++) printf("%8.4f ", input[i]);
    printf("\n");

    // Compute errors
    printf("Errors[0:8]: ");
    for (int i = 0; i < 8; i++) printf("%8.4f ", input[i] - backup[i]);
    printf("\n");

    // Compute overall SNR
    float sig2 = 0, err2 = 0;
    for (int i = 0; i < N; i++) {
        sig2 += backup[i] * backup[i];
        float e = input[i] - backup[i];
        err2 += e * e;
    }
    float snr = (err2 > 0) ? 10.0f * log10f(sig2 / err2) : 999.0f;
    printf("SNR: %.1f dB (sig=%.4f, err=%.6f)\n", snr, sqrtf(sig2), sqrtf(err2));
    printf("RMS relative error: %.1f%%\n", 100.0f * sqrtf(err2 / sig2));

    // Per-block SNR
    for (int b = 0; b < N / 32; b++) {
        float bs = 0, be = 0;
        for (int j = 0; j < 32; j++) {
            int idx = b * 32 + j;
            bs += backup[idx] * backup[idx];
            float e = input[idx] - backup[idx];
            be += e * e;
        }
        float bsnr = (be > 0) ? 10.0f * log10f(bs / be) : 999.0f;
        printf("Block %d SNR: %.1f dB\n", b, bsnr);
    }

    // Also test TQ2
    printf("\n--- TQ2 roundtrip ---\n");
    memcpy(input, backup, sizeof(input));
    for (int b = 0; b < N / 32; b++) {
        block_ktq2_0 blk;
        quantize_row_ktq2_0_ref(input + b * 32, &blk, 32);
        dequantize_row_ktq2_0(&blk, input + b * 32, 32);
    }
    sig2 = 0; err2 = 0;
    for (int i = 0; i < N; i++) {
        sig2 += backup[i] * backup[i];
        float e = input[i] - backup[i];
        err2 += e * e;
    }
    snr = (err2 > 0) ? 10.0f * log10f(sig2 / err2) : 999.0f;
    printf("TQ2 SNR: %.1f dB, RMS relative error: %.1f%%\n", snr, 100.0f * sqrtf(err2 / sig2));

    return 0;
}
