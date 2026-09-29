// CUDA-backend correctness test for the ConvRot INT8 tensorwise matmul.
// Same cases and same double-precision reference as the CPU test
// (convrot_correctness_test.c), which already passed on CPU. This proves
// the CUDA kernel (ggml_cuda_mul_mat_i8 in ggml-cuda.cu) computes the same
// thing on real GPU hardware, not just that it compiles and links.

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cuda.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static void hadamard_group_ref(double *values, int group_size) {
    double scale = 1.0 / sqrt((double)group_size);
    for (int i = 0; i < group_size; ++i) values[i] *= scale;
    for (int stride = 1; stride < group_size; stride *= 4) {
        for (int base = 0; base < group_size; base += 4 * stride) {
            for (int j = 0; j < stride; ++j) {
                int i0 = base + j, i1 = i0 + stride, i2 = i1 + stride, i3 = i2 + stride;
                double a = values[i0], b = values[i1], c = values[i2], d = values[i3];
                values[i0] =  a + b + c - d;
                values[i1] =  a + b - c + d;
                values[i2] =  a - b + c + d;
                values[i3] = -a + b + c + d;
            }
        }
    }
}

static void reference_mul_mat_i8(
        const int8_t *weight, int64_t k, int64_t n,
        const float *input, int64_t rows,
        int convrot_group_size,
        const float *weight_scale, const float *bias,
        double *out /* n*rows */) {
    for (int64_t row = 0; row < rows; ++row) {
        double amax = 0.0;
        double *qsrc = (double *)malloc(sizeof(double) * k);
        if (convrot_group_size == 0) {
            for (int64_t i = 0; i < k; ++i) {
                qsrc[i] = (double)input[row * k + i];
                amax = fmax(amax, fabs(qsrc[i]));
            }
        } else {
            for (int64_t group = 0; group < k; group += convrot_group_size) {
                double values[256];
                for (int i = 0; i < convrot_group_size; ++i) values[i] = (double)input[row * k + group + i];
                hadamard_group_ref(values, convrot_group_size);
                for (int i = 0; i < convrot_group_size; ++i) {
                    qsrc[group + i] = values[i];
                    amax = fmax(amax, fabs(values[i]));
                }
            }
        }
        double row_scale = amax / 127.0;
        int8_t *q = (int8_t *)malloc(k);
        if (row_scale == 0.0) {
            memset(q, 0, k);
        } else {
            double inv_scale = 1.0 / row_scale;
            for (int64_t i = 0; i < k; ++i) {
                long v = lrint(qsrc[i] * inv_scale);
                if (v > 127) v = 127;
                if (v < -127) v = -127;
                q[i] = (int8_t)v;
            }
        }
        for (int64_t out_row = 0; out_row < n; ++out_row) {
            int64_t sum = 0;
            for (int64_t i = 0; i < k; ++i) {
                sum += (int64_t)weight[out_row * k + i] * (int64_t)q[i];
            }
            double value = (double)sum * row_scale;
            if (weight_scale) value *= (double)weight_scale[out_row];
            if (bias) value += (double)bias[out_row];
            out[row * n + out_row] = value;
        }
        free(q);
        free(qsrc);
    }
}

static int run_case(ggml_backend_t backend, const char *label, int64_t k, int64_t n, int64_t rows,
                     const int8_t *weight_data, const float *input_data,
                     int convrot_group_size,
                     const float *weight_scale_data /* NULL => all 1.0 */,
                     const float *bias_data /* NULL => no bias */) {
    printf("=== %s (k=%lld n=%lld rows=%lld group_size=%d) ===\n",
           label, (long long)k, (long long)n, (long long)rows, convrot_group_size);

    size_t mem_size = ggml_tensor_overhead() * 16 + ggml_graph_overhead() + 1024;
    struct ggml_init_params params = { mem_size, NULL, /*no_alloc=*/ true };
    struct ggml_context *ctx = ggml_init(params);
    if (!ctx) { printf("FAIL: ggml_init returned NULL\n"); return 1; }

    struct ggml_tensor *weight = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, k, n);
    struct ggml_tensor *input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, rows);
    struct ggml_tensor *weight_scale = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    struct ggml_tensor *bias = NULL;
    if (bias_data) {
        bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    }

    struct ggml_tensor *result = ggml_mul_mat_i8_tensorwise(ctx, weight, input, weight_scale, bias, convrot_group_size);

    struct ggml_cgraph *graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);

    struct ggml_backend_buffer *buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) { printf("FAIL: ggml_backend_alloc_ctx_tensors returned NULL\n"); ggml_free(ctx); return 1; }

    ggml_backend_tensor_set(weight, weight_data, 0, (size_t)(k * n));
    ggml_backend_tensor_set(input, input_data, 0, sizeof(float) * (size_t)(k * rows));

    float *ws_buf = (float *)malloc(sizeof(float) * (size_t)n);
    for (int64_t i = 0; i < n; ++i) ws_buf[i] = weight_scale_data ? weight_scale_data[i] : 1.0f;
    ggml_backend_tensor_set(weight_scale, ws_buf, 0, sizeof(float) * (size_t)n);
    free(ws_buf);

    if (bias) {
        ggml_backend_tensor_set(bias, bias_data, 0, sizeof(float) * (size_t)n);
    }

    enum ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        printf("FAIL: ggml_backend_graph_compute status=%d\n", status);
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        return 1;
    }

    float *actual = (float *)malloc(sizeof(float) * (size_t)(n * rows));
    ggml_backend_tensor_get(result, actual, 0, sizeof(float) * (size_t)(n * rows));

    double *expected = (double *)malloc(sizeof(double) * (size_t)(n * rows));
    reference_mul_mat_i8(weight_data, k, n, input_data, rows, convrot_group_size,
                          weight_scale_data, bias_data, expected);

    int ok = 1;
    for (int64_t row = 0; row < rows; ++row) {
        for (int64_t out_row = 0; out_row < n; ++out_row) {
            double exp = expected[row * n + out_row];
            double act = (double)actual[row * n + out_row];
            double diff = fabs(exp - act);
            double tol = 1e-3 * fmax(1.0, fabs(exp));
            const char *mark = (diff <= tol) ? "OK" : "MISMATCH";
            if (diff > tol) ok = 0;
            printf("  row=%lld out=%lld  expected=%.6f actual=%.6f diff=%.6g  %s\n",
                   (long long)row, (long long)out_row, exp, act, diff, mark);
        }
    }

    free(expected);
    free(actual);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);

    printf("%s: %s\n\n", label, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

int main(void) {
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) {
        printf("FAIL: ggml_backend_cuda_init(0) returned NULL - no CUDA device found?\n");
        return 1;
    }
    printf("CUDA backend initialized OK.\n\n");

    int failures = 0;

    {
        int8_t weight[8] = { 1, 2, 3, 4,   -1, 0, 5, -2 };
        float  input[4]  = { 10.0f, -4.0f, 2.0f, 1.0f };
        failures += run_case(backend, "no-rotation", 4, 2, 1, weight, input, 0, NULL, NULL);
    }
    // NOTE: the small hand-picked group_size=4 rotation cases from the CPU
    // test are dropped here on purpose - ggml-cuda.cu's own kernel hard-
    // enforces GGML_ASSERT(convrot_group_size == 0 || convrot_group_size ==
    // 256), unlike the CPU path which accepts any power-of-4 group size.
    // Rotation correctness on CUDA is covered by the group256-random case
    // below instead, which uses the one group size the kernel actually
    // supports.
    {
        int8_t weight[4] = { 1, 2, 3, 4 };
        float input[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
        failures += run_case(backend, "zero-row", 4, 1, 1, weight, input, 0, NULL, NULL);
    }
    {
        int8_t weight[8] = { 1, 2, 3, 4,   -1, 0, 5, -2 };
        float  input[4]  = { 10.0f, -4.0f, 2.0f, 1.0f };
        float  weight_scale[2] = { 0.037f, 1.245f };
        float  bias[2] = { 0.5f, -3.25f };
        failures += run_case(backend, "weight_scale-and-bias", 4, 2, 1, weight, input, 0, weight_scale, bias);
    }
    // Case: group_size=256, matching the real production-shaped group size
    // (the only value the CUDA kernel actually accepts, per ggml-cuda.cu's
    // own GGML_ASSERT(convrot_group_size == 0 || convrot_group_size == 256)).
    {
        int64_t k = 256, n = 2, rows = 1;
        int8_t *weight = (int8_t *)malloc((size_t)(k * n));
        float *input = (float *)malloc(sizeof(float) * (size_t)(k * rows));
        srand(42);
        for (int64_t i = 0; i < k * n; ++i) weight[i] = (int8_t)((rand() % 255) - 127);
        for (int64_t i = 0; i < k; ++i) input[i] = ((float)(rand() % 2000) - 1000.0f) / 37.0f;
        failures += run_case(backend, "group256-random", k, n, rows, weight, input, 256, NULL, NULL);
        free(weight);
        free(input);
    }

    ggml_backend_free(backend);

    if (failures == 0) {
        printf("ALL CUDA CASES PASSED\n");
        return 0;
    } else {
        printf("%d CUDA CASE(S) FAILED\n", failures);
        return 1;
    }
}
