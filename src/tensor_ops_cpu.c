#include "../include/tensor.h"
#include "../include/tensor_ops.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>  

void tensor_add_cpu(const Tensor* a, const Tensor* b, Tensor* out) {
    tensor_check_elementwise(a, b, out, "tensor_add");

    int64_t n = tensor_numel(a);
    float* pa = a->data; float* pb = b->data; float* po = out->data;
    for (int64_t i = 0; i < n; i++) po[i] = pa[i] + pb[i];
}

void tensor_subtract_cpu(const Tensor* a, const Tensor* b, Tensor* out) {
    tensor_check_elementwise(a, b, out,"tensor_subtract");

    int64_t n = tensor_numel(a);
    float* pa = a->data; float* pb = b->data; float* po = out->data;
    for (int64_t i = 0; i < n; i++) po[i] = pa[i] - pb[i];
}

void tensor_mul_cpu(const Tensor* a, const Tensor* b, Tensor* out) {
    tensor_check_elementwise(a, b, out, "tensor_multiply");

    int64_t n = tensor_numel(a);
    float* pa = a->data;
    float* pb = b->data;
    float* po = out->data;

    for (int64_t i = 0; i < n; i++)
        po[i] = pa[i] * pb[i];
}

void tensor_div_cpu(const Tensor* a, const Tensor* b, Tensor* out) {
    tensor_check_elementwise(a, b, out, "tensor_divide");

    int64_t n = tensor_numel(a);
    float* pa = a->data;
    float* pb = b->data;
    float* po = out->data;

    for (int64_t i = 0; i < n; i++)
        po[i] = pa[i] / pb[i];  
}

void tensor_matmul_cpu(const Tensor* A, const Tensor* B, Tensor* out) {
    check_matmul_shapes(A, B, out);

    int64_t batch_size = 1;
    int64_t M = A->shape[A->ndim - 2];
    int64_t K = A->shape[A->ndim - 1];
    int64_t N = B->shape[B->ndim - 1];

    for (int i = 0; i < A->ndim - 2; i++) batch_size *= A->shape[i];

    float* a = A->data;
    float* b = B->data;
    float* c = out->data;

    for (int64_t bidx = 0; bidx < batch_size; bidx++) {
        int64_t a_off = bidx * M * K;
        int64_t b_off = bidx * K * N;
        int64_t c_off = bidx * M * N;
        for (int64_t i = 0; i < M; i++)
            for (int64_t j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int64_t p = 0; p < K; p++)
                    sum += a[a_off + i * K + p] * b[b_off + p * N + j];
                c[c_off + i * N + j] = sum;
            }
    }
}

void tensor_transpose_cpu(const Tensor* a, Tensor* out) {
    if (a->ndim < 2) {
        fprintf(stderr, "tensor_transpose: need at least 2 dims\n");
        exit(1);
    }

    int64_t batch = 1;
    for (int i = 0; i < a->ndim - 2; i++)
        batch *= a->shape[i];

    int64_t M = a->shape[a->ndim - 2];
    int64_t N = a->shape[a->ndim - 1];

    float* pa = a->data;
    float* po = out->data;

    for (int64_t b = 0; b < batch; b++) {
        int64_t in_off  = b * M * N;
        int64_t out_off = b * M * N;
        for (int64_t i = 0; i < M; i++)
            for (int64_t j = 0; j < N; j++)
                po[out_off + j * M + i] = pa[in_off + i * N + j];
    }
}

void tensor_mean_cpu(const Tensor* a, Tensor* out) {
    int64_t n = tensor_numel(a);
    float s = 0.0f;
    float* pa = (float*)a->data;
    for (int64_t i = 0; i < n; i++) s += pa[i];

    float* po = (float*)out->data;
    po[0] = s / n;
}

void tensor_max_cpu(const Tensor* a, Tensor* out) {
    float* pa = (float*)a->data;
    int64_t n = tensor_numel(a);
    float max_val = pa[0];

    for (int64_t i = 1; i < n; i++)
        if (pa[i] > max_val) max_val = pa[i];

    float* po = (float*)out->data;
    po[0] = max_val;
}

void tensor_argmax_cpu(const Tensor* a, Tensor* out) {
    float* pa = (float*)a->data;
    int64_t n = tensor_numel(a);
    int64_t max_idx = 0;
    float max_val = pa[0];

    for (int64_t i = 1; i < n; i++)
        if (pa[i] > max_val) {
            max_val = pa[i];
            max_idx = i;
        }

    int64_t* po = (int64_t*)out->data;
    po[0] = max_idx;
}

Tensor* tensor_argmax_dim1_cpu(const Tensor* src) {
    if (src->ndim != 2) {
        fprintf(stderr, "tensor_argmax_dim1: only 2D tensors supported\n");
        exit(1);
    }

    int64_t rows = src->shape[0];
    int64_t cols = src->shape[1];

    Tensor* out = tensor_create(1, (int64_t[]){rows}, INT64, CPU);

    float* data = (float*)src->data;
    int64_t* out_data = (int64_t*)out->data;

    for (int64_t i = 0; i < rows; i++) {
        int64_t max_idx = 0;
        float max_val = data[i * cols];
        for (int64_t j = 1; j < cols; j++) {
            if (data[i * cols + j] > max_val) {
                max_val = data[i * cols + j];
                max_idx = j;
            }
        }
        out_data[i] = max_idx;
    }

    return out;
}

void tensor_sum_cpu(const Tensor* a, Tensor* out) {
    int64_t n = tensor_numel(a);
    float s = 0.0f;
    float* pa = (float*)a->data;

    for (int64_t i = 0; i < n; i++)
        s += pa[i];

    float* po = (float*)out->data;
    po[0] = s;
}

// Sum along a specific axis (2D tensors only for now)
void tensor_sum_axis_cpu(const Tensor* a, int axis, Tensor* out) {
    int64_t N = a->shape[0];  // batch
    int64_t M = a->shape[1];  // features

    const float* pa = (float*)a->data;
    float* po = (float*)out->data;

    if (axis == 0) {
        // Sum over batch -> out shape [M]
        for (int64_t j = 0; j < M; j++) po[j] = 0.0f;

        for (int64_t i = 0; i < N; i++) {
            const float* row = pa + i*M;
            for (int64_t j = 0; j < M; j++) {
                po[j] += row[j];
            }
        }
    }
    else if (axis == 1) {
        // Sum over features -> out shape [N]
        for (int64_t i = 0; i < N; i++) {
            float s = 0.0f;
            for (int64_t j = 0; j < M; j++) s += pa[i*M + j];
            po[i] = s;
        }
    }
    else {
        fprintf(stderr, "tensor_sum_axis_cpu: unsupported axis %d\n", axis);
        exit(1);
    }
}

void tensor_check_unary(const Tensor* X, const Tensor* out, const char* op_name) {
    if (tensor_numel(X) != tensor_numel(out)) {
        fprintf(stderr, "%s: size mismatch\n", op_name);
        exit(1);
    }
}

void tensor_exp_cpu(const Tensor* X, Tensor* out) {
    tensor_check_unary(X, out, "tensor_exp");

    int64_t n = tensor_numel(X);
    float* px = X->data;
    float* po = out->data;

    for (int64_t i = 0; i < n; i++)
        po[i] = expf(px[i]);
}

void tensor_add_bias_cpu(const Tensor* input, const Tensor* bias, Tensor* out) {

    if (input->ndim != 2 || bias->ndim != 1 || out->ndim != 2) {
        fprintf(stderr, "tensor_add_bias_cpu: input must be 2D, bias 1D, out 2D\n");
        exit(1);
    }
    int64_t batch_size = input->shape[0];
    int64_t out_features = input->shape[1];

    if (bias->shape[0] != out_features) {
        fprintf(stderr, "tensor_add_bias_cpu: bias size %lld does not match input second dim %lld\n",
                (long long)bias->shape[0], (long long)out_features);
        exit(1);
    }

    float* pin = input->data;
    float* pb = bias->data;
    float* po = out->data;

    for (int64_t i = 0; i < batch_size; i++) {
        for (int64_t j = 0; j < out_features; j++) {
            po[i*out_features + j] = pin[i*out_features + j] + pb[j];
        }
    }
}