#include "../include/tensor.h"
#include "../include/tensor_ops_cpu.h"
#include "../include/tensor_ops_cu.h"
#include <stdio.h>
#include <stdlib.h>  

void tensor_check_elementwise( const Tensor* a, const Tensor* b, const Tensor* out, const char* op_name) {
    // Check dtype
    if (a->dtype != b->dtype || a->dtype != out->dtype) {
        fprintf(stderr, "%s: dtype mismatch\n", op_name);
        exit(1);
    }

    // Check device
    if (a->device != b->device || a->device != out->device) {
        fprintf(stderr, "%s: device mismatch\n", op_name);
        exit(1);
    }

    // Check ndim
    if (a->ndim != b->ndim || a->ndim != out->ndim) {
        fprintf(stderr, "%s: ndim mismatch\n", op_name);
        exit(1);
    }

    // Check shapes
    for (int i = 0; i < a->ndim; i++) {
        if (a->shape[i] != b->shape[i] || a->shape[i] != out->shape[i]) {
            fprintf(stderr, "%s: shape mismatch at dim %d\n", op_name, i);
            exit(1);
        }
    }
}

void check_matmul_shapes(const Tensor* A, const Tensor* B, const Tensor* out) {
    // dtype check
    if (A->dtype != FLOAT32 || B->dtype != FLOAT32 || (out && out->dtype != FLOAT32)) {
        fprintf(stderr, "Only float32 matmul supported\n");
        exit(1);
    }

    // ndim check
    if (A->ndim != B->ndim || (out && A->ndim != out->ndim)) {
        fprintf(stderr, "Tensor ndim mismatch\n");
        exit(1);
    }

    int batch_dims = A->ndim - 2;
    for (int i = 0; i < batch_dims; i++) {
        if (A->shape[i] != B->shape[i] || (out && A->shape[i] != out->shape[i])) {
            fprintf(stderr, "Batch dimension mismatch\n");
            exit(1);
        }
    }

    int64_t M = A->shape[A->ndim - 2];
    int64_t K = A->shape[A->ndim - 1];
    int64_t K_B = B->shape[B->ndim - 2];
    int64_t N = B->shape[B->ndim - 1];

    if (K != K_B || (out && (M != out->shape[out->ndim - 2] || N != out->shape[out->ndim - 1]))) {
        fprintf(stderr, "Matmul inner/outer dimension mismatch\n");
        exit(1);
    }

}

Tensor* tensor_matmul_out(const Tensor* A, const Tensor* B) {
    check_matmul_shapes(A, B, NULL);

    int64_t batch_dims = A->ndim - 2;
    int64_t M = A->shape[A->ndim - 2];
    int64_t N = B->shape[B->ndim - 1];

    int64_t out_shape[8];  // MAX_DIMS
    for (int i = 0; i < batch_dims; i++)
        out_shape[i] = A->shape[i];

    out_shape[batch_dims]     = M;
    out_shape[batch_dims + 1] = N;

    return tensor_create(A->ndim, out_shape, A->dtype, A->device);
}

Tensor* tensor_transpose_out(const Tensor* a) {
    int64_t shape[MAX_DIMS];
    for (int i = 0; i < a->ndim; i++) shape[i] = a->shape[i];

    // swap last two dims
    int64_t tmp = shape[a->ndim - 2];
    shape[a->ndim - 2] = shape[a->ndim - 1];
    shape[a->ndim - 1] = tmp;

    Tensor* out = tensor_create(a->ndim, shape, a->dtype, a->device);
    return out;
}

void tensor_add(const Tensor* a, const Tensor* b, Tensor* out) {
    tensor_check_elementwise(a, b, out, "tensor_add");

    switch (a->device) {
        case CPU:
            tensor_add_cpu(a, b, out);
            break;

        case CUDA:
            tensor_add_cuda(a, b, out);
            break;

        default:
            fprintf(stderr, "tensor_add: unsupported device\n");
            exit(1);
    }
}

void tensor_subtract(const Tensor* a, const Tensor* b, Tensor* out) {
    tensor_check_elementwise(a, b, out, "tensor_subtract");

    switch (a->device) {
        case CPU:
            tensor_subtract_cpu(a, b, out);
            break;

        case CUDA:
            tensor_sub_cuda(a, b, out);  // added later
            break;

        default:
            fprintf(stderr, "tensor_subtract: unsupported device\n");
            exit(1);
    }
}

void tensor_matmul(const Tensor* A, const Tensor* B, Tensor* out) {
    switch (A->device) {
        case CPU:
            tensor_matmul_cpu(A, B, out);
            break;
        case CUDA:
            tensor_matmul_cuda(A, B, out);
            break;
        default:
            fprintf(stderr, "tensor_matmul: unsupported device\n");
            exit(1);
    }
}

void tensor_mul(const Tensor* A, const Tensor* B, Tensor* out) {
    switch (A->device) {
        case CPU:
            tensor_mul_cpu(A, B, out);
            break;
        case CUDA:
            tensor_mul_cuda(A, B, out);
            break;
        default:
            fprintf(stderr, "tensor_mul: unsupported device\n");
            exit(1);
    }
}

void tensor_div(const Tensor* A, const Tensor* B, Tensor* out) {
    switch (A->device) {
        case CPU:
            tensor_div_cpu(A, B, out);
            break;
        case CUDA:
            tensor_div_cuda(A, B, out);
            break;
        default:
            fprintf(stderr, "tensor_div: unsupported device\n");
            exit(1);
    }
}

void tensor_transpose(const Tensor* a, Tensor* out) {
    switch (a->device) {
        case CPU:
            tensor_transpose_cpu(a, out);
            break;
        case CUDA:
            tensor_transpose_cuda(a, out);
            break;
        default:
            fprintf(stderr, "tensor_transpose: unsupported device\n");
            exit(1);
    }
}

void tensor_mean(const Tensor* a, Tensor* out) {
    switch (a->device) {
        case CPU:
            tensor_mean_cpu(a, out);
            break;
        case CUDA:
            tensor_mean_cuda(a, out);
            break;
        default:
            fprintf(stderr, "tensor_mean: unsupported device\n");
            exit(1);
    }
}

void tensor_sum(const Tensor* a, Tensor* out) {
    switch (a->device) {
        case CPU:
            tensor_sum_cpu(a, out);
            break;
        case CUDA:
            tensor_sum_cuda(a, out);
            break;
        default:
            fprintf(stderr, "tensor_sum: unsupported device\n");
            exit(1);
    }
}

void tensor_sum_axis(const Tensor* a, int axis, Tensor* out){
    switch (a->device) {
        case CPU:
            tensor_sum_axis_cpu(a, axis, out);
            break;
        case CUDA:
            tensor_sum_axis_cuda(a, axis, out);
            break;
        default:
            fprintf(stderr, "tensor_sum: unsupported device\n");
            exit(1);
    }
}

void tensor_argmax(const Tensor* a, Tensor* out) {
    switch (a->device) {
        case CPU:
            tensor_argmax_cpu(a, out);
            break;
        case CUDA:
            tensor_argmax_cuda(a, out);
            break;
        default:
            fprintf(stderr, "tensor_argmax: unsupported device\n");
            exit(1);
    }
}

Tensor* tensor_argmax_dim1(const Tensor* src) {
    switch (src->device) {
        case CPU:
            tensor_argmax_dim1_cpu(src);
            break;
        case CUDA:
            tensor_argmax_dim1_cuda(src);
            break;
        default:
            fprintf(stderr, "tensor_argmax: unsupported device\n");
            exit(1);
    }
}

void tensor_max(const Tensor* a, Tensor* out) {
    switch (a->device) {
        case CPU:
            tensor_max_cpu(a, out);
            break;
        case CUDA:
            tensor_max_cuda(a, out);
            break;
        default:
            fprintf(stderr, "tensor_max: unsupported device\n");
            exit(1);
    }
}

void tensor_exp(const Tensor* a, Tensor* out) {
    switch (a->device) {
        case CPU:
            tensor_exp_cpu(a, out);
            break;
        case CUDA:
            tensor_exp_cuda(a, out);
            break;
        default:
            fprintf(stderr, "tensor_max: unsupported device\n");
            exit(1);
    }
}

void tensor_add_bias(const Tensor* a, const Tensor* b, Tensor* out) {
    switch (a->device) {
        case CPU:
            tensor_add_bias_cpu(a, b, out);
            break;
        case CUDA:
            tensor_add_bias_cuda(a, b, out);
            break;
        default:
            fprintf(stderr, "tensor_add_bias: unsupported device\n");
            exit(1);
    }
}

