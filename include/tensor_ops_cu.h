#ifndef TENSOR_OPS_CU_H
#define TENSOR_OPS_CU_H

#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

// GPU versions of tensor operations
void tensor_add_cuda(const Tensor* a, const Tensor* b, Tensor* out);
void tensor_sub_cuda(const Tensor* a, const Tensor* b, Tensor* out);

// GPU version of batched matrix multiplication
void tensor_matmul_cuda(const Tensor* A, const Tensor* B, Tensor* out);

void tensor_mul_cuda(const Tensor* a, const Tensor* b, Tensor* out);
void tensor_div_cuda(const Tensor* a, const Tensor* b, Tensor* out);

// Tanspose
void tensor_transpose_cuda(const Tensor* a, Tensor* out);

// Reduce

void tensor_argmax_cuda(const Tensor* a, Tensor* out);
Tensor* tensor_argmax_dim1_cuda(const Tensor* src); //TODO
void tensor_max_cuda(const Tensor* a, Tensor* out);
void tensor_mean_cuda(const Tensor* a, Tensor* out);
void tensor_sum_cuda(const Tensor* a, Tensor* out);

// Math
void tensor_exp_cuda(const Tensor* a, Tensor* out);

void tensor_add_bias_cuda(const Tensor* input, const Tensor* bias, Tensor* out);

void tensor_sum_axis_cuda(const Tensor* a, int axis, Tensor* out);

#ifdef __cplusplus
}
#endif

#endif // TENSOR_OPS_CU_H