#ifndef TENSOR_OPS_CPU_H
#define TENSOR_OPS_CPU_H

#include "tensor.h"
#include <stdint.h> 

// Elementwise addition
void tensor_add_cpu(const Tensor* a, const Tensor* b, Tensor* out);
void tensor_subtract_cpu(const Tensor* a, const Tensor* b, Tensor* out);

// Matrix multiplication
void tensor_matmul_cpu(const Tensor* A, const Tensor* B, Tensor* out);

// Element wise mul / div
void tensor_mul_cpu(const Tensor* a, const Tensor* b, Tensor* out); 
void tensor_div_cpu(const Tensor* a, const Tensor* b, Tensor* out);

// Transpose
void tensor_transpose_cpu(const Tensor* a, Tensor* out);

// Reduce
void tensor_mean_cpu(const Tensor* a, Tensor* out);
void tensor_max_cpu(const Tensor* a, Tensor* out);
void tensor_argmax_cpu(const Tensor* a, Tensor* out);
Tensor* tensor_argmax_dim1_cpu(const Tensor* src); //TODO

void tensor_sum_cpu(const Tensor* a, Tensor* out);

void tensor_exp_cpu(const Tensor* a, Tensor* out);

void tensor_add_bias_cpu(const Tensor* input, const Tensor* bias, Tensor* out);

void tensor_sum_axis_cpu(const Tensor* a, int axis, Tensor* out);

#endif // TENSOR_OPS_CPU_H