#ifndef TENSOR_OPS_H
#define TENSOR_OPS_H

#include "tensor.h"
#include <stdint.h> // for int64_t

// Elementwise addition
void tensor_add(const Tensor* a, const Tensor* b, Tensor* out);
void tensor_subtract(const Tensor* a, const Tensor* b, Tensor* out);
void tensor_check_elementwise(const Tensor* a, const Tensor* b, const Tensor* out, const char* op_name);

// Matrix multiplication
void tensor_matmul(const Tensor* A, const Tensor* B, Tensor* out);
Tensor* tensor_matmul_out(const Tensor* A, const Tensor* B);
void check_matmul_shapes(const Tensor* A, const Tensor* B, const Tensor* out);

// Element wise mul / div
void tensor_mul(const Tensor* a, const Tensor* b, Tensor* out); 
void tensor_div(const Tensor* a, const Tensor* b, Tensor* out);

// Transpose Tensor
Tensor* tensor_transpose_out(const Tensor* a);
void tensor_transpose(const Tensor* a, Tensor* out);

// Reduce
void tensor_sum(const Tensor* a, Tensor* out);
void tensor_mean(const Tensor* a, Tensor* out);
void tensor_argmax(const Tensor* a, Tensor* out);
void tensor_max(const Tensor* a, Tensor* out);

void tensor_exp(const Tensor* a, Tensor* out);

void tensor_add_bias(const Tensor* a, const Tensor* b, Tensor* out);

void tensor_sum_axis(const Tensor* a, int axis, Tensor* out);

#endif // TENSOR_OPS_H