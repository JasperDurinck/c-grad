#ifndef TENSOR_CU_H
#define TENSOR_CU_H

#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

Tensor* tensor_create_cuda(int ndim, const int64_t* shape, TensorType dtype);
Tensor* tensor_to_cuda(const Tensor* cpu_tensor);
Tensor* tensor_to_cpu(const Tensor* gpu_tensor);
void tensor_free_cuda(Tensor* t);
void tensor_fill_cuda(Tensor* t, double value);
Tensor* tensor_create_scalar_cuda(TensorType dtype);
void tensor_fill_random_cuda(Tensor* t, float min_val, float max_val);

#ifdef __cplusplus
}
#endif

#endif // TENSOR_CU_H
