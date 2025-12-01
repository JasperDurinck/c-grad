#ifdef __cplusplus
extern "C" {
#endif

#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h> 
#include <stdint.h>

#define MAX_DIMS 8

typedef enum { FLOAT32, FLOAT16, BFLOAT16, INT32, INT64 } TensorType;
typedef enum { CPU, CUDA } Device;

typedef struct Tensor {
    void*       data;
    void*       grad;
    int         ndim;
    int64_t     shape[MAX_DIMS];
    int64_t     stride[MAX_DIMS];
    TensorType  dtype;
    Device      device;
    int         requires_grad;
} Tensor;


size_t dtype_size(TensorType dtype);
int64_t tensor_numel(const Tensor* t);
Tensor* tensor_create_cpu(int ndim, const int64_t* shape, TensorType dtype);
Tensor* tensor_create(int ndim, const int64_t* shape, TensorType dtype, Device dev);
Tensor* tensor_create_scalar(TensorType dtype, Device device);
void tensor_fill(Tensor* t, double value);
void tensor_fill_random(Tensor* t, float min_val, float max_val);
void tensor_free_cpu(Tensor* t);
void tensor_free(Tensor* t);
void tensor_print(const Tensor* t);
void tensor_print_shape(const Tensor* t);
Tensor* tensor_create_scalar(TensorType dtype, Device device);

#endif

#ifdef __cplusplus
}
#endif