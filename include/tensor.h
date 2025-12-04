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
    int is_view;  // (1 == does NOT own data)
} Tensor;


size_t dtype_size(TensorType dtype);
int64_t tensor_numel(const Tensor* t);
Tensor* tensor_create_cpu(int ndim, const int64_t* shape, TensorType dtype);
Tensor* tensor_create(int ndim, const int64_t* shape, TensorType dtype, Device dev);
Tensor* tensor_create_as(const Tensor* src);
Tensor* tensor_create_scalar(TensorType dtype, Device device);
Tensor* tensor_slice(const Tensor* src, int index);
void tensor_copy_slice_cpu(Tensor* dest, const Tensor* src, int dest_index);
void tensor_copy_slice(Tensor* dest, const Tensor* src, int dest_index);
void tensor_fill(Tensor* t, double value);
void tensor_fill_random(Tensor* t, float min_val, float max_val);
void tensor_free_cpu(Tensor* t);
void tensor_free(Tensor* t);
void tensor_print(const Tensor* t);
void tensor_print_shape(const Tensor* t);
Tensor* tensor_create_scalar(TensorType dtype, Device device);
Tensor* tensor_slice_cuda(const Tensor* src, int index);
Tensor* tensor_slice_cpu(const Tensor* src, int index);
Tensor* tensor_slice(const Tensor* src, int index);
Tensor* tensor_reshape(Tensor* src, int new_ndim, const int64_t* new_shape);

#endif

#ifdef __cplusplus
}
#endif