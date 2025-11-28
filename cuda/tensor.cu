#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../include/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

Tensor* tensor_create_cuda(int ndim, const int64_t* shape, TensorType dtype) {
    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    t->ndim = ndim;
    t->dtype = dtype;
    t->device = CUDA;
    t->requires_grad = 1;
    t->grad = NULL;

    memcpy(t->shape, shape, ndim * sizeof(int64_t));

    if (ndim > 0) {
        t->stride[ndim - 1] = 1;
        for (int i = ndim - 2; i >= 0; i--) {
            t->stride[i] = t->stride[i + 1] * t->shape[i + 1];
        }
    }

    int64_t numel = tensor_numel(t);
    cudaMalloc(&t->data, numel * dtype_size(dtype));

    return t;
}

void tensor_free_cuda(Tensor* t) {
    if (!t) return;
    if (t->data) cudaFree(t->data);
    if (t->grad) cudaFree(t->grad);
    free(t);
}

Tensor* tensor_to_cuda(const Tensor* cpu_tensor) {
    Tensor* gpu_tensor = tensor_create_cuda(cpu_tensor->ndim, cpu_tensor->shape, cpu_tensor->dtype);

    int64_t numel = tensor_numel(cpu_tensor);
    size_t bytes = numel * dtype_size(cpu_tensor->dtype);
    cudaMemcpy(gpu_tensor->data, cpu_tensor->data, bytes, cudaMemcpyHostToDevice);

    return gpu_tensor;
}

Tensor* tensor_to_cpu(const Tensor* gpu_tensor) {
    Tensor* cpu_tensor = tensor_create(gpu_tensor->ndim, gpu_tensor->shape, gpu_tensor->dtype, CPU);

    int64_t numel = tensor_numel(gpu_tensor);
    size_t bytes = numel * dtype_size(gpu_tensor->dtype);
    cudaMemcpy(cpu_tensor->data, gpu_tensor->data, bytes, cudaMemcpyDeviceToHost);

    return cpu_tensor;
}

__global__ void fill_kernel(float* data, int64_t total, float value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) data[idx] = value;
}

void tensor_fill_cuda(Tensor* t, double value) {
    int64_t numel = tensor_numel(t);
    int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    fill_kernel<<<blocks, threads>>>((float*)t->data, numel, (float)value);
    cudaDeviceSynchronize();
}

Tensor* tensor_create_scalar_cuda(TensorType dtype) {
    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    if (!t) return NULL;

    // Represent scalar as 1-dim size 1 
    t->ndim = 1;            
    t->dtype = dtype;
    t->device = CUDA;       
    t->requires_grad = 0;
    t->grad = NULL;
    t->shape[0] = 1;
    t->stride[0] = 1;

    int64_t numel = 1;
    size_t bytes = (size_t)numel * dtype_size(dtype);
    cudaError_t err = cudaMalloc(&t->data, bytes);
    if (err != cudaSuccess) {
        free(t);
        return NULL;
    }

    return t;
}



#ifdef __cplusplus
}
#endif
