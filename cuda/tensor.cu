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

Tensor* tensor_to_cpu(const Tensor* src) {
    Tensor* cpu_tensor = tensor_create(src->ndim, src->shape, src->dtype, CPU);

    int64_t numel = tensor_numel(src);
    size_t bytes = numel * dtype_size(src->dtype);

    if (src->device == CPU) {
        // Source already on CPU, just copy memory
        memcpy(cpu_tensor->data, src->data, bytes);
    } else if (src->device == CUDA) {
        // Copy from GPU to CPU
        cudaError_t err = cudaMemcpy(cpu_tensor->data, src->data, bytes, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            printf("cudaMemcpy error: %s\n", cudaGetErrorString(err));
        }
    } else {
        printf("Unknown device in tensor_to_cpu\n");
    }

    return cpu_tensor;
}

Tensor* tensor_to_cuda(const Tensor* src) {
    // Create a new GPU tensor
    Tensor* gpu_tensor = tensor_create(src->ndim, src->shape, src->dtype, CUDA);

    int64_t numel = tensor_numel(src);
    size_t bytes = numel * dtype_size(src->dtype);

    if (src->device == CUDA) {
        // Source already on GPU, just copy memory
        cudaMemcpy(gpu_tensor->data, src->data, bytes, cudaMemcpyDeviceToDevice);
    } else if (src->device == CPU) {
        // Copy from CPU to GPU
        cudaError_t err = cudaMemcpy(gpu_tensor->data, src->data, bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            printf("cudaMemcpy error: %s\n", cudaGetErrorString(err));
        }
    } else {
        printf("Unknown device in tensor_to_cuda\n");
    }

    return gpu_tensor;
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

__global__ void tensor_fill_random_kernel(float* data, float* rnd, int64_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = rnd[idx];
    }
}

void tensor_fill_random_cuda(Tensor* t, float min_val, float max_val) {
    int64_t n = tensor_numel(t);

    // 1) Allocate temporary host buffer
    float* h_tmp = (float*)malloc(n * sizeof(float));

    // 2) Fill host buffer with random numbers
    for (int64_t i = 0; i < n; i++) {
        float r = (float)rand() / (float)RAND_MAX;
        h_tmp[i] = min_val + r * (max_val - min_val);
    }

    // 3) Allocate GPU buffer for random values
    float* d_tmp;
    cudaMalloc((void**)&d_tmp, n * sizeof(float));

    // 4) Copy random values to GPU
    cudaMemcpy(d_tmp, h_tmp, n * sizeof(float), cudaMemcpyHostToDevice);

    // 5) Launch kernel to copy into final tensor
    float* d_data = (float*)t->data;

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    tensor_fill_random_kernel<<<blocks, threads>>>(d_data, d_tmp, n);
    cudaDeviceSynchronize();

    // 6) Free temporary buffers
    cudaFree(d_tmp);
    free(h_tmp);
}

#ifdef __cplusplus
}
#endif
