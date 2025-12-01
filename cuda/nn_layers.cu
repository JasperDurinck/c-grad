
#include <stdio.h>
#include <cuda_runtime.h>
#include "../include/tensor.h"
#include "../include/nn_layers.h"
#include <float.h>

#ifdef __cplusplus
extern "C" {
#endif

__global__ void relu_forward_kernel(const float* x, float* y, int64_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = x[idx];
        y[idx] = (v > 0.0f) ? v : 0.0f;
    }
}

void relu_layer_fn_cuda(Layer* layer, Tensor* input) {
    layer->input = input;
    // Allocate output tensor lazily (on GPU)
    if (!layer->output) {
        layer->output = tensor_create(input->ndim, input->shape, input->dtype, input->device);
    }

    int64_t n = tensor_numel(input);
    float* x_data = (float*)input->data;
    float* y_data = (float*)layer->output->data;

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    relu_forward_kernel<<<blocks, threads>>>(x_data, y_data, n);
    cudaDeviceSynchronize();
}

__global__ void relu_backward_kernel(const float* x, const float* grad_out, float* grad_in, int64_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float xi = x[idx];
        grad_in[idx] = (xi > 0.0f) ? grad_out[idx] : 0.0f;
    }
}

void relu_layer_fn_grad_cuda(Layer* layer, Tensor* grad_output) {
    if (!layer->input) {
        fprintf(stderr, "ReLU backward called without forward input!\n");
        exit(1);
    }

    Tensor* X = layer->input;

    // Allocate grad_input on GPU if needed
    if (!layer->grad_input) {
        layer->grad_input = tensor_create(X->ndim, X->shape, X->dtype, X->device);
    }

    int64_t n = tensor_numel(X);

    float* x  = (float*)X->data;
    float* go = (float*)grad_output->data;
    float* gi = (float*)layer->grad_input->data;

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    relu_backward_kernel<<<blocks, threads>>>(x, go, gi, n);
    cudaDeviceSynchronize();
}




#ifdef __cplusplus
}
#endif
