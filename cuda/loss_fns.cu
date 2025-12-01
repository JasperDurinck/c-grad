#include "../include/tensor.h"
#include <math.h>
#include <float.h>


#ifdef __cplusplus
extern "C" {
#endif


__global__ void mse_loss_kernel(const float* y_pred, const float* y_true, float* grad_out, float* loss_accum, int64_t N, int64_t batch) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float diff = y_pred[idx] - y_true[idx];
        grad_out[idx] = 2.0f * diff / batch;  // avg grad per batch
        atomicAdd(loss_accum, diff * diff);   // accumulate total sum
    }
}

float mse_loss_cuda(Tensor* y_pred, Tensor* y_true, Tensor* grad_out) {
    int64_t N = tensor_numel(y_pred);
    int64_t batch = y_pred->shape[0];

    float* d_yp = (float*)y_pred->data;
    float* d_yt = (float*)y_true->data;
    float* d_go = (float*)grad_out->data;

    float h_loss = 0.0f;
    float* d_loss;
    cudaMalloc(&d_loss, sizeof(float));
    cudaMemcpy(d_loss, &h_loss, sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    mse_loss_kernel<<<blocks, threads>>>(d_yp, d_yt, d_go, d_loss, N, batch);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_loss);

    // divide by batch after summing
    h_loss /= batch;

    return h_loss;
}

// BCE


__global__ void bce_loss_kernel(const float* y_pred, const float* y_true, float* grad_out, float* loss_accum, int64_t N, int64_t batch) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float p = y_pred[idx];
        if (p < FLT_EPSILON) p = FLT_EPSILON;
        if (p > 1.0f - FLT_EPSILON) p = 1.0f - FLT_EPSILON;

        float t = y_true[idx];
        grad_out[idx] = (p - t) / (p * (1.0f - p) * batch);

        atomicAdd(loss_accum, - (t * logf(p) + (1.0f - t) * logf(1.0f - p)) / batch);
    }
}


float binary_cross_entropy_loss_cuda(Tensor* y_pred, Tensor* y_true, Tensor* grad_out) {
    int64_t N = tensor_numel(y_pred);
    int64_t batch = y_pred->shape[0];

    float* d_yp = (float*)y_pred->data;
    float* d_yt = (float*)y_true->data;
    float* d_go = (float*)grad_out->data;

    float h_loss = 0.0f;
    float* d_loss;
    cudaMalloc(&d_loss, sizeof(float));
    cudaMemcpy(d_loss, &h_loss, sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    bce_loss_kernel<<<blocks, threads>>>(d_yp, d_yt, d_go, d_loss, N, batch);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_loss);

    return h_loss;
}


#ifdef __cplusplus
}
#endif
