#include "../include/tensor.h"
#include "../include/optimizers.h"
#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

__global__ void sgd_update_kernel(float* w, const float* g, int64_t N, float lr) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        w[idx] -= lr * g[idx];
    }
}

void sgd_update_optimizer_cuda(Optimizer* opt, float lr)
{
    if (!opt) return;

    for (int li = 0; li < opt->n_layers; li++) {
        Layer* layer = opt->layers[li];
        if (!layer) continue;

        for (int wi = 0; wi < layer->n_weights; wi++) {
            Tensor* W = layer->weights[wi];
            if (!W || !W->grad) continue;

            Tensor* G = (Tensor*)W->grad;

            // Number of elements in W
            int64_t N = 1;
            for (int d = 0; d < W->ndim; d++)
                N *= W->shape[d];

            if (N <= 0) continue;

            float* d_w = (float*)W->data; 
            float* d_g = (float*)G->data;  

            // CUDA launch configuration
            int threads = 256;
            int blocks = (N + threads - 1) / threads;

            sgd_update_kernel<<<blocks, threads>>>(d_w, d_g, N, lr);
            cudaDeviceSynchronize();
        }
    }
}

#ifdef __cplusplus
}
#endif
