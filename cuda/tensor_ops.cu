#include <stdio.h>
#include <cuda_runtime.h>
#include "../include/tensor.h"
#include <float.h>

#ifdef __cplusplus
extern "C" {
#endif

// Element-wise addition kernel
__global__ void tensor_add_kernel(float* A, float* B, float* C, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) C[idx] = A[idx] + B[idx];
}

// Element-wise subtraction kernel
__global__ void tensor_sub_kernel(float* A, float* B, float* C, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) C[idx] = A[idx] - B[idx];
}

void tensor_add_cuda(const Tensor* a, const Tensor* b, Tensor* out) {
    int64_t numel = tensor_numel(a);

    float *d_a = (float*)a->data;
    float *d_b = (float*)b->data;
    float *d_out = (float*)out->data;

    int threads = 256;
    int blocks = (numel + threads - 1) / threads;

    tensor_add_kernel<<<blocks, threads>>>(d_a, d_b, d_out, numel);
    cudaDeviceSynchronize(); // wait for kernel
}

void tensor_sub_cuda(const Tensor* a, const Tensor* b, Tensor* out) {
    int64_t numel = tensor_numel(a);

    float *d_a = (float*)a->data;
    float *d_b = (float*)b->data;
    float *d_out = (float*)out->data;

    int threads = 256;
    int blocks = (numel + threads - 1) / threads;

    tensor_sub_kernel<<<blocks, threads>>>(d_a, d_b, d_out, numel);
    cudaDeviceSynchronize(); //  wait for kernel
}


// Element-wise multiply kernel
__global__ void tensor_mul_kernel(float* A, float* B, float* C, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) C[idx] = A[idx] * B[idx];
}

// Element-wise divide kernel
__global__ void tensor_div_kernel(float* A, float* B, float* C, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) C[idx] = A[idx] / B[idx]; 
}

void tensor_mul_cuda(const Tensor* a, const Tensor* b, Tensor* out) {
    int64_t numel = tensor_numel(a);

    float *d_a = (float*)a->data;
    float *d_b = (float*)b->data;
    float *d_out = (float*)out->data;

    int threads = 256;
    int blocks = (numel + threads - 1) / threads;

    tensor_mul_kernel<<<blocks, threads>>>(d_a, d_b, d_out, numel);
    cudaDeviceSynchronize();
}

void tensor_div_cuda(const Tensor* a, const Tensor* b, Tensor* out) {
    int64_t numel = tensor_numel(a);

    float *d_a = (float*)a->data;
    float *d_b = (float*)b->data;
    float *d_out = (float*)out->data;

    int threads = 256;
    int blocks = (numel + threads - 1) / threads;

    tensor_div_kernel<<<blocks, threads>>>(d_a, d_b, d_out, numel);
    cudaDeviceSynchronize();
}


// Batched matmul kernel for float32 tensors
// Supports both 2D (batch x K) and 3D (batch x M x K) inputs
__global__ void tensor_matmul_kernel_generic(const float* A, const float* B, float* C,
                                             int batch, int M, int K, int N,
                                             int B_has_batch) {
    int b = blockIdx.z;                           // batch index
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (b < batch && row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            // If B has batch dimension, index like 3D, otherwise like 2D
            float b_val = B_has_batch ? B[b*K*N + k*N + col] : B[k*N + col];
            sum += A[b*M*K + row*K + k] * b_val;
        }
        C[b*M*N + row*N + col] = sum;
    }
}

// Wrapper to launch CUDA kernel
void tensor_matmul_cuda(const Tensor* A, const Tensor* B, Tensor* out) {
    // Determine shapes
    int batch = A->shape[0];
    int M = (A->ndim == 3) ? A->shape[1] : 1;
    int K = (A->ndim == 3) ? A->shape[2] : A->shape[1];
    int N = (B->ndim == 3) ? B->shape[2] : B->shape[1];

    int B_has_batch = (B->ndim == 3) ? 1 : 0;

    float* d_A = (float*)A->data;
    float* d_B = (float*)B->data;
    float* d_out = (float*)out->data;

    // Thread/block configuration
    dim3 threadsPerBlock(16, 16, 1);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (M + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   batch);

    tensor_matmul_kernel_generic<<<numBlocks, threadsPerBlock>>>(
        d_A, d_B, d_out, batch, M, K, N, B_has_batch
    );

    cudaDeviceSynchronize();
}

__global__ void tensor_transpose_kernel(
    const float* A, float* B,
    int batch, int M, int N)
{
    int b = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (b < batch && row < M && col < N) {
        int in_idx  = b * (M*N) + row * N + col;
        int out_idx = b * (M*N) + col * M + row;
        B[out_idx] = A[in_idx];
    }
}

void tensor_transpose_cuda(const Tensor* a, Tensor* out) {
    int64_t batch = 1;
    for (int i = 0; i < a->ndim - 2; i++)
        batch *= a->shape[i];

    int64_t M = a->shape[a->ndim - 2];
    int64_t N = a->shape[a->ndim - 1];

    float* d_a   = (float*)a->data;
    float* d_out = (float*)out->data;

    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16,
                (M + 15) / 16,
                 batch);

    tensor_transpose_kernel<<<blocks, threads>>>(d_a, d_out, batch, M, N);
    cudaDeviceSynchronize();
}

// -------- Reduction Kernels -----------

__global__ void tensor_sum_kernel(float* input, float* output, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];
}

__global__ void tensor_max_kernel(float* input, float* output, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < n) ? input[idx] : -FLT_MAX;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && sdata[tid + s] > sdata[tid]) sdata[tid] = sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];
}

__global__ void tensor_argmax_kernel(float* input, int64_t* output, int n) {
    extern __shared__ float sdata[];
    __shared__ int64_t sidx[1024]; // max threads per block

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < n) ? input[idx] : -FLT_MAX;
    sidx[tid] = idx;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && sdata[tid + s] > sdata[tid]) {
            sdata[tid] = sdata[tid + s];
            sidx[tid] = sidx[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = sidx[0];
}

// Wrappers

void tensor_sum_cuda(const Tensor* a, Tensor* out) {
    int64_t n = tensor_numel(a);
    float* d_in = (float*)a->data;

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    if (blocks == 0) blocks = 1;

    float* d_intermediate = NULL;
    cudaMalloc(&d_intermediate, blocks * sizeof(float));

    tensor_sum_kernel<<<blocks, threads, threads * sizeof(float)>>>(d_in, d_intermediate, (int)n);
    cudaDeviceSynchronize();

    // copy intermediate back to host and finish
    float* h_intermediate = (float*)malloc(blocks * sizeof(float));
    cudaMemcpy(h_intermediate, d_intermediate, blocks * sizeof(float), cudaMemcpyDeviceToHost);

    float sum = 0.0f;
    for (int i = 0; i < blocks; ++i) sum += h_intermediate[i];

    // write result to out: if out is on device, copy; if on host, write directly
    if (out->device == CUDA) {
        cudaMemcpy(out->data, &sum, sizeof(float), cudaMemcpyHostToDevice);
    } else {
        float* h_out = (float*)out->data;
        h_out[0] = sum;
    }

    free(h_intermediate);
    cudaFree(d_intermediate);
}

void tensor_mean_cuda(const Tensor* a, Tensor* out) {
    int64_t n = tensor_numel(a);
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    if (blocks == 0) blocks = 1;

    float* d_in = (float*)a->data;
    float* d_intermediate = NULL;
    cudaMalloc(&d_intermediate, blocks * sizeof(float));

    tensor_sum_kernel<<<blocks, threads, threads * sizeof(float)>>>(d_in, d_intermediate, (int)n);
    cudaDeviceSynchronize();

    float* h_intermediate = (float*)malloc(blocks * sizeof(float));
    cudaMemcpy(h_intermediate, d_intermediate, blocks * sizeof(float), cudaMemcpyDeviceToHost);

    float sum = 0.0f;
    for (int i = 0; i < blocks; ++i) sum += h_intermediate[i];
    float mean = sum / (float)n;

    if (out->device == CUDA) {
        cudaMemcpy(out->data, &mean, sizeof(float), cudaMemcpyHostToDevice);
    } else {
        float* h_out = (float*)out->data;
        h_out[0] = mean;
    }

    free(h_intermediate);
    cudaFree(d_intermediate);
}

void tensor_max_cuda(const Tensor* a, Tensor* out) {
    int64_t n = tensor_numel(a);
    float* d_in = (float*)a->data;

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    if (blocks == 0) blocks = 1;

    float* d_intermediate = NULL;
    cudaMalloc(&d_intermediate, blocks * sizeof(float));

    tensor_max_kernel<<<blocks, threads, threads * sizeof(float)>>>(d_in, d_intermediate, (int)n);
    cudaDeviceSynchronize();

    float* h_intermediate = (float*)malloc(blocks * sizeof(float));
    cudaMemcpy(h_intermediate, d_intermediate, blocks * sizeof(float), cudaMemcpyDeviceToHost);

    // find max on host
    float max_val = h_intermediate[0];
    for (int i = 1; i < blocks; ++i)
        if (h_intermediate[i] > max_val) max_val = h_intermediate[i];

    if (out->device == CUDA) {
        cudaMemcpy(out->data, &max_val, sizeof(float), cudaMemcpyHostToDevice);
    } else {
        float* h_out = (float*)out->data;
        h_out[0] = max_val;
    }

    free(h_intermediate);
    cudaFree(d_intermediate);
}

void tensor_argmax_cuda(const Tensor* a, Tensor* out) {
    int64_t n = tensor_numel(a);
    float* d_in = (float*)a->data;

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    if (blocks == 0) blocks = 1;

    int64_t* d_intermediate = NULL;
    cudaMalloc(&d_intermediate, blocks * sizeof(int64_t));

    tensor_argmax_kernel<<<blocks, threads, threads * sizeof(float)>>>(d_in, d_intermediate, (int)n);
    cudaDeviceSynchronize();

    // copy block indices to host
    int64_t* h_intermediate = (int64_t*)malloc(blocks * sizeof(int64_t));
    cudaMemcpy(h_intermediate, d_intermediate, blocks * sizeof(int64_t), cudaMemcpyDeviceToHost);


    float* h_all = (float*)malloc(n * sizeof(float));
    cudaMemcpy(h_all, d_in, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Final argmax on host
    int64_t max_idx = h_intermediate[0];
    float max_val = h_all[max_idx];
    for (int i = 1; i < blocks; ++i) {
        int64_t idx = h_intermediate[i];
        float val = h_all[idx];
        if (val > max_val) {
            max_val = val;
            max_idx = idx;
        }
    }

    if (out->device == CUDA) {
        cudaMemcpy(out->data, &max_idx, sizeof(int64_t), cudaMemcpyHostToDevice);
    } else {
        int64_t* h_out = (int64_t*)out->data;
        h_out[0] = max_idx;
    }

    free(h_intermediate);
    free(h_all);
    cudaFree(d_intermediate);
}

__global__ void exp_kernel(float* a, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = expf(a[idx]);
}

void tensor_exp_cuda(const Tensor* a, Tensor* out) {
    int64_t n = tensor_numel(a);
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    exp_kernel<<<blocks, threads>>>((float*)a->data, (float*)out->data, (int)n);
    cudaDeviceSynchronize();
}

// CUDA kernel, add 1D bias to each row of 2D tensor
__global__ void tensor_add_bias_kernel(float* input, const float* bias, float* out, int batch_size, int out_features) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < batch_size && col < out_features) {
        out[row * out_features + col] = input[row * out_features + col] + bias[col];
    }
}

// wrapper function
void tensor_add_bias_cuda(const Tensor* input, const Tensor* bias, Tensor* out) {
    if (input->ndim != 2 || bias->ndim != 1 || out->ndim != 2) {
        fprintf(stderr, "tensor_add_bias_cuda: input must be 2D, bias 1D, out 2D\n");
        exit(1);
    }

    int64_t batch_size = input->shape[0];
    int64_t out_features = input->shape[1];

    if (bias->shape[0] != out_features) {
        fprintf(stderr, "tensor_add_bias_cuda: bias size %lld does not match input second dim %lld\n",
                (long long)bias->shape[0], (long long)out_features);
        exit(1);
    }

    float* d_input = (float*)input->data;
    float* d_bias = (float*)bias->data;
    float* d_out = (float*)out->data;

    dim3 blockDim(16, 16); // 16x16 threads
    dim3 gridDim((out_features + blockDim.x - 1) / blockDim.x,
                 (batch_size + blockDim.y - 1) / blockDim.y);

    tensor_add_bias_kernel<<<gridDim, blockDim>>>(d_input, d_bias, d_out, batch_size, out_features);
    cudaDeviceSynchronize();
}


__global__ void tensor_sum_axis0_kernel(const float* __restrict__ a,
                                        float* __restrict__ out,
                                        int64_t N, int64_t M)
{
    // One thread per feature index j in [0, M)
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= M) return;

    float sum = 0.0f;
    for (int64_t i = 0; i < N; i++) {
        sum += a[i * M + j];
    }
    out[j] = sum;
}

__global__ void tensor_sum_axis1_kernel(const float* __restrict__ a,
                                        float* __restrict__ out,
                                        int64_t N, int64_t M)
{
    // One thread per batch index i in [0, N)
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float sum = 0.0f;
    const float* row = a + i * M;
    for (int64_t j = 0; j < M; j++) {
        sum += row[j];
    }
    out[i] = sum;
}

void tensor_sum_axis_cuda(const Tensor* a, int axis, Tensor* out)
{
    int64_t N = a->shape[0];
    int64_t M = a->shape[1];

    const float* d_a = (const float*)a->data;
    float* d_out = (float*)out->data;

    int threads = 256;

    if (axis == 0) {
        // sum over rows -> output length M
        int blocks = (M + threads - 1) / threads;
        tensor_sum_axis0_kernel<<<blocks, threads>>>(d_a, d_out, N, M);
    }
    else if (axis == 1) {
        // sum over cols -> output length N
        int blocks = (N + threads - 1) / threads;
        tensor_sum_axis1_kernel<<<blocks, threads>>>(d_a, d_out, N, M);
    }
    else {
        fprintf(stderr, "tensor_sum_axis_cuda: unsupported axis %d\n", axis);
        exit(1);
    }

    cudaDeviceSynchronize();
}


// CUDA kernel for argmax along dim=1
__global__ void tensor_argmax_dim1_kernel(const float* src, int64_t* out, int64_t rows, int64_t cols) {
    int64_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    const float* row_ptr = src + row * cols;
    int64_t max_idx = 0;
    float max_val = row_ptr[0];

    for (int64_t j = 1; j < cols; j++) {
        if (row_ptr[j] > max_val) {
            max_val = row_ptr[j];
            max_idx = j;
        }
    }

    out[row] = max_idx;
}

// Wrapper function
Tensor* tensor_argmax_dim1_cuda(const Tensor* src) {
    if (src->ndim != 2) {
        fprintf(stderr, "tensor_argmax_dim1_cuda: only 2D tensors supported\n");
        exit(1);
    }

    int64_t rows = src->shape[0];
    int64_t cols = src->shape[1];

    int64_t shape[1]; 
    shape[0] = rows;

    Tensor* out = tensor_create(1, shape, INT64, CUDA);

    int threads = 256;
    int blocks = (rows + threads - 1) / threads;

    tensor_argmax_dim1_kernel<<<blocks, threads>>>(
        (const float*)src->data,
        (int64_t*)out->data,
        rows,
        cols
    );
    cudaDeviceSynchronize();

    return out;
}

#ifdef __cplusplus
}
#endif
