
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

// cnn


__global__ void conv2d_forward_kernel(
    const float* x, const float* w, const float* b,
    float* y,
    int N, int C_in, int H_in, int W_in,
    int C_out, int kH, int kW,
    int H_out, int W_out
) {
    int n = blockIdx.z;
    int co = blockIdx.y;
    int ho = blockIdx.x / W_out;
    int wo = blockIdx.x % W_out;

    if(n >= N || co >= C_out || ho >= H_out || wo >= W_out) return;

    float sum = 0.0f;
    for(int ci = 0; ci < C_in; ci++) {
        for(int kh = 0; kh < kH; kh++) {
            for(int kw = 0; kw < kW; kw++) {
                int h_in = ho + kh;
                int w_in = wo + kw;
                if(h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in){
                    int x_idx = n*C_in*H_in*W_in + ci*H_in*W_in + h_in*W_in + w_in;
                    int w_idx = co*C_in*kH*kW + ci*kH*kW + kh*kW + kw;
                    sum += x[x_idx] * w[w_idx];
                }
            }
        }
    }
    if(b) sum += b[co];
    int y_idx = n*C_out*H_out*W_out + co*H_out*W_out + ho*W_out + wo;
    y[y_idx] = sum;
}

void conv2d_layer_forward_cuda(Layer* layer, Tensor* input) {
    layer->input = input;

    Tensor* W = layer->weights[0];
    Tensor* b = (layer->n_weights > 1) ? layer->weights[1] : NULL;

    int N = input->shape[0];
    int C_in = input->shape[1];
    int H_in = input->shape[2];
    int W_in = input->shape[3];

    int C_out = W->shape[0];
    int kH = W->shape[2];
    int kW = W->shape[3];

    int stride = 1;
    int pad = 0;

    int H_out = (H_in + 2*pad - kH)/stride + 1;
    int W_out = (W_in + 2*pad - kW)/stride + 1;

    // allocate output  
    if(!layer->output){
        int64_t out_shape[4] = {N, C_out, H_out, W_out};
        layer->output = tensor_create(4, out_shape, input->dtype, input->device);
        tensor_fill(layer->output, 0.0f);
    }

    // start CUDA kernel
    dim3 threads(1);
    dim3 blocks(H_out*W_out, C_out, N);  // mapping

    conv2d_forward_kernel<<<blocks, threads>>>(
        (float*)input->data, (float*)W->data, b ? (float*)b->data : NULL,
        (float*)layer->output->data,
        N, C_in, H_in, W_in,
        C_out, kH, kW,
        H_out, W_out
    );

    cudaDeviceSynchronize();
}


__global__ void conv2d_backward_input_kernel(
    const float* grad_out, const float* W, float* grad_input,
    int N, int C_in, int H_in, int W_in,
    int C_out, int kH, int kW,
    int H_out, int W_out
) {
    int n = blockIdx.z;
    int ci = blockIdx.y;
    int hi = blockIdx.x / W_in;
    int wi = blockIdx.x % W_in;

    if(n >= N || ci >= C_in || hi >= H_in || wi >= W_in) return;

    float sum = 0.0f;
    for(int co = 0; co < C_out; co++){
        for(int kh = 0; kh < kH; kh++){
            for(int kw = 0; kw < kW; kw++){
                int ho = hi - kh;
                int wo = wi - kw;
                if(ho >= 0 && ho < H_out && wo >= 0 && wo < W_out){
                    int go_idx = n*C_out*H_out*W_out + co*H_out*W_out + ho*W_out + wo;
                    int w_idx = co*C_in*kH*kW + ci*kH*kW + kh*kW + kw;
                    sum += grad_out[go_idx] * W[w_idx];
                }
            }
        }
    }
    int gx_idx = n*C_in*H_in*W_in + ci*H_in*W_in + hi*W_in + wi;
    grad_input[gx_idx] = sum;
}

__global__ void conv2d_backward_weight_kernel(
    const float* X, const float* grad_out, float* grad_W,
    int N, int C_in, int H_in, int W_in,
    int C_out, int kH, int kW,
    int H_out, int W_out
) {
    int co = blockIdx.z;
    int ci = blockIdx.y;
    int kh = blockIdx.x / kW;
    int kw = blockIdx.x % kW;

    if(co >= C_out || ci >= C_in || kh >= kH || kw >= kW) return;

    float sum = 0.0f;
    for(int n = 0; n < N; n++){
        for(int ho = 0; ho < H_out; ho++){
            for(int wo = 0; wo < W_out; wo++){
                int hi = ho + kh;
                int wi = wo + kw;
                if(hi >= 0 && hi < H_in && wi >= 0 && wi < W_in){
                    int x_idx = n*C_in*H_in*W_in + ci*H_in*W_in + hi*W_in + wi;
                    int go_idx = n*C_out*H_out*W_out + co*H_out*W_out + ho*W_out + wo;
                    sum += X[x_idx] * grad_out[go_idx];
                }
            }
        }
    }
    int w_idx = co*C_in*kH*kW + ci*kH*kW + kh*kW + kw;
    grad_W[w_idx] = sum;
}

__global__ void conv2d_backward_bias_kernel(
    const float* grad_out, float* grad_b,
    int N, int C_out, int H_out, int W_out
) {
    int co = blockIdx.x;
    if(co >= C_out) return;

    float sum = 0.0f;
    for(int n = 0; n < N; n++){
        for(int ho = 0; ho < H_out; ho++){
            for(int wo = 0; wo < W_out; wo++){
                int go_idx = n*C_out*H_out*W_out + co*H_out*W_out + ho*W_out + wo;
                sum += grad_out[go_idx];
            }
        }
    }
    grad_b[co] = sum;
}

void conv2d_layer_backward_cuda(Layer* layer, Tensor* grad_output) {
    if(!layer || !layer->input || !grad_output){
        fprintf(stderr,"ERROR: conv2d_backward called with NULL pointers!\n");
        exit(1);
    }

    Tensor* X = layer->input;
    Tensor* W = layer->weights[0];
    Tensor* b = (layer->n_weights > 1) ? layer->weights[1] : NULL;

    int N = X->shape[0];
    int C_in = X->shape[1];
    int H_in = X->shape[2];
    int W_in = X->shape[3];

    int C_out = W->shape[0];
    int kH = W->shape[2];
    int kW = W->shape[3];

    int H_out = grad_output->shape[2];
    int W_out = grad_output->shape[3];

    // allocate grad input
    if(!layer->grad_input) {
        layer->grad_input = tensor_create(X->ndim, X->shape, X->dtype, X->device);
        tensor_fill(layer->grad_input, 0.0f);
    }

    // allocate grad weights/bias
    if(!W->grad) W->grad = tensor_create(W->ndim, W->shape, W->dtype, W->device);
    tensor_fill(W->grad, 0.0f);

    if(b && !b->grad) b->grad = tensor_create(b->ndim, b->shape, b->dtype, b->device);
    if(b) tensor_fill(b->grad, 0.0f);

    // start kernels (naive blocks and threads mapping)
    dim3 threads(1);

    dim3 grid_input(H_in*W_in, C_in, N);
    conv2d_backward_input_kernel<<<grid_input, threads>>>(
        (float*)grad_output->data,
        (float*)W->data,
        (float*)layer->grad_input->data,
        N,C_in,H_in,W_in,
        C_out,kH,kW,
        H_out,W_out
    );

    dim3 grid_weight(kH*kW, C_in, C_out);
    conv2d_backward_weight_kernel<<<grid_weight, threads>>>(
        (float*)X->data,
        (float*)grad_output->data,
        (float*)W->grad->data,
        N,C_in,H_in,W_in,
        C_out,kH,kW,
        H_out,W_out
    );

    if(b){
        dim3 grid_bias(C_out);
        conv2d_backward_bias_kernel<<<grid_bias, threads>>>(
            (float*)grad_output->data,
            (float*)b->grad->data,
            N,C_out,H_out,W_out
        );
    }

    cudaDeviceSynchronize();
}


__global__ void maxpool2d_forward_kernel(
    const float* x, float* y,
    int N, int C, int H, int W,
    int kH, int kW, int stride,
    int H_out, int W_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N*C*H_out*W_out;
    if (idx >= total) return;

    int wo = idx % W_out;
    int ho = (idx / W_out) % H_out;
    int c  = (idx / (W_out*H_out)) % C;
    int n  = idx / (W_out*H_out*C);

    float max_val = -1e20f;
    int H_start = ho*stride;
    int W_start = wo*stride;

    for (int kh = 0; kh < kH; kh++) {
        for (int kw = 0; kw < kW; kw++) {
            int h_in = H_start + kh;
            int w_in = W_start + kw;
            int x_idx = n*C*H*W + c*H*W + h_in*W + w_in;
            if (x[x_idx] > max_val) max_val = x[x_idx];
        }
    }
    int y_idx = n*C*H_out*W_out + c*H_out*W_out + ho*W_out + wo;
    y[y_idx] = max_val;
}

void maxpool2d_layer_forward_cuda(Layer* layer, Tensor* input) {
    layer->input = input;

    int N = input->shape[0];
    int C = input->shape[1];
    int H = input->shape[2];
    int W = input->shape[3];
    int kH = 2;
    int kW = 2;
    int stride = 2;
    int H_out = H / kH;
    int W_out = W / kW;

    if (!layer->output) {
        int64_t out_shape[4] = {N, C, H_out, W_out};
        layer->output = tensor_create(4, out_shape, input->dtype, CUDA);
        tensor_fill(layer->output, 0.0f);
    }

    int total_threads = N*C*H_out*W_out;
    int block = 256;
    int grid = (total_threads + block - 1)/block;

    maxpool2d_forward_kernel<<<grid, block>>>(
        (float*)input->data, (float*)layer->output->data,
        N, C, H, W, kH, kW, stride, H_out, W_out
    );
    cudaDeviceSynchronize();
}


__global__ void maxpool2d_backward_kernel(
    const float* x, const float* go, float* gx,
    int N, int C, int H, int W,
    int kH, int kW, int stride,
    int H_out, int W_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N*C*H_out*W_out;
    if (idx >= total) return;

    int wo = idx % W_out;
    int ho = (idx / W_out) % H_out;
    int c  = (idx / (W_out*H_out)) % C;
    int n  = idx / (W_out*H_out*C);

    float max_val = -1e20f;
    int max_h = 0, max_w = 0;
    int H_start = ho*stride;
    int W_start = wo*stride;

    for (int kh = 0; kh < kH; kh++) {
        for (int kw = 0; kw < kW; kw++) {
            int h_in = H_start + kh;
            int w_in = W_start + kw;
            int x_idx = n*C*H*W + c*H*W + h_in*W + w_in;
            if (x[x_idx] > max_val) {
                max_val = x[x_idx];
                max_h = h_in;
                max_w = w_in;
            }
        }
    }

    int y_idx = n*C*H_out*W_out + c*H_out*W_out + ho*W_out + wo;
    int gx_idx = n*C*H*W + c*H*W + max_h*W + max_w;
    atomicAdd(&gx[gx_idx], go[y_idx]);  // multiple threads may write same element
}

void maxpool2d_layer_backward_cuda(Layer* layer, Tensor* grad_output) {
    if (!layer->grad_input) {
        layer->grad_input = tensor_create(4, layer->input->shape, layer->input->dtype, CUDA);
        tensor_fill(layer->grad_input, 0.0f);
    }

    int N = layer->input->shape[0];
    int C = layer->input->shape[1];
    int H = layer->input->shape[2];
    int W = layer->input->shape[3];
    int kH = 2;
    int kW = 2;
    int stride = 2;
    int H_out = grad_output->shape[2];
    int W_out = grad_output->shape[3];

    int total_threads = N*C*H_out*W_out;
    int block = 256;
    int grid = (total_threads + block - 1)/block;

    maxpool2d_backward_kernel<<<grid, block>>>(
        (float*)layer->input->data,
        (float*)grad_output->data,
        (float*)layer->grad_input->data,
        N, C, H, W, kH, kW, stride, H_out, W_out
    );
    cudaDeviceSynchronize();
}



#ifdef __cplusplus
}
#endif
