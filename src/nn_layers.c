#include "../include/nn_layers.h"
#include "../include/nn_layers_cu.h"
#include "../include/tensor.h"
#include "../include/tensor_cu.h"
#include "../include/tensor_ops.h"
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>

// ---------------------- Linear layer CPU functions ----------------------

void linear_layer_forward(Layer* layer, Tensor* input) {
    layer->input = input;

    Tensor* W = layer->weights[0];
    Tensor* b = layer->weights[1];

    // Allocate output tensor if not allocated
    if (layer->output == NULL) {
        int64_t out_shape[2] = { input->shape[0], W->shape[1] };
        layer->output = tensor_create(2, out_shape, input->dtype, input->device);
    }

    // out = input @ W
    tensor_matmul(input, W, layer->output);

    // out += b
    tensor_add_bias(layer->output, b, layer->output);
}

void linear_layer_backward(Layer* layer, Tensor* grad_output) {
    if (!layer || !layer->input || !grad_output) {
        fprintf(stderr, "ERROR: linear_backward called with NULL pointers!\n");
        exit(1);
    }

    Tensor* X = layer->input;
    Tensor* W = layer->weights[0];
    Tensor* b = layer->weights[1];

    // Allocate grad_input if not yet allocated
    if (!layer->grad_input) {
        layer->grad_input = tensor_create(X->ndim, X->shape, X->dtype, X->device);
    }

    // allocate grad for weights if not allocated 
    if (!W->grad) W->grad = tensor_create(W->ndim, W->shape, W->dtype, W->device);
    if (!b->grad) b->grad = tensor_create(b->ndim, b->shape, b->dtype, b->device);

    Tensor* W_T = tensor_create(2, (int64_t[]){W->shape[1], W->shape[0]}, W->dtype, W->device);
    tensor_transpose(W, W_T);
    tensor_matmul(grad_output, W_T, layer->grad_input);
    tensor_free(W_T);

    // W->grad = X^T @ grad_output
    Tensor* X_T = tensor_create(2, (int64_t[]){X->shape[1], X->shape[0]}, X->dtype, W->device);
    tensor_transpose(X, X_T);
    tensor_matmul(X_T, grad_output, W->grad);
    tensor_free(X_T);

    // b->grad = sum over batch dimensions
    tensor_sum_axis(grad_output, 0, b->grad);

}

Layer* create_linear_layer(int in_features, int out_features, Device dev) {
    Layer* layer = (Layer*)malloc(sizeof(Layer));
    layer->input  = NULL;
    layer->output = NULL;
    layer->grad_input = NULL;

    layer->n_weights = 2;
    layer->weights   = (Tensor**)malloc(2 * sizeof(Tensor*));

    // Weight matrix W: [in_features, out_features]
    int64_t w_shape[2] = { in_features, out_features };
    layer->weights[0] = tensor_create(2, w_shape, FLOAT32, dev);
    tensor_fill_random(layer->weights[0], -0.01, 0.01f);

    // Bias b: [out_features]
    int64_t b_shape[1] = { out_features };
    layer->weights[1] = tensor_create(1, b_shape, FLOAT32, dev);
    tensor_fill(layer->weights[1], 0.0f);

    layer->forward  = linear_layer_forward;   
    layer->backward = linear_layer_backward;  
    return layer;
}

void relu_layer_fn_cpu(Layer* layer, Tensor* input) {
    layer->input = input;

    if (!layer->output) {
        layer->output = tensor_create(input->ndim, input->shape, input->dtype, input->device);
    }

    int64_t n = tensor_numel(input);
    float* x_data = (float*)input->data;
    float* y_data = (float*)layer->output->data;

    for (int64_t i = 0; i < n; i++) {
        y_data[i] = fmaxf(0.0f, x_data[i]);
    }
}

void relu_layer_fn_grad_cpu(Layer* layer, Tensor* grad_output) {
    if (!layer->input) {
        fprintf(stderr, "ReLU backward called without forward input!\n");
        exit(1);
    }

    Tensor* X = layer->input;

    if (!layer->grad_input) {
        layer->grad_input = tensor_create(X->ndim, X->shape, X->dtype, X->device);
    }

    int64_t n = tensor_numel(X);
    float* go = (float*)grad_output->data;
    float* x  = (float*)X->data;
    float* gi = (float*)layer->grad_input->data;

    for (int64_t i = 0; i < n; i++) {
        gi[i] = (x[i] > 0.0f) ? go[i] : 0.0f;
    }
}

// Forward/backward wrappers
void relu_forward(Layer* layer, Tensor* input) {
    switch (input->device) {
        case CPU:
            relu_layer_fn_cpu(layer, input);
            break;
        case CUDA:
            relu_layer_fn_cuda(layer, input);
            break;
        default:
            fprintf(stderr, "relu_forward: unknown device!\n");
            exit(1);
    }
}

void relu_backward(Layer* layer, Tensor* grad_output) {
    switch (layer->input->device) {
        case CPU:
            relu_layer_fn_grad_cpu(layer, grad_output);
            break;
        case CUDA:
            relu_layer_fn_grad_cuda(layer, grad_output);
            break;
        default:
            fprintf(stderr, "relu_backward: unknown device!\n");
            exit(1);
    }
}

// Layer constructor
Layer* create_relu_layer() {
    Layer* layer = (Layer*)malloc(sizeof(Layer));
    layer->input = NULL;
    layer->output = NULL;
    layer->grad_input = NULL;

    layer->weights = NULL;
    layer->n_weights = 0;

    layer->forward  = relu_forward;
    layer->backward = relu_backward;
    return layer;
}



// cnn


void conv2d_layer_forward_cpu(Layer* layer, Tensor* input) {
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
    if (!layer->output) {
        int64_t out_shape[4] = { N, C_out, H_out, W_out };
        layer->output = tensor_create(4, out_shape, input->dtype, input->device);
        tensor_fill(layer->output, 0.0f); // initialize out
    }

    float* x = (float*)input->data;
    float* w = (float*)W->data;
    float* y = (float*)layer->output->data;
    float* bias = b ? (float*)b->data : NULL;

    // Naive nested loops
    for (int n = 0; n < N; n++) {
        for (int co = 0; co < C_out; co++) {
            for (int ho = 0; ho < H_out; ho++) {
                for (int wo = 0; wo < W_out; wo++) {
                    float sum = 0.0f;
                    for (int ci = 0; ci < C_in; ci++) {
                        for (int kh = 0; kh < kH; kh++) {
                            for (int kw = 0; kw < kW; kw++) {
                                int h_in = ho*stride + kh - pad;
                                int w_in = wo*stride + kw - pad;
                                if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                                    int x_idx = n*C_in*H_in*W_in + ci*H_in*W_in + h_in*W_in + w_in;
                                    int w_idx = co*C_in*kH*kW + ci*kH*kW + kh*kW + kw;
                                    sum += x[x_idx] * w[w_idx];
                                }
                            }
                        }
                    }
                    if (bias) sum += bias[co];
                    int y_idx = n*C_out*H_out*W_out + co*H_out*W_out + ho*W_out + wo;
                    y[y_idx] = sum;
                }
            }
        }
    }
}

void conv2d_layer_forward(Layer* layer, Tensor* input) {
    switch (input->device) {
        case CPU:
            conv2d_layer_forward_cpu(layer, input);
            break;
        case CUDA:
            conv2d_layer_forward_cuda(layer, input);
            break;
        default:
            fprintf(stderr, "conv2d_layer_forward: unknown device!\n");
            exit(1);
    }
}

void conv2d_layer_backward_cpu(Layer* layer, Tensor* grad_output) {
    if (!layer || !layer->input || !grad_output) {
        fprintf(stderr, "ERROR: conv2d_backward called with NULL pointers!\n");
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

    int stride = 1;
    int pad = 0;

    int H_out = grad_output->shape[2];
    int W_out = grad_output->shape[3];

    // allocate grad_input
    if (!layer->grad_input) {
        layer->grad_input = tensor_create(X->ndim, X->shape, X->dtype, X->device);
        tensor_fill(layer->grad_input, 0.0f);
    }

    // allocate grad for weights/bias 
    if (!W->grad) W->grad = tensor_create(W->ndim, W->shape, W->dtype, W->device);
    tensor_fill(W->grad, 0.0f);

    if (b && !b->grad) b->grad = tensor_create(b->ndim, b->shape, b->dtype, b->device);
    if (b) tensor_fill(b->grad, 0.0f);

    float* x = (float*)X->data;
    float* w = (float*)W->data;
    float* go = (float*)grad_output->data;
    float* gx = (float*)layer->grad_input->data;
    float* gw = (float*)W->grad->data;
    float* gb = b ? (float*)b->grad->data : NULL;

    // compute grad_input
    for (int n = 0; n < N; n++) {
        for (int ci = 0; ci < C_in; ci++) {
            for (int hi = 0; hi < H_in; hi++) {
                for (int wi = 0; wi < W_in; wi++) {
                    float sum = 0.0f;
                    for (int co = 0; co < C_out; co++) {
                        for (int kh = 0; kh < kH; kh++) {
                            for (int kw = 0; kw < kW; kw++) {
                                int ho = hi - kh + pad;
                                int wo = wi - kw + pad;
                                if (ho % stride != 0 || wo % stride != 0) continue;
                                ho /= stride;
                                wo /= stride;
                                if (ho >= 0 && ho < H_out && wo >= 0 && wo < W_out) {
                                    int go_idx = n*C_out*H_out*W_out + co*H_out*W_out + ho*W_out + wo;
                                    int w_idx = co*C_in*kH*kW + ci*kH*kW + kh*kW + kw;
                                    sum += go[go_idx] * w[w_idx];
                                }
                            }
                        }
                    }
                    int gx_idx = n*C_in*H_in*W_in + ci*H_in*W_in + hi*W_in + wi;
                    gx[gx_idx] = sum;
                }
            }
        }
    }

    // compute grad_weight
    for (int co = 0; co < C_out; co++) {
        for (int ci = 0; ci < C_in; ci++) {
            for (int kh = 0; kh < kH; kh++) {
                for (int kw = 0; kw < kW; kw++) {
                    float sum = 0.0f;
                    for (int n = 0; n < N; n++) {
                        for (int ho = 0; ho < H_out; ho++) {
                            for (int wo = 0; wo < W_out; wo++) {
                                int hi = ho*stride - pad + kh;
                                int wi = wo*stride - pad + kw;
                                if (hi >= 0 && hi < H_in && wi >= 0 && wi < W_in) {
                                    int x_idx = n*C_in*H_in*W_in + ci*H_in*W_in + hi*W_in + wi;
                                    int go_idx = n*C_out*H_out*W_out + co*H_out*W_out + ho*W_out + wo;
                                    sum += x[x_idx] * go[go_idx];
                                }
                            }
                        }
                    }
                    int w_idx = co*C_in*kH*kW + ci*kH*kW + kh*kW + kw;
                    gw[w_idx] = sum;
                }
            }
        }
    }

    // compute grad bias 
    if (b) {
        for (int co = 0; co < C_out; co++) {
            float sum = 0.0f;
            for (int n = 0; n < N; n++) {
                for (int ho = 0; ho < H_out; ho++) {
                    for (int wo = 0; wo < W_out; wo++) {
                        int go_idx = n*C_out*H_out*W_out + co*H_out*W_out + ho*W_out + wo;
                        sum += go[go_idx];
                    }
                }
            }
            gb[co] = sum;
        }
    }
}

void conv2d_layer_backward(Layer* layer, Tensor* grad_output) {
    switch (grad_output->device) {
        case CPU:
            conv2d_layer_backward_cpu(layer, grad_output);
            break;
        case CUDA:
            conv2d_layer_backward_cuda(layer, grad_output);
            break;
        default:
            fprintf(stderr, "conv2d_layer_backward: unknown device!\n");
            exit(1);
    }
}

void maxpool2d_layer_forward_cpu(Layer* layer, Tensor* input) {
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

    // allocate output
    if (!layer->output) {
        int64_t out_shape[4] = {N, C, H_out, W_out};
        layer->output = tensor_create(4, out_shape, input->dtype, input->device);
        tensor_fill(layer->output, 0.0f);
    }

    float* x = (float*)input->data;
    float* y = (float*)layer->output->data;

    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int ho = 0; ho < H_out; ho++) {
                for (int wo = 0; wo < W_out; wo++) {
                    float max_val = -1e20f;
                    for (int kh = 0; kh < kH; kh++) {
                        for (int kw = 0; kw < kW; kw++) {
                            int h_in = ho*stride + kh;
                            int w_in = wo*stride + kw;
                            int idx = n*C*H*W + c*H*W + h_in*W + w_in;
                            if (x[idx] > max_val) max_val = x[idx];
                        }
                    }
                    int out_idx = n*C*H_out*W_out + c*H_out*W_out + ho*W_out + wo;
                    y[out_idx] = max_val;
                }
            }
        }
    }
}

void maxpool2d_layer_forward(Layer* layer, Tensor* input) {
    switch (input->device) {
        case CPU:
            maxpool2d_layer_forward_cpu(layer, input);
            break;
        case CUDA:
            maxpool2d_layer_forward_cuda(layer, input);
            break;
        default:
            fprintf(stderr, "maxpool2d_layer_forward: unknown device!\n");
            exit(1);
    }
}

void maxpool2d_layer_backward_cpu(Layer* layer, Tensor* grad_output) {
    if (!layer->input || !grad_output) return;

    int N = layer->input->shape[0];
    int C = layer->input->shape[1];
    int H = layer->input->shape[2];
    int W = layer->input->shape[3];

    int kH = 2;
    int kW = 2;
    int stride = 2;

    int H_out = grad_output->shape[2];
    int W_out = grad_output->shape[3];

    // allocate grad_input
    if (!layer->grad_input) {
        layer->grad_input = tensor_create(4, layer->input->shape, layer->input->dtype, layer->input->device);
        tensor_fill(layer->grad_input, 0.0f);
    }

    float* x = (float*)layer->input->data;
    float* gx = (float*)layer->grad_input->data;
    float* go = (float*)grad_output->data;

    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int ho = 0; ho < H_out; ho++) {
                for (int wo = 0; wo < W_out; wo++) {
                    float max_val = -1e20f;
                    int max_h=0, max_w=0;
                    // find max element
                    for (int kh = 0; kh < kH; kh++) {
                        for (int kw = 0; kw < kW; kw++) {
                            int h_in = ho*stride + kh;
                            int w_in = wo*stride + kw;
                            int idx = n*C*H*W + c*H*W + h_in*W + w_in;
                            if (x[idx] > max_val) {
                                max_val = x[idx];
                                max_h = h_in;
                                max_w = w_in;
                            }
                        }
                    }
                    int out_idx = n*C*H_out*W_out + c*H_out*W_out + ho*W_out + wo;
                    int gx_idx = n*C*H*W + c*H*W + max_h*W + max_w;
                    gx[gx_idx] += go[out_idx];  // propagate grad
                }
            }
        }
    }
}

void maxpool2d_layer_backward(Layer* layer, Tensor* grad_output) {
    switch (grad_output->device) {
        case CPU:
            maxpool2d_layer_backward_cpu(layer, grad_output);
            break;
        case CUDA:
            maxpool2d_layer_backward_cuda(layer, grad_output);
            break;
        default:
            fprintf(stderr, "maxpool2d_layer_backward: unknown device!\n");
            exit(1);
    }
}

Layer* create_conv2d_layer(int C_in, int C_out, int kH, int kW, Device dev) {
    Layer* layer = (Layer*)malloc(sizeof(Layer));
    layer->input = NULL;
    layer->output = NULL;
    layer->grad_input = NULL;

    layer->n_weights = 2; // weights + bias
    layer->weights = (Tensor**)malloc(sizeof(Tensor*) * 2);

    // Weight tensor: [C_out, C_in, kH, kW]
    int64_t w_shape[4] = {C_out, C_in, kH, kW};
    layer->weights[0] = tensor_create(4, w_shape, FLOAT32, dev);
    tensor_fill_random(layer->weights[0], -0.01, 0.01f);

    // Bias: [C_out]
    int64_t b_shape[1] = {C_out};
    layer->weights[1] = tensor_create(1, b_shape, FLOAT32, dev);
    tensor_fill(layer->weights[1], 0.0f);

    layer->forward  = conv2d_layer_forward;
    layer->backward = conv2d_layer_backward;

    return layer;
}

Layer* create_maxpool2d_layer() {
    Layer* layer = (Layer*)malloc(sizeof(Layer));
    layer->input = NULL;
    layer->output = NULL;
    layer->grad_input = NULL;

    layer->weights = NULL;
    layer->n_weights = 0;

    layer->forward  = maxpool2d_layer_forward;
    layer->backward = maxpool2d_layer_backward;

    return layer;
}

void flatten_layer_forward(Layer* layer, Tensor* input) {
    layer->input = input;

    int64_t new_shape[2] = {
        input->shape[0],  // N
        -1                // infer C*H*W
    };

    layer->output = tensor_reshape(input, 2, new_shape);
}

void flatten_layer_backward(Layer* layer, Tensor* grad_output) {
    Tensor* input = layer->input;

    layer->grad_input = tensor_reshape(
        grad_output,
        input->ndim,
        input->shape
    );
}

Layer* create_flatten_layer() {
    Layer* layer = (Layer*)malloc(sizeof(Layer));

    layer->input = NULL;
    layer->output = NULL;
    layer->grad_input = NULL;

    layer->forward = flatten_layer_forward;
    layer->backward = flatten_layer_backward;

    return layer;
}