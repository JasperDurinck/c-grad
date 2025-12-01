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