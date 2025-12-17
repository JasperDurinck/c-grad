
#ifdef __cplusplus
extern "C" {
#endif

#ifndef NN_LAYERS_H
#define NN_LAYERS_H


#include "../include/tensor.h"

typedef struct Layer {
    // Forward/backward storage
    Tensor* input;       // input from forward pass
    Tensor* output;      // output of forward pass
    Tensor* grad_input;  // gradient w.r.t input for backward pass

    // Parameters (weights)
    Tensor** weights;    // array of weight tensors (NULL for weightless layers)
    int       n_weights; // number of weight tensors (0 for ReLU, etc.)

    // Function pointers
    void (*forward)(struct Layer* self, Tensor* input);
    void (*backward)(struct Layer* self, Tensor* grad_output);
} Layer;

// Linear FF
void linear_layer_forward(Layer* layer, Tensor* input);
void linear_layer_backward(Layer* layer, Tensor* grad_output);
Layer* create_linear_layer(int in_features, int out_features, Device dev);

// ReLU

void relu_layer_fn_cpu(Layer* layer, Tensor* input);
void relu_layer_fn_grad_cpu(Layer* layer, Tensor* grad_output);
void relu_forward(Layer* layer, Tensor* input);
void relu_backward(Layer* layer, Tensor* grad_output);
Layer* create_relu_layer();

// CNN

void conv2d_layer_forward_cpu(Layer* layer, Tensor* input);
void conv2d_layer_backward_cpu(Layer* layer, Tensor* grad_output);

void maxpool2d_layer_forward_cpu(Layer* layer, Tensor* input);
void maxpool2d_layer_backward_cpu(Layer* layer, Tensor* grad_output);

void conv2d_layer_forward(Layer* layer, Tensor* input);
void conv2d_layer_backward(Layer* layer, Tensor* grad_output);

Layer* create_maxpool2d_layer();
Layer* create_conv2d_layer(int C_in, int C_out, int kH, int kW, Device dev);

void flatten_layer_forward(Layer* layer, Tensor* input);
void flatten_layer_backward(Layer* layer, Tensor* grad_output);

Layer* create_flatten_layer();


void maxpool2d_layer_forward(Layer* layer, Tensor* input);
void maxpool2d_layer_backward(Layer* layer, Tensor* grad_output);

#endif

#ifdef __cplusplus
}
#endif