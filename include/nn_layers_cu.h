
#ifdef __cplusplus
extern "C" {
#endif

#ifndef NN_LAYERS_CU_H
#define NN_LAYERS_CU_H

#include "../include/tensor.h"
#include "../include/nn_layers.h"


void relu_layer_fn_cuda(Layer* layer, Tensor* input);
void relu_layer_fn_grad_cuda(Layer* layer, Tensor* grad_output);

// cnn

void conv2d_layer_forward_cuda(Layer* layer, Tensor* input);
void conv2d_layer_backward_cuda(Layer* layer, Tensor* grad_output);

void maxpool2d_layer_forward_cuda(Layer* layer, Tensor* input);
void maxpool2d_layer_backward_cuda(Layer* layer, Tensor* grad_output);

#endif

#ifdef __cplusplus
}
#endif