
#ifdef __cplusplus
extern "C" {
#endif

#ifndef NN_H
#define NN_H


#include "../include/nn_layers.h"
#include <stdbool.h>


typedef struct Network {
    Layer** layers;
    int n_layers;
} Network;

Tensor* network_forward(Network* net, Tensor* x, bool verbose);
void network_backward(Network* net, Tensor* grad_output, bool verbose);

Network* create_mlp(int input_dim, int hidden_dim, int output_dim, int hidden_layers, Device dev);

// cnn

typedef struct CNNConfig {
    int input_channels;
    int input_height;
    int input_width;

    int conv1_out_channels;
    int conv1_kernel_h;
    int conv1_kernel_w;

    int conv2_out_channels;
    int conv2_kernel_h;
    int conv2_kernel_w;

    int linear_out_features;
} CNNConfig;

Network* create_cnn(Device dev, const CNNConfig* cfg);

#endif

#ifdef __cplusplus
}
#endif