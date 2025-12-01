
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



#endif

#ifdef __cplusplus
}
#endif