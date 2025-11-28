
#ifdef __cplusplus
extern "C" {
#endif

#ifndef NN_H
#define NN_H


#include "../include/nn_layers.h"

typedef struct Network {
    Layer** layers;
    int n_layers;
} Network;

Network* create_mlp(int input_dim, int hidden_dim, int output_dim, int hidden_layers, Device dev);


#endif

#ifdef __cplusplus
}
#endif