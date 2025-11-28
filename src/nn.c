#include <stdio.h>
#include <stdlib.h>          
#include "../include/nn_layers.h"  
#include "../include/nn.h"


Network* create_mlp(int input_dim, int hidden_dim, int output_dim, int hidden_layers, Device dev)
{
    Network* net = malloc(sizeof(Network));
    net->n_layers = 2 * hidden_layers + 1;
    net->layers = malloc(sizeof(Layer*) * net->n_layers);

    int idx = 0;

    // Input ->  first hidden
    net->layers[idx++] = create_linear_layer(input_dim, hidden_dim, dev);
    net->layers[idx++] = create_relu_layer();

    // Hidden → hidden layers
    for (int i = 1; i < hidden_layers; i++) {
        net->layers[idx++] = create_linear_layer(hidden_dim, hidden_dim, dev);
        net->layers[idx++] = create_relu_layer();
    }

    // Last layer ->output
    net->layers[idx++] = create_linear_layer(hidden_dim, output_dim, dev);

    return net;
}
