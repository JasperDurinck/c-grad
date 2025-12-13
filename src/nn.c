#include <stdio.h>
#include <stdlib.h>          
#include "../include/nn_layers.h"  
#include "../include/nn.h"
#include <stdbool.h>


Tensor* network_forward(Network* net, Tensor* x, bool verbose) {
    Tensor* current = x;


    for (int i = 0; i < net->n_layers; i++) {
        //tensor_print(current);       
        
        Layer* layer = net->layers[i];

        if (!layer) {
            fprintf(stderr, "ERROR: layer[%d] is NULL!\n", i);
            exit(1);
        }

        if (verbose) printf("\n--- Forward Layer %d ---\n", i);

        if (verbose && current) {
            printf("Input shape: (");
            for (int d = 0; d < current->ndim; d++) {
                printf("%ld%s", current->shape[d], (d==current->ndim-1 ? "" : ", "));
            }
            printf(")\n");
        }

        // If linear layer (has weights), print W and b
        if (verbose && layer->n_weights >= 1 && layer->weights) {
            Tensor* W = layer->weights[0];
            printf("W shape: (%ld, %ld)\n", W->shape[0], W->shape[1]);

            if (layer->n_weights == 2) {
                Tensor* b = layer->weights[1];
                printf("b shape: (%ld)\n", b->shape[0]);
            }
        }
        
        layer->forward(layer, current);

        Tensor* out = layer->output;
        if (verbose && out) {
            printf("Output shape: (");
            for (int d = 0; d < out->ndim; d++) {
                printf("%ld%s", out->shape[d], (d==out->ndim-1 ? "" : ", "));
            }
            printf(")\n");
        }

        current = out;
    }

    return current;
}

void network_backward(Network* net, Tensor* grad_output, bool verbose) {
    Tensor* current_grad = grad_output; // start from loss gradient w.r.t output

    if (verbose) printf("\n--- Starting backward pass ---\n");

    // backward pass
    for (int i = net->n_layers - 1; i >= 0; i--) {
        Layer* layer = net->layers[i];
        if (!layer) {
            fprintf(stderr, "ERROR: layer[%d] is NULL!\n", i);
            exit(1);
        }
        if (!layer->backward) {
            fprintf(stderr, "ERROR: layer[%d]->backward pointer is NULL!\n", i);
            exit(1);
        }

        // print input gradient shape
        if (verbose && current_grad) {
            printf("\nLayer %d backward:\n", i);
            printf("Input grad shape (grad_output): (");
            for (int d = 0; d < current_grad->ndim; d++) {
                printf("%ld%s", current_grad->shape[d], (d == current_grad->ndim - 1 ? "" : ", "));
            }
            printf(")\n");
        }

        // If linear layer, print weight shapes
        if (verbose && layer->n_weights >= 1 && layer->weights) {
            Tensor* W = layer->weights[0];
            printf("W shape: (%ld, %ld)\n", W->shape[0], W->shape[1]);

            if (layer->n_weights == 2) {
                Tensor* b = layer->weights[1];
                printf("b shape: (%ld)\n", b->shape[0]);
            }

            if (W->grad) {
                Tensor* Wgrad = (Tensor*)W->grad;  // cast to Tensor*
                printf("Previous W->grad shape: (%ld, %ld)\n", Wgrad->shape[0], Wgrad->shape[1]);
            }
        }

        //  backward
        layer->backward(layer, current_grad);

        if (verbose && layer->grad_input) {
            printf("Output grad shape (grad_input): (");
            for (int d = 0; d < layer->grad_input->ndim; d++) {
                printf("%ld%s", layer->grad_input->shape[d], (d == layer->grad_input->ndim - 1 ? "" : ", "));
            }
            printf(")\n");
        }

        // propagate to prior layer
        current_grad = layer->grad_input;
    }

    if (verbose) printf("\n--- Backward pass complete ---\n");
}

Network* create_mlp(int input_dim, int hidden_dim, int output_dim, int hidden_layers, Device dev)
{
    Network* net = malloc(sizeof(Network));
    net->n_layers = 2 * hidden_layers + 1;
    net->layers = malloc(sizeof(Layer*) * net->n_layers);

    int idx = 0;

    // Input ->  first hidden
    net->layers[idx++] = create_linear_layer(input_dim, hidden_dim, dev);
    net->layers[idx++] = create_relu_layer();

    // Hidden -> hidden layers
    for (int i = 1; i < hidden_layers; i++) {
        net->layers[idx++] = create_linear_layer(hidden_dim, hidden_dim, dev);
        net->layers[idx++] = create_relu_layer();
    }

    // Last layer ->output
    net->layers[idx++] = create_linear_layer(hidden_dim, output_dim, dev);

    return net;
}

Network* create_cnn(Device dev, const CNNConfig* cfg) {
    Network* net = malloc(sizeof(Network));
    net->n_layers = 7;
    net->layers = malloc(sizeof(Layer*) * net->n_layers);

    int idx = 0;

    // Track spatial size after each layer
    int H = cfg->input_height;
    int W = cfg->input_width;

    // ---------------- Conv1 ----------------
    net->layers[idx++] = create_conv2d_layer(
        cfg->input_channels,
        cfg->conv1_out_channels,
        cfg->conv1_kernel_h,
        cfg->conv1_kernel_w,
        dev
    );

    // Update output spatial size after Conv1
    H = H - cfg->conv1_kernel_h + 1;
    W = W - cfg->conv1_kernel_w + 1;

    // ---------------- ReLU ----------------
    net->layers[idx++] = create_relu_layer();

    // ---------------- MaxPool ----------------
    net->layers[idx++] = create_maxpool2d_layer();  // assume 2x2, stride 2

    // Update size after 2x2 pool
    H = H / 2;
    W = W / 2;

    // ---------------- Conv2 ----------------
    net->layers[idx++] = create_conv2d_layer(
        cfg->conv1_out_channels,
        cfg->conv2_out_channels,
        cfg->conv2_kernel_h,
        cfg->conv2_kernel_w,
        dev
    );

    // Update output size
    H = H - cfg->conv2_kernel_h + 1;
    W = W - cfg->conv2_kernel_w + 1;

    // ---------------- ReLU ----------------
    net->layers[idx++] = create_relu_layer();

    // ---------------- Flatten ----------------
    net->layers[idx++] = create_flatten_layer();

    // ---------------- Linear ----------------
    int linear_in_features = cfg->conv2_out_channels * H * W;
    net->layers[idx++] = create_linear_layer(
        linear_in_features,
        cfg->linear_out_features,
        dev
    );

    return net;
}