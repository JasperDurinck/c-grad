

#include "../include/tensor.h"
#include "../include/tensor_ops.h"
#include "../include/tensor_cu.h"
#include "../include/nn.h"
#include "../include/loss_fns.h"
#include "../include/loss_fns_cu.h"
#include "../include/optimizers.h"
#include "../include/dataset.h"
#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>
#include <stdbool.h>

void main(){

    Dataset* mnist = create_mnist_dataset(
        "/home/pc/code/c-grad/examples/mnist/data/MNIST/raw/train-images-idx3-ubyte", 
        "/home/pc/code/c-grad/examples/mnist/data/MNIST/raw/train-labels-idx1-ubyte"
    );

    DataLoader* loader = dataloader_create(mnist, 64, 1);

    Network* net = create_mlp(28*28, 256, 10, 1, CUDA);

    Optimizer opt = {
        .layers = net->layers,
        .n_layers = net->n_layers,
        .update_fn = sgd_update_optimizer,
        .state = NULL
    };

    int epochs = 10;
    float lr = 0.01f;

    for (int epoch = 1; epoch <= epochs; epoch++) {
        dataloader_reset(loader);

        Tensor *X, *Y;
        float total_loss = 0;
        int batches = 0;

        while (dataloader_next(loader, &X, &Y)) {

            // reshape X to (batch, 784)
            Tensor* X_flat = tensor_reshape(X, 2, (int64_t[]){X->shape[0], 784});
            Tensor* X_flat_dev = tensor_to_cuda(X_flat);

            Tensor* logits = network_forward(net, X_flat_dev, 0);

            Tensor* grad_out = tensor_create_as(logits);
            Tensor* Y_dev = tensor_to_cuda(Y);

            float loss = cross_entropy_loss(logits, Y_dev, grad_out);

            network_backward(net, grad_out, 0);
            optimizer_step(&opt, lr);

            total_loss += loss;
            batches++;

            tensor_free(X);
            tensor_free(Y);
            tensor_free(X_flat);
            tensor_free(grad_out);
            
        }

        printf("Epoch %d | Loss = %.4f\n", epoch, total_loss / batches);
    }

}
