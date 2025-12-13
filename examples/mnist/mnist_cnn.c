#include "../include/tensor.h"
#include "../include/tensor_ops.h"
#include "../include/tensor_cu.h"
#include "../include/nn.h"
#include "../include/loss_fns.h"
#include "../include/loss_fns_cu.h"
#include "../include/optimizers.h"
#include "../include/dataset.h"
#include "../include/metrics.h"
#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>
#include <stdbool.h>
#include <string.h>

void main() {

    //TODO improve speed, copying to device is slow, simple example for now

    Dataset* mnist = create_mnist_dataset(
        "/home/pc/code/c-grad/examples/mnist/data/MNIST/raw/train-images-idx3-ubyte", 
        "/home/pc/code/c-grad/examples/mnist/data/MNIST/raw/train-labels-idx1-ubyte"
    );

    DataLoader* loader = dataloader_create(mnist, 64, 1);

    CNNConfig cfg = {
        .input_channels = 1,
        .input_height = 28,
        .input_width = 28,

        .conv1_out_channels = 8,
        .conv1_kernel_h = 3,
        .conv1_kernel_w = 3,

        .conv2_out_channels = 16,
        .conv2_kernel_h = 3,
        .conv2_kernel_w = 3,

        .linear_out_features = 10
    };

    Network* net = create_cnn(CUDA, &cfg);


    Optimizer opt = {
        .layers = net->layers,
        .n_layers = net->n_layers,
        .update_fn = sgd_update_optimizer,
        .state = NULL
    };

    int epochs = 10;
    float lr = 0.01f;

    int64_t n_samples = 60000;

    for (int epoch = 1; epoch <= epochs; epoch++) {
        dataloader_reset(loader);

        Tensor *X, *Y;
        float total_loss = 0;
        int batches = 0;

        Tensor* all_preds = tensor_create(2, (int64_t[]){n_samples, 10}, FLOAT32, CPU);
        Tensor* all_labels = tensor_create(1, (int64_t[]){n_samples}, INT64, CPU);

        int offset = 0; // position in the pre allocated tensors

        while (dataloader_next(loader, &X, &Y)) {

            int batch_size = X->shape[0];

            Tensor* X_dev = tensor_to_cuda(X);

            Tensor* logits = network_forward(net, X_dev, 0);

            Tensor* grad_out = tensor_create_as(logits);
            Tensor* Y_dev = tensor_to_cuda(Y);

            float loss = cross_entropy_loss(logits, Y_dev, grad_out);

            network_backward(net, grad_out, 0);
            optimizer_step(&opt, lr);

            total_loss += loss;
            batches++;

            Tensor* logits_cpu = tensor_to_cpu(logits);
            Tensor* Y_cpu = tensor_to_cpu(Y);

            for (int i = 0; i < batch_size; i++) {
                // copy logits as floats
                memcpy((float*)all_preds->data + (offset + i) * 10,
                    (float*)logits_cpu->data + i * 10,
                    10 * sizeof(float));

                // copy labels safely depending on type
                if (Y_cpu->dtype == FLOAT32 && all_labels->dtype == FLOAT32) {
                    ((float*)all_labels->data)[offset + i] = ((float*)Y_cpu->data)[i];
                } else if (Y_cpu->dtype == INT64 && all_labels->dtype == INT64) {
                    ((int64_t*)all_labels->data)[offset + i] = ((int64_t*)Y_cpu->data)[i];
                } else if (Y_cpu->dtype == INT32 && all_labels->dtype == INT32) {
                    ((int32_t*)all_labels->data)[offset + i] = ((int32_t*)Y_cpu->data)[i];
                } else {
                    fprintf(stderr, "Unsupported label dtype combination: Y_cpu=%d, all_labels=%d\n",
                            Y_cpu->dtype, all_labels->dtype);
                    exit(1);
                }
            }

            offset += batch_size;

            // printf("iteration\n");
            // tensor_print(tensor_argmax_dim1(logits_cpu));
            // tensor_print(Y_cpu);



            // free temporary tensors
            tensor_free(X);
            tensor_free(Y);
            tensor_free(grad_out);
            tensor_free(Y_dev);
            tensor_free(Y_cpu);
            tensor_free(logits_cpu);
        }

        all_preds = tensor_argmax_dim1(all_preds);

        float acc = accuracy(all_preds, all_labels);
        float mcc = mcc_mc_score(all_preds, all_labels, 10);

        printf("Epoch %d | Loss = %.4f | Accuracy = %.4f | MCC = %.4f\n",
               epoch, total_loss / batches, acc, mcc);

        tensor_free(all_preds);
        tensor_free(all_labels);
    }
}