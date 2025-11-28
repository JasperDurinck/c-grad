
#include "../include/tensor.h"
#include "../include/tensor_ops.h"
#include "../include/tensor_cu.h"
#include "../include/nn.h"
#include "../include/loss_fns.h"
#include "../include/optimizers.h"
#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>
#include <stdbool.h>

// CPU Demo
void demo_cpu() {
    printf("=== CPU Demo ===\n");

    int64_t shape[3] = {2, 3, 5};

    Tensor* A = tensor_create(3, shape, FLOAT32, CPU);
    Tensor* B = tensor_create(3, shape, FLOAT32, CPU);
    Tensor* C = tensor_create(3, shape, FLOAT32, CPU);

    tensor_fill(A, 3.0);
    tensor_fill(B, 6.0);

    printf("A (CPU):\n"); tensor_print(A);
    printf("B (CPU):\n"); tensor_print(B);

    tensor_add(A, B, C);
    printf("\nC = A + B (CPU):\n"); tensor_print(C);

    tensor_subtract(A, B, C);
    printf("\nC = A - B (CPU):\n"); tensor_print(C);

    tensor_mul(A, B, C);
    printf("\nC = A * B (CPU):\n"); 
    tensor_print(C);

    tensor_div(A, B, C);
    printf("\nC = A / B (CPU):\n"); 
    tensor_print(C);

    Tensor* At = tensor_transpose_out(A);
    tensor_transpose(A, At);
    printf("\nA transpose (CPU):\n");
    printf("\nA:\n"); tensor_print(A); 
    printf("\nAt:\n"); tensor_print(At); 
    
    Tensor* s = tensor_create_scalar(FLOAT32, CPU);
    tensor_sum(A, s);
    printf("\nSum of A (CPU): %f\n", *((float*)s->data));

    Tensor* m = tensor_create_scalar(FLOAT32, CPU);
    tensor_mean(A, m);
    printf("Mean of A (CPU): %f\n", *((float*)m->data));

    Tensor* mx = tensor_create_scalar(FLOAT32, CPU);
    tensor_max(A, mx);
    printf("Max of A (CPU): %f\n", *((float*)mx->data));

    Tensor* arg = tensor_create_scalar(INT64, CPU);
    tensor_argmax(A, arg);
    printf("Argmax of A (CPU): %" PRId64 "\n", *((int64_t*)arg->data));

    // Matmul
    int64_t shape_D[3] = {2, 5, 4};
    Tensor* D = tensor_create(3, shape_D, FLOAT32, CPU);
    tensor_fill(A, 1.0);
    tensor_fill(D, 2.0);

    Tensor* F = tensor_matmul_out(A, D);
    tensor_matmul(A, D, F);
    printf("\nF = A @ D (CPU):\n"); tensor_print(F);

    tensor_exp(A, B);
    printf("\ntensor_exp(A, B) (CPU):\n"); tensor_print(B);

    tensor_free(A);
    tensor_free(B);
    tensor_free(C);
    tensor_free(D);
    tensor_free(F);
}

void demo_add_bias_cpu() {
    printf("\n=== CPU tensor_add_bias Demo ===\n");

    // Create a 2D tensor: batch_size=3, out_features=4
    int64_t input_shape[2] = {3, 4};
    Tensor* input = tensor_create(2, input_shape, FLOAT32, CPU);
    tensor_fill(input, 1.0f);  // fill with 1s

    // Create a bias vector: out_features=4
    int64_t bias_shape[1] = {4};
    Tensor* bias = tensor_create(1, bias_shape, FLOAT32, CPU);
    tensor_fill(bias, 0.5f);  // fill with 0.5

    // Output tensor
    Tensor* out = tensor_create(2, input_shape, FLOAT32, CPU);

    printf("Input (CPU):\n"); tensor_print(input);
    printf("Bias (CPU):\n"); tensor_print(bias);

    // Add bias
    tensor_add_bias(input, bias, out);

    printf("\nOutput = input + bias (CPU):\n"); tensor_print(out);

    // Cleanup
    tensor_free(input);
    tensor_free(bias);
    tensor_free(out);
}

// GPU Demo (direct GPU ops)

void demo_gpu() {
    printf("\n=== GPU Demo ===\n");

    int64_t shape[3] = {2, 3, 5};

    // CPU tensors
    Tensor* A = tensor_create(3, shape, FLOAT32, CPU);
    Tensor* B = tensor_create(3, shape, FLOAT32, CPU);
    tensor_fill(A, 3.0);
    tensor_fill(B, 6.0);

    // Move to GPU
    Tensor* A_gpu = tensor_to_cuda(A);
    Tensor* B_gpu = tensor_to_cuda(B);
    Tensor* C_gpu = tensor_create(3, shape, FLOAT32, CUDA);

    printf("A (GPU):\n"); tensor_print(A);
    printf("B (GPU):\n"); tensor_print(B);

    tensor_add(A_gpu, B_gpu, C_gpu);
    Tensor* C = tensor_to_cpu(C_gpu);
    printf("\nC = A + B (GPU):\n"); tensor_print(C);

    tensor_subtract(A_gpu, B_gpu, C_gpu);
    C = tensor_to_cpu(C_gpu);
    printf("\nC = A - B (GPU):\n"); tensor_print(C);

    tensor_mul(A_gpu, B_gpu, C_gpu);
    C = tensor_to_cpu(C_gpu);
    printf("\nC = A * B (GPU):\n"); 
    tensor_print(C);

    tensor_div(A_gpu, B_gpu, C_gpu);
    C = tensor_to_cpu(C_gpu);
    printf("\nC = A / B (GPU):\n"); 
    tensor_print(C);

    Tensor* At_gpu = tensor_transpose_out(A_gpu);
    tensor_transpose(A_gpu, At_gpu);
    Tensor* At = tensor_to_cpu(At_gpu);
    printf("\nA transpose (GPU):\n");
    printf("\nA:\n"); tensor_print(A); 
    printf("\nAt:\n"); tensor_print(At); 


    tensor_exp(A_gpu, B_gpu);
    Tensor* B_exp = tensor_to_cpu(B_gpu);
    printf("\ntensor_exp(B_gpu, C_gpu) (CPU):\n"); tensor_print(B_exp);

    // ---------------------- GPU reductions ----------------------
    Tensor* s = tensor_create_scalar(FLOAT32, A_gpu->device);
    tensor_sum(A_gpu, s);
    Tensor* s_cpu = tensor_to_cpu(s);
    printf("\nSum of A (GPU): %f\n", *((float*)s_cpu->data));

    Tensor* m = tensor_create_scalar(FLOAT32, A_gpu->device);
    tensor_mean(A_gpu, m);
    Tensor* m_cpu = tensor_to_cpu(m);
    printf("Mean of A (GPU): %f\n", *((float*)m_cpu->data));

    Tensor* mx = tensor_create_scalar(FLOAT32, A_gpu->device);
    tensor_max(A_gpu, mx);
    Tensor* mx_cpu = tensor_to_cpu(mx);
    printf("Max of A (GPU): %f\n", *((float*)mx_cpu->data));

    Tensor* arg = tensor_create_scalar(INT64, A_gpu->device);
    tensor_argmax(A_gpu, arg);
    Tensor* arg_cpu = tensor_to_cpu(arg);
    printf("Argmax of A (GPU): %" PRId64 "\n", *((int64_t*)arg_cpu->data));
    
    // GPU matmul
    int64_t shape_D[3] = {2, 5, 4};
    Tensor* D_gpu = tensor_create(3, shape_D, FLOAT32, CUDA);
    tensor_fill(D_gpu, 2.0);

    Tensor* F_gpu = tensor_matmul_out(A_gpu, D_gpu);
    tensor_matmul(A_gpu, D_gpu, F_gpu);

    Tensor* F = tensor_to_cpu(F_gpu);
    printf("\nF = A @ D (GPU):\n"); tensor_print(F);

    // Free memory
    tensor_free(A);
    tensor_free(B);
    tensor_free(C);
    tensor_free(A_gpu);
    tensor_free(B_gpu);
    tensor_free(C_gpu);
    tensor_free(D_gpu);
    tensor_free(F_gpu);
    tensor_free(F);
    
}

void demo_add_bias_gpu() {
    printf("\n=== GPU tensor_add_bias Demo ===\n");

    // CPU input tensor
    int64_t input_shape[2] = {3, 4};  // batch_size=3, out_features=4
    Tensor* input_cpu = tensor_create(2, input_shape, FLOAT32, CPU);
    tensor_fill(input_cpu, 1.0f);

    // CPU bias vector
    int64_t bias_shape[1] = {4};
    Tensor* bias_cpu = tensor_create(1, bias_shape, FLOAT32, CPU);
    tensor_fill(bias_cpu, 0.5f);

    // Move tensors to GPU
    Tensor* input_gpu = tensor_to_cuda(input_cpu);
    Tensor* bias_gpu = tensor_to_cuda(bias_cpu);
    Tensor* out_gpu = tensor_create(2, input_shape, FLOAT32, CUDA);

    printf("Input (GPU, CPU view):\n"); tensor_print(input_cpu);
    printf("Bias (GPU, CPU view):\n"); tensor_print(bias_cpu);

    // Perform bias add on GPU
    tensor_add_bias(input_gpu, bias_gpu, out_gpu);

    // Move result back to CPU for printing
    Tensor* out_cpu = tensor_to_cpu(out_gpu);
    printf("\nOutput = input + bias (GPU):\n"); tensor_print(out_cpu);

    // Cleanup
    tensor_free(input_cpu);
    tensor_free(bias_cpu);
    tensor_free(input_gpu);
    tensor_free(bias_gpu);
    tensor_free(out_gpu);
    tensor_free(out_cpu);
}

// GPU to CPU transfer demo
void demo_transfers() {
    printf("\n=== GPU to CPU Transfer Demo ===\n");

    int64_t shape[3] = {2, 3, 5};

    Tensor* X = tensor_create(3, shape, FLOAT32, CPU);
    tensor_fill(X, 5.0);

    // Multiple moves between CPU and GPU
    Tensor* X_gpu = tensor_to_cuda(X);
    Tensor* X_cpu = tensor_to_cpu(X_gpu);
    Tensor* X_gpu2 = tensor_to_cuda(X_cpu);

    // Do computation on GPU
    Tensor* Y_gpu = tensor_create(3, shape, FLOAT32, CUDA);
    tensor_fill(Y_gpu, 2.0);
    Tensor* Z_gpu = tensor_create(3, shape, FLOAT32, CUDA);
    tensor_add(X_gpu2, Y_gpu, Z_gpu);

    Tensor* Z_cpu = tensor_to_cpu(Z_gpu);
    printf("Z = X + Y after multiple transfers:\n");
    tensor_print(Z_cpu);

    // Free all
    tensor_free(X);
    tensor_free(X_gpu);
    tensor_free(X_cpu);
    tensor_free(X_gpu2);
    tensor_free(Y_gpu);
    tensor_free(Z_gpu);
    tensor_free(Z_cpu);
}

Tensor* network_forward(Network* net, Tensor* x, bool verbose) {
    Tensor* current = x;

    for (int i = 0; i < net->n_layers; i++) {
        Layer* layer = net->layers[i];

        if (!layer) {
            fprintf(stderr, "ERROR: layer[%d] is NULL!\n", i);
            exit(1);
        }

        if (verbose) printf("\n--- Forward Layer %d ---\n", i);

        // Print input shape BEFORE calling forward
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

        // Run forward
        layer->forward(layer, current);

        // Print output shape AFTER forward
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

        // Print input gradient shape
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

        // Print resulting grad_input shape
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

void demo_mlp() {
    Network* mlp = create_mlp(128, 256, 10, 1, CPU);
    printf("\nMLP was made!\n");

    // Create example input: shape (3,128)
    int64_t shape[3] = {3, 128};
    Tensor* X = tensor_create(2, shape, FLOAT32, CPU);
    tensor_fill(X, 5.0);

    printf("Running forward pass...\n");
    network_forward(mlp, X, 1);


    int64_t out_shape[2] = {3, 10}; // final layer output shape
    Tensor* grad_output = tensor_create(2, out_shape, FLOAT32, CPU);
    tensor_fill(grad_output, 1.0f);  // pretend dL/dY = 1

    network_backward(mlp, grad_output, 1);

    printf("Forward output is stored in out->data\n");
}

void demo_mlp_and_train() {
    int input_dim = 128;
    int hidden_dim = 256;
    int output_dim = 10;
    int hidden_layers = 1;
    Device dev = CPU;

    Network* net = create_mlp(input_dim, hidden_dim, output_dim, hidden_layers, dev);
    printf("\nMLP was made!\n");

    // create example input and target
    int64_t X_shape[2] = {3, input_dim};
    Tensor* X = tensor_create(2, X_shape, FLOAT32, dev);
    tensor_fill(X, 5.0f);

    int64_t y_shape[2] = {3, output_dim};
    Tensor* y_true = tensor_create(2, y_shape, FLOAT32, dev);
    tensor_fill(y_true, 0.5f);

    // Create a tensor to store gradient of loss w.r.t output
    Tensor* grad_output = tensor_create(2, y_shape, FLOAT32, dev);

    // Create optimizer (SGD example)
    Optimizer sgd_opt = {
        .layers = net->layers,
        .n_layers = net->n_layers,
        .update_fn = sgd_update_optimizer,
        .state = NULL
    };

    int epochs = 100;
    float lr = 0.01f;

    for (int epoch = 0; epoch < epochs; epoch++) {
        printf("\n=== Epoch %d ===\n", epoch);

        Tensor* y_pred = network_forward(net, X, 0);

        float loss = mse_loss(y_pred, y_true, grad_output);
        printf("Loss = %f\n", loss);

        network_backward(net, grad_output, 0);
        optimizer_step(&sgd_opt, lr);
    }

    printf("\nTraining loop complete!\n");
    printf("Forward output of last pass is stored in y_pred->data \n");
}

int main() {
    demo_cpu();
    demo_gpu();
    demo_transfers();
    demo_add_bias_gpu();
    demo_add_bias_cpu();
    demo_mlp();
    demo_mlp_and_train();

    return 0;
}
