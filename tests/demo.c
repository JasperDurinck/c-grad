
#include "../include/tensor.h"
#include "../include/tensor_ops.h"
#include "../include/tensor_cu.h"
#include "../include/nn.h"
#include "../include/loss_fns.h"
#include "../include/loss_fns_cu.h"
#include "../include/optimizers.h"
#include "../include/metrics.h"
#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>
#include <stdbool.h>
#include <assert.h>
#include <string.h>

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
    
    tensor_exp(A, B);
    printf("\ntensor_exp(A, B) (CPU):\n"); tensor_print(B);

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

int demo_relu_forward_cpu() {
    int64_t shape[2] = {2, 5};
    Tensor* t = tensor_create(2, shape, FLOAT32, CPU);
    tensor_fill(t, 5.0f);

    printf("Input tensor:\n");
    tensor_print(t);

    Layer* relu = create_relu_layer();

    relu->forward(relu, t);
    printf("\nReLU output:\n");
    tensor_print(relu->output);

    Tensor* grad_out = tensor_create(2, shape, FLOAT32, CPU);
    tensor_fill(grad_out, 1.0f);


    relu->backward(relu, grad_out);
    printf("\nReLU grad_input:\n");
    tensor_print(relu->grad_input);

    tensor_free(t);
    tensor_free(grad_out);
    tensor_free(relu->output);
    tensor_free(relu->grad_input);
    free(relu);

    return 0;
}

int demo_relu_forward_cuda() {
    // 1. Define larger tensor shape
    int64_t shape[2] = {64, 5024};  // batch=64, features=1024
    Tensor* t = tensor_create(2, shape, FLOAT32, CUDA);
    tensor_fill(t, 5.0f);  // fill with positive values

    printf("Input tensor (GPU):\n");
    tensor_print(t);  // prints truncated output if your tensor_print is updated

    // 2. Create ReLU layer
    Layer* relu = create_relu_layer();

    // 3. Forward pass on GPU
    relu->forward(relu, t);
    printf("\nReLU output (GPU):\n");
    tensor_print(relu->output);

    // 4. Backward pass with gradient tensor of ones
    Tensor* grad_out = tensor_create(2, shape, FLOAT32, CUDA);
    tensor_fill(grad_out, 1.0f);

    relu->backward(relu, grad_out);
    printf("\nReLU grad_input (GPU):\n");
    tensor_print(relu->grad_input);

    // 5. Free memory
    tensor_free(t);
    tensor_free(grad_out);
    tensor_free(relu->output);
    tensor_free(relu->grad_input);
    free(relu);

    return 0;
}

int demo_relu_forward_cuda_loop_simple(int num_iterations) {
    for (int i = 0; i < num_iterations; i++) {
        printf("\n=== Iteration %d ===\n", i + 1);
        demo_relu_forward_cuda();  // call your existing CUDA demo
    }
    return 0;
}

int demo_relu_forward_backward_loop(int num_iterations) {
    int64_t shape[2] = {64, 1024};
    Tensor* t = tensor_create(2, shape, FLOAT32, CUDA);
    tensor_fill(t, 5.0f);

    Tensor* grad_out = tensor_create(2, shape, FLOAT32, CUDA);
    tensor_fill(grad_out, 1.0f);

    Layer* relu = create_relu_layer();

    for (int i = 0; i < num_iterations; i++) {
        printf("\n=== Iteration %d ===\n", i + 1);

        // Forward pass
        relu->forward(relu, t);
        printf("ReLU output (truncated):\n");
        tensor_print(relu->output);

        // Backward pass
        relu->backward(relu, grad_out);
        printf("ReLU grad_input (truncated):\n");
        tensor_print(relu->grad_input);
    }

    // Free memory
    tensor_free(t);
    tensor_free(grad_out);
    tensor_free(relu->output);
    tensor_free(relu->grad_input);
    free(relu);

    return 0;
}

int demo_linear_forward_backward_loop(int num_iterations) {
    // Layer dimensions
    int64_t batch = 64;
    int64_t in_features = 1024;
    int64_t out_features = 512;

    // 1. Create input tensor on CUDA
    int64_t input_shape[2] = {batch, in_features};
    Tensor* t = tensor_create(2, input_shape, FLOAT32, CUDA);
    tensor_fill(t, 1.0f);  // or random values

    // 2. Create linear layer on CUDA
    Layer* linear = create_linear_layer(in_features, out_features, CUDA);

    // 3. Create grad_output tensor (for backward)
    int64_t grad_shape[2] = {batch, out_features};
    Tensor* grad_out = tensor_create(2, grad_shape, FLOAT32, CUDA);
    tensor_fill(grad_out, 1.0f);

    // 4. Loop over forward/backward
    for (int i = 0; i < num_iterations; i++) {
        printf("\n=== Iteration %d ===\n", i + 1);

        // Forward
        linear->forward(linear, t);
        printf("Linear output (truncated):\n");
        tensor_print(linear->output);

        // Backward
        linear->backward(linear, grad_out);
        printf("Linear grad_input (truncated):\n");
        tensor_print(linear->grad_input);
    }

    // 5. Free memory
    tensor_free(t);
    tensor_free(grad_out);
    tensor_free(linear->output);
    tensor_free(linear->grad_input);
    tensor_free(linear->weights[0]);
    tensor_free(linear->weights[1]);
    free(linear->weights);
    free(linear);

    return 0;
}

void demo_gpu_large() {
    printf("\n=== GPU Demo (Large Tensors) ===\n");

    int64_t shape[3] = {64, 1024, 1024};  // batch=64, features=1024, example extra dim=1024

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
    int64_t shape_D[3] = {64, 1024, 512};  // example matmul shape
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

    tensor_print(X);

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


    Tensor* X_cpu_back = tensor_to_cpu(X_gpu);
    printf("gpu back to cpu:\n");
    tensor_print(X_cpu_back);

    // Free all
    tensor_free(X);
    tensor_free(X_gpu);
    tensor_free(X_cpu);
    tensor_free(X_gpu2);
    tensor_free(Y_gpu);
    tensor_free(Z_gpu);
    tensor_free(Z_cpu);
}

void demo_mlp_cpu() {
    Network* mlp = create_mlp(128, 256, 10, 1, CPU);
    printf("\nMLP was made!\n");

    // Create example input: shape (3,128)
    int64_t shape[3] = {3, 128};
    Tensor* X = tensor_create(2, shape, FLOAT32, CPU);
    tensor_fill(X, 5.0);

    printf("Running forward pass...\n");
    Tensor* y_pred = network_forward(mlp, X, 1);


    int64_t out_shape[2] = {3, 10}; // final layer output shape
    Tensor* grad_output = tensor_create(2, out_shape, FLOAT32, CPU);
    tensor_fill(grad_output, 1.0f);  // pretend dL/dY = 1

    network_backward(mlp, grad_output, 1);

    tensor_print(y_pred);

    printf("Forward output is stored in out->data\n");
}

void demo_mlp_gpu() {
    Network* mlp = create_mlp(5, 10, 10, 1, CUDA);
    printf("\nMLP was made!\n");

    // Create example input: shape (3,128)
    int64_t shape[3] = {3, 5};
    Tensor* X = tensor_create(2, shape, FLOAT32, CUDA);
    tensor_fill(X, 5.0);

    printf("Running forward pass...\n");
    Tensor* y_pred = network_forward(mlp, X, 0);

    int64_t out_shape[2] = {3, 10}; // final  output shape
    Tensor* grad_output = tensor_create(2, out_shape, FLOAT32, CUDA);
    tensor_fill(grad_output, 1.0f);  // pretend dL/dY = 1

    network_backward(mlp, grad_output, 0);
    

    printf("Forward output is stored in out->data\n");
}

void demo_mlp_and_train_cpu() {
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

    int epochs = 10000;
    int print_every = 1000;
    float lr = 0.01f;

    for (int epoch = 0; epoch < epochs; epoch++) {
        Tensor* y_pred = network_forward(net, X, 0);

        float loss = mse_loss(y_pred, y_true, grad_output);

        if ((epoch % print_every) == 0) {
            printf("\n=== Epoch %d ===\n", epoch);
            printf("Loss = %f\n", loss);
        }

        network_backward(net, grad_output, 0);
        optimizer_step(&sgd_opt, lr);
    }

    printf("\nTraining loop complete!\n");
    printf("Forward output of last pass is stored in y_pred->data \n");
}

void demo_mlp_and_train_gpu() {
    int input_dim = 128;
    int hidden_dim = 256;
    int output_dim = 10;
    int hidden_layers = 1;
    Device dev = CUDA;

    Network* net = create_mlp(input_dim, hidden_dim, output_dim, hidden_layers, dev);
    printf("\nMLP was made!\n");

    // create example input and target
    int64_t X_shape[2] = {3, input_dim};
    Tensor* X = tensor_create(2, X_shape, FLOAT32, dev);
    tensor_fill(X, 5.0f);

    int64_t y_shape[2] = {3, output_dim};
    Tensor* y_true = tensor_create(2, y_shape, FLOAT32, dev);
    tensor_fill(y_true, 0.5f);

    // create a tensor to store gradient of loss w.r.t output
    Tensor* grad_output = tensor_create(2, y_shape, FLOAT32, dev);

    Optimizer sgd_opt = {
        .layers = net->layers,
        .n_layers = net->n_layers,
        .update_fn = sgd_update_optimizer,
        .state = NULL,
    };

    int epochs = 10000;
    int print_every = 1000;
    float lr = 0.01f;

    for (int epoch = 0; epoch < epochs; epoch++) {
        Tensor* y_pred = network_forward(net, X, 0);

        float loss = mse_loss(y_pred, y_true, grad_output);

        if ((epoch % print_every) == 0) {
            printf("\n=== Epoch %d ===\n", epoch);
            printf("Loss = %f\n", loss);
        }

        network_backward(net, grad_output, 0);
        optimizer_step(&sgd_opt, lr);
    }


    printf("\nTraining loop complete!\n");
    printf("Forward output of last pass is stored in y_pred->data \n");
}

void demo_tensor_slice() {
    int input_dim = 128;
    int64_t X_shape[2] = {3, input_dim};

    // CPU tensor
    Tensor* X_cpu = tensor_create(2, X_shape, FLOAT32, CPU);
    tensor_print_shape(X_cpu);

    Tensor* X_slice_cpu = tensor_slice(X_cpu, 0);
    tensor_print_shape(X_slice_cpu);

    // Assertions CPU
    assert(X_cpu->ndim == 2);
    assert(X_cpu->shape[0] == 3 && X_cpu->shape[1] == 128);
    assert(X_cpu->device == CPU);

    assert(X_slice_cpu->ndim == 2);
    assert(X_slice_cpu->shape[0] == 1 && X_slice_cpu->shape[1] == 128);
    assert(X_slice_cpu->device == CPU);

    // GPU tensor
    Tensor* X_gpu = tensor_create(2, X_shape, FLOAT32, CUDA);
    tensor_print_shape(X_gpu);

    Tensor* X_slice_gpu = tensor_slice(X_gpu, 0);
    tensor_print_shape(X_slice_gpu);

    // Assertions GPU
    assert(X_gpu->ndim == 2);
    assert(X_gpu->shape[0] == 3 && X_gpu->shape[1] == 128);
    assert(X_gpu->device == CUDA);

    assert(X_slice_gpu->ndim == 2);
    assert(X_slice_gpu->shape[0] == 1 && X_slice_gpu->shape[1] == 128);
    assert(X_slice_gpu->device == CUDA);

    printf("All tensor_slice asserts passed!\n");

    // Free tensors
    tensor_free(X_cpu);
    tensor_free(X_slice_cpu);
    tensor_free(X_gpu);
    tensor_free(X_slice_gpu);
}

void demo_tensor_create_as() {
    int input_dim = 128;
    int64_t X_shape[2] = {3, input_dim};

    Tensor* X = tensor_create(2, X_shape, FLOAT32, CUDA);
    tensor_print_shape(X);

    Tensor* X_copy = tensor_create_as(X);
    tensor_print_shape(X_copy);

    // Assert metadata matches
    assert(X_copy->ndim == X->ndim);
    assert(X_copy->dtype == X->dtype);
    assert(X_copy->device == X->device);
    for (int i = 0; i < X->ndim; i++) {
        assert(X_copy->shape[i] == X->shape[i]);
        assert(X_copy->stride[i] == X->stride[i]);
    }

    Tensor* X_cpu = tensor_create(2, X_shape, FLOAT32, CPU);
    tensor_print_shape(X_cpu);

    Tensor* X_cpu_copy = tensor_create_as(X_cpu);
    tensor_print_shape(X_cpu_copy);

    // Assert metadata matches for CPU copy
    assert(X_cpu_copy->ndim == X_cpu->ndim);
    assert(X_cpu_copy->dtype == X_cpu->dtype);
    assert(X_cpu_copy->device == X_cpu->device);
    for (int i = 0; i < X_cpu->ndim; i++) {
        assert(X_cpu_copy->shape[i] == X_cpu->shape[i]);
        assert(X_cpu_copy->stride[i] == X_cpu->stride[i]);
    }

    printf("All asserts passed!\n");
}

void demo_tensor_copy_slice() {
    int input_dim = 4; 
    int64_t X_shape[2] = {3, input_dim};

    // CPU tensor
    Tensor* X_cpu = tensor_create(2, X_shape, FLOAT32, CPU);
    tensor_fill_random(X_cpu, 0.0f, 10.0f);

    // Destination batch tensor (3 samples)
    Tensor* batch_cpu = tensor_create(2, X_shape, FLOAT32, CPU);
    tensor_fill(batch_cpu, 0.0); 

    // Copy first slice into batch
    Tensor* slice_cpu = tensor_slice(X_cpu, 0);
    tensor_copy_slice(batch_cpu, slice_cpu, 0);

    // Assertions CPU
    assert(batch_cpu->ndim == 2);
    assert(batch_cpu->shape[0] == 3 && batch_cpu->shape[1] == input_dim);
    assert(batch_cpu->device == CPU);

    // Compare first row with slice data
    int64_t slice_size = input_dim;
    float* batch_data = (float*)batch_cpu->data;
    float* slice_data = (float*)slice_cpu->data;
    assert(memcmp(batch_data, slice_data, slice_size * sizeof(float)) == 0);

    printf("CPU tensor_copy_slice passed!\n");

    // Free CPU tensors
    tensor_free(X_cpu);
    tensor_free(slice_cpu);
    tensor_free(batch_cpu);

    // --- GPU version ---
    Tensor* X_gpu = tensor_create(2, X_shape, FLOAT32, CUDA);
    tensor_fill_random(X_gpu, 0.0f, 10.0f);

    Tensor* batch_gpu = tensor_create(2, X_shape, FLOAT32, CUDA);
    tensor_fill(batch_gpu, 0.0);

    Tensor* slice_gpu = tensor_slice(X_gpu, 0);
    tensor_copy_slice(batch_gpu, slice_gpu, 0);

    // Assertions GPU
    assert(batch_gpu->ndim == 2);
    assert(batch_gpu->shape[0] == 3 && batch_gpu->shape[1] == input_dim);
    assert(batch_gpu->device == CUDA);

    // Copy back to CPU to verify contents
    Tensor* batch_gpu_cpu = tensor_to_cpu(batch_gpu);
    Tensor* slice_gpu_cpu = tensor_to_cpu(slice_gpu);
    float* batch_cpu_data = (float*)batch_gpu_cpu->data;
    float* slice_cpu_data = (float*)slice_gpu_cpu->data;

    assert(memcmp(batch_cpu_data, slice_cpu_data, slice_size * sizeof(float)) == 0);

    printf("GPU tensor_copy_slice passed!\n");

    tensor_free(X_gpu);
    tensor_free(slice_gpu);
    tensor_free(batch_gpu);
    tensor_free(batch_gpu_cpu);
    tensor_free(slice_gpu_cpu);
}

void demo_reshape() {
    int input_dim = 4; 
    int64_t X_shape[2] = {3, input_dim}; 

    // ---------------- CPU ----------------
    Tensor* X_cpu = tensor_create(2, X_shape, FLOAT32, CPU);
    tensor_fill_random(X_cpu, 0.0f, 10.0f);

    printf("Original CPU 2D tensor:\n");
    tensor_print_shape(X_cpu);

    int64_t new_shape[3] = {3, 2, 2};
    Tensor* X_cpu_3d = tensor_reshape(X_cpu, 3, new_shape);

    printf("Reshaped CPU 3D tensor:\n");
    tensor_print_shape(X_cpu_3d);

    // Assertions
    int64_t expected_numel = 1;
    for (int i = 0; i < 3; i++) expected_numel *= new_shape[i];
    assert(tensor_numel(X_cpu) == expected_numel && "CPU: Number of elements must match after reshape");
    for (int i = 0; i < 3; i++) assert(X_cpu_3d->shape[i] == new_shape[i] && "CPU: Shape mismatch after reshape");

    printf("CPU reshape assertions passed!\n");

    // Cleanup CPU
    tensor_free(X_cpu_3d); 
    tensor_free(X_cpu);    

    // -------------- CUDA --------------
    Tensor* X_gpu = tensor_create(2, X_shape, FLOAT32, CUDA);
    tensor_fill_random(X_gpu, 0.0f, 10.0f);

    printf("Original CUDA 2D tensor:\n");
    tensor_print_shape(X_gpu);

    Tensor* X_gpu_3d = tensor_reshape(X_gpu, 3, new_shape);
    printf("Reshaped CUDA 3D tensor:\n");
    tensor_print_shape(X_gpu_3d);

    // Assertions
    assert(tensor_numel(X_gpu) == expected_numel && "CUDA: Number of elements must match after reshape");
    for (int i = 0; i < 3; i++) assert(X_gpu_3d->shape[i] == new_shape[i] && "CUDA: Shape mismatch after reshape");

    printf("CUDA reshape assertions passed!\n");

    // Cleanup CUDA
    tensor_free(X_gpu_3d); 
    tensor_free(X_gpu);   
}

int demo_concat() {
    // --- CPU Demo ---
    printf("=== CPU tensor concat demo ===\n");
    int64_t shape[2] = {2, 3}; 

    Tensor* a_cpu = tensor_create_cpu(2, shape, FLOAT32);
    Tensor* b_cpu = tensor_create_cpu(2, shape, FLOAT32);

    tensor_fill(a_cpu, 1.0);
    tensor_fill(b_cpu, 2.0); 

    printf("Tensor A (CPU):\n"); tensor_print(a_cpu);
    printf("Tensor B (CPU):\n"); tensor_print(b_cpu);

    Tensor* c_cpu = tensor_concat(a_cpu, b_cpu, 0);

    printf("Concatenated Tensor (CPU):\n"); tensor_print(c_cpu);

    tensor_free(a_cpu);
    tensor_free(b_cpu);
    tensor_free(c_cpu);

    // --- GPU Demo ---
    printf("\n=== GPU tensor concat demo ===\n");

    Tensor* a_gpu = tensor_create(2, shape, FLOAT32, CUDA);
    Tensor* b_gpu = tensor_create(2, shape, FLOAT32, CUDA);

    tensor_fill(a_gpu, 3.0); 
    tensor_fill(b_gpu, 4.0); 

    printf("Tensor A (GPU):\n"); tensor_print(a_gpu);
    printf("Tensor B (GPU):\n"); tensor_print(b_gpu);

    Tensor* c_gpu = tensor_concat_cuda(a_gpu, b_gpu);

    printf("Concatenated Tensor (GPU):\n"); tensor_print(c_gpu);

    tensor_free(a_gpu);
    tensor_free(b_gpu);
    tensor_free(c_gpu);

    return 0;
}

int demo_metrics()
{
    srand(0);

    int64_t shape[2] = {4, 6};
    int batch = shape[0];
    int labels = shape[1];

    // Create CPU tensors (your API)
    Tensor* y_pred = tensor_create(2, shape, FLOAT32, CPU);
    Tensor* y_true = tensor_create(2, shape, FLOAT32, CPU);

    printf("=== TEST 1: All correct ===\n");
    tensor_fill(y_pred, 1.0);
    tensor_fill(y_true, 1.0);
    printf("Pred:\n"); tensor_print(y_pred);
    printf("True:\n"); tensor_print(y_true);
    printf("Accuracy = %f\n", accuracy(y_pred, y_true));
    printf("MCC = %f\n\n", mcc_score(y_pred, y_true));

    printf("=== TEST 2: All wrong ===\n");
    tensor_fill(y_pred, 1.0);
    tensor_fill(y_true, 0.0);
    printf("Pred:\n"); tensor_print(y_pred);
    printf("True:\n"); tensor_print(y_true);
    printf("Accuracy = %f\n", accuracy(y_pred, y_true));
    printf("MCC = %f\n\n", mcc_score(y_pred, y_true));

    printf("=== TEST 3: Random values ∈ [0,1] ===\n");
    tensor_fill_random(y_pred, 0.0f, 1.0f);
    tensor_fill_random(y_true, 0.0f, 1.0f);
    printf("Pred:\n"); tensor_print(y_pred);
    printf("True:\n"); tensor_print(y_true);
    printf("Accuracy = %f\n", accuracy(y_pred, y_true));
    printf("MCC = %f\n\n", mcc_score(y_pred, y_true));

    tensor_free(y_pred);
    tensor_free(y_true);

    return 0;
}

int demo_metrics_concat() {
    srand(0);

    int64_t shape[2] = {4, 6};
    int batch = shape[0];
    int labels = shape[1];

    // Create two CPU tensors
    Tensor* y_pred1 = tensor_create(2, shape, FLOAT32, CPU);
    Tensor* y_pred2 = tensor_create(2, shape, FLOAT32, CPU);
    Tensor* y_true1 = tensor_create(2, shape, FLOAT32, CPU);
    Tensor* y_true2 = tensor_create(2, shape, FLOAT32, CPU);

    // Fill first half with all 1s (correct)
    tensor_fill(y_pred1, 1.0);
    tensor_fill(y_true1, 1.0);

    // Fill second half with all 0s (incorrect)
    tensor_fill(y_pred2, 0.0);
    tensor_fill(y_true2, 1.0);  // true labels are 1, so this half is wrong

    // Concatenate along dim 0
    Tensor* y_pred = tensor_concat(y_pred1, y_pred2, 0);
    Tensor* y_true = tensor_concat(y_true1, y_true2, 0);

    printf("Concatenated Predictions:\n"); tensor_print(y_pred);
    printf("Concatenated True Labels:\n"); tensor_print(y_true);

    // Compute metrics
    printf("Accuracy = %f\n", accuracy(y_pred, y_true));
    printf("MCC = %f\n", mcc_score(y_pred, y_true));

    // Free tensors
    tensor_free(y_pred1);
    tensor_free(y_pred2);
    tensor_free(y_true1);
    tensor_free(y_true2);
    tensor_free(y_pred);
    tensor_free(y_true);

    return 0;
}

int demo_argmax_dim1() {
    srand(0);

    int64_t shape[2] = {4, 6}; // 4 samples, 6 features each
    Tensor* X = tensor_create(2, shape, FLOAT32, CPU);

    // Fill with random values [0, 1]
    tensor_fill_random(X, 0.0f, 1.0f);

    printf("Input Tensor X:\n");
    tensor_print(X);

    // Argmax along dim=1 (per row)
    Tensor* argmax_out = tensor_argmax_dim1(X);

    printf("Argmax indices (per row):\n");
    tensor_print(argmax_out);

    tensor_free(X);
    tensor_free(argmax_out);

    return 0;
}


void demo_metrics_simple() {
    // --- 4 samples, 3 classes ---
    int64_t shape[2] = {4, 3};
    Tensor* y_pred = tensor_create(2, shape, FLOAT32, CPU);
    Tensor* y_true = tensor_create(1, (int64_t[]){4}, INT64, CPU);

    // Fill predictions
    float pred_data[12] = {0.1, 0.9, 0.0,
                           0.8, 0.1, 0.1,
                           0.2, 0.3, 0.5,
                           0.0, 0.1, 0.9};
    memcpy(y_pred->data, pred_data, sizeof(pred_data));

    int64_t true_data[4] = {1, 1, 2, 2};
    memcpy(y_true->data, true_data, sizeof(true_data));

    Tensor* y_argmax = tensor_argmax_dim1(y_pred);

    printf("Predictions argmax:\n");
    tensor_print(y_argmax);
    printf("True labels:\n");
    tensor_print(y_true);

    printf("Accuracy = %f\n", accuracy(y_argmax, y_true));
    printf("MCC = %f\n", mcc_mc_score(y_argmax, y_true, 3));

    // expected
    // Accuracy = 0.750000
    // MCC = 0.670820

    tensor_free(y_pred);
    tensor_free(y_true);
    tensor_free(y_argmax);
}

void demo_metrics_gpu() {
    // --- 4 samples, 3 classes ---
    int64_t shape[2] = {4, 3};
    Tensor* y_pred_cpu = tensor_create(2, shape, FLOAT32, CPU);
    Tensor* y_true_cpu = tensor_create(1, (int64_t[]){4}, INT64, CPU);

    // Fill predictions (CPU)
    float pred_data[12] = {0.1, 0.9, 0.0,
                           0.8, 0.1, 0.1,
                           0.2, 0.3, 0.5,
                           0.0, 0.1, 0.9};
    memcpy(y_pred_cpu->data, pred_data, sizeof(pred_data));

    int64_t true_data[4] = {1, 1, 2, 2};
    memcpy(y_true_cpu->data, true_data, sizeof(true_data));

    // Copy to GPU
    Tensor* y_pred = tensor_to_cuda(y_pred_cpu);
    Tensor* y_true = tensor_to_cuda(y_true_cpu);

    // Argmax on GPU
    Tensor* y_argmax = tensor_argmax_dim1(y_pred);  // should handle GPU internally

    // Copy back to CPU for printing and metrics
    Tensor* y_argmax_cpu = tensor_to_cpu(y_argmax);
    Tensor* y_true_cpu2 = tensor_to_cpu(y_true);

    printf("Predictions argmax:\n");
    tensor_print(y_argmax_cpu);
    printf("True labels:\n");
    tensor_print(y_true_cpu2);

    // Metrics
    printf("Accuracy = %f\n", accuracy(y_argmax_cpu, y_true_cpu2));
    printf("MCC = %f\n", mcc_mc_score(y_argmax_cpu, y_true_cpu2, 3));

    // Cleanup
    tensor_free(y_pred_cpu);
    tensor_free(y_true_cpu);
    tensor_free(y_pred);
    tensor_free(y_true);
    tensor_free(y_argmax);
    tensor_free(y_argmax_cpu);
    tensor_free(y_true_cpu2);
}


int main() {
    demo_cpu();
    demo_gpu();
    demo_relu_forward_cpu();
    demo_relu_forward_backward_loop(100);
    demo_linear_forward_backward_loop(1);
    demo_gpu_large();
    demo_transfers();
    demo_add_bias_gpu();
    demo_add_bias_cpu();
    demo_mlp_cpu();
    demo_mlp_and_train_cpu();
    demo_mlp_gpu();
    demo_mlp_and_train_gpu();
    demo_tensor_create_as();
    demo_tensor_slice();
    demo_tensor_copy_slice();
    demo_reshape();
    demo_metrics();
    demo_concat();
    demo_concat();
    demo_metrics_concat();
    demo_argmax_dim1();
    demo_metrics_simple();
    demo_metrics_gpu();
    return 0;
}
