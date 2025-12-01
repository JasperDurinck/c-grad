#include "../include/tensor.h"
#include "../include/loss_fns_cu.h"
#include <math.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>


float mse_loss_cpu(Tensor* y_pred, Tensor* y_true, Tensor* grad_out) {
    float* yp = (float*)y_pred->data;
    float* yt = (float*)y_true->data;
    float* go = (float*)grad_out->data;

    int64_t N = 1;
    for (int i = 0; i < y_pred->ndim; i++) N *= y_pred->shape[i];

    float loss = 0.0f;
    for (int64_t i = 0; i < N; i++) {
        float diff = yp[i] - yt[i];
        loss += diff * diff;
        go[i] = 2.0f * diff / y_pred->shape[0];  // grad w.r.t batch
    }

    return loss / y_pred->shape[0];
}

float mse_loss(Tensor* y_pred, Tensor* y_true, Tensor* grad_out) {
    switch (y_pred->device) {
        case CPU:
            return mse_loss_cpu(y_pred, y_true, grad_out);
            break;
        case CUDA:
            return mse_loss_cuda(y_pred, y_true, grad_out);
            break;
        default:
            fprintf(stderr, "mse_loss: unknown device!\n");
            exit(1);
    }
}

float binary_cross_entropy_loss_cpu(Tensor* y_pred, Tensor* y_true, Tensor* grad_out) {
    float* yp = (float*)y_pred->data;
    float* yt = (float*)y_true->data;
    float* go = (float*)grad_out->data;

    int64_t N = 1;
    for (int i = 0; i < y_pred->ndim; i++) N *= y_pred->shape[i];

    float loss = 0.0f;
    for (int64_t i = 0; i < N; i++) {
        // clamp preds for num stability
        float p = yp[i];
        if (p < FLT_EPSILON) p = FLT_EPSILON;
        if (p > 1.0f - FLT_EPSILON) p = 1.0f - FLT_EPSILON;

        loss += - (yt[i] * logf(p) + (1.0f - yt[i]) * logf(1.0f - p));

        // gradient w.r.t prediction
        go[i] = (p - yt[i]) / (p * (1.0f - p) * y_pred->shape[0]);  // normalized by batch
    }

    return loss / y_pred->shape[0];
}

float binary_cross_entropy_loss(Tensor* y_pred, Tensor* y_true, Tensor* grad_out) {
    switch (y_pred->device) {
        case CPU:
            return binary_cross_entropy_loss_cpu(y_pred, y_true, grad_out);
            break;
        case CUDA:
            return binary_cross_entropy_loss_cuda(y_pred, y_true, grad_out);
            break;
        default:
            fprintf(stderr, "binary_cross_entropy_loss: unknown device!\n");
            exit(1);
    }
}