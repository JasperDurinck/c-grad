#include "../include/tensor.h"
#include "../include/metrics_cu.h"
#include <math.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

float accuracy_cpu(Tensor* y_pred, Tensor* y_true) {
    int64_t N = y_true->shape[0];
    int64_t correct = 0;

    if (y_pred->ndim == 2) {
        for (int64_t i = 0; i < N; i++) {
            float* row = (float*)y_pred->data + i * y_pred->shape[1];
            int max_idx = 0;
            float max_val = row[0];
            for (int64_t j = 1; j < y_pred->shape[1]; j++) {
                if (row[j] > max_val) {
                    max_val = row[j];
                    max_idx = j;
                }
            }
            int truev = (int)((int64_t*)y_true->data)[i]; // assuming int64 labels
            if (max_idx == truev) correct++;
        }
    } else {
        // y_pred is already argmax (1D)
        int64_t* yp = (int64_t*)y_pred->data;
        int64_t* yt = (int64_t*)y_true->data;
        for (int64_t i = 0; i < N; i++)
            if (yp[i] == yt[i]) correct++;
    }

    return (float)correct / (float)N;
}


float accuracy(Tensor* y_pred, Tensor* y_true) {
    switch (y_pred->device) {
        case CPU:
            return accuracy_cpu(y_pred, y_true);
        case CUDA:
            return accuracy_cuda(y_pred, y_true);
        default:
            fprintf(stderr, "mcc_score: unknown device!\n");
            exit(1);
    }
}


float mcc_score_cpu(Tensor* y_pred, Tensor* y_true) {
    float* yp = (float*)y_pred->data;
    float* yt = (float*)y_true->data;

    int64_t batch = y_pred->shape[0];
    int64_t labels = y_pred->shape[1];

    float mcc_sum = 0.0f;

    for (int64_t l = 0; l < labels; l++) {
        int64_t tp = 0, tn = 0, fp = 0, fn = 0;

        for (int64_t i = 0; i < batch; i++) {
            int64_t idx = i * labels + l;

            // Threshold at 0.5 for prediction
            int pred = yp[idx] >= 0.5f ? 1 : 0;
            int truth = yt[idx] >= 0.5f ? 1 : 0;

            if (pred && truth) tp++;
            else if (pred && !truth) fp++;
            else if (!pred && truth) fn++;
            else tn++;
        }

        // Use float for numerator/denominator to avoid integer overflow
        float num = (float)(tp * tn - fp * fn);
        float denom = sqrtf(
            (float)(tp + fp) *
            (float)(tp + fn) *
            (float)(tn + fp) *
            (float)(tn + fn)
        );

        // avoid division by zero
        float mcc = (denom < 1e-8f ? 0.0f : num / denom);

        mcc_sum += mcc;
    }

    return mcc_sum / (float)labels;
}


float mcc_score(Tensor* y_pred, Tensor* y_true) {
    switch (y_pred->device) {
        case CPU:
            return mcc_score_cpu(y_pred, y_true);
        case CUDA:
            return mcc_score_cuda(y_pred, y_true);
        default:
            fprintf(stderr, "mcc_score: unknown device!\n");
            exit(1);
    }
}


float mcc_mc_score_cpu(Tensor* y_pred, Tensor* y_true, int num_classes) {
    int64_t N = y_pred->shape[0];
    int64_t* yp = (int64_t*)y_pred->data;
    int64_t* yt = (int64_t*)y_true->data;

    int64_t* conf = calloc(num_classes * num_classes, sizeof(int64_t));

    for (int64_t i = 0; i < N; i++) {
        conf[yp[i]*num_classes + yt[i]]++;
    }

    int64_t sum_pred[num_classes], sum_true[num_classes];
    int64_t c = 0;

    for (int k = 0; k < num_classes; k++) {
        sum_pred[k] = sum_true[k] = 0;
        for (int j = 0; j < num_classes; j++) {
            sum_pred[k] += conf[k*num_classes + j];
            sum_true[k] += conf[j*num_classes + k];
        }
        c += conf[k*num_classes + k];
    }

    int64_t s_pred = 0, s_true = 0, prod_sum = 0;
    for (int k = 0; k < num_classes; k++) {
        s_pred += sum_pred[k] * sum_pred[k];
        s_true += sum_true[k] * sum_true[k];
        prod_sum += sum_pred[k] * sum_true[k];
    }

    float numerator = (float)(c * N - prod_sum);
    float denominator = sqrtf((float)(N*N - s_pred) * (float)(N*N - s_true));

    free(conf);
    if (denominator < 1e-8f) return 0.0f;
    return numerator / denominator;
}

float mcc_mc_score(Tensor* y_pred, Tensor* y_true, int num_classes) {
    switch (y_pred->device) {
        case CPU:
            return mcc_mc_score_cpu(y_pred, y_true, num_classes);
        case CUDA:
            return mcc_mc_score_cuda(y_pred, y_true, num_classes);
        default:
            fprintf(stderr, "mcc_score: unknown device!\n");
            exit(1);
    }
}