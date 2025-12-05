#include "../include/tensor.h"
#include <math.h>
#include <float.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

//  Accuracy 
__global__ void accuracy_kernel(
    const float* y_pred,
    const float* y_true,
    int64_t N,
    unsigned long long* correct_count   
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    int pred = y_pred[idx] >= 0.5f;
    int truth = (int)y_true[idx];

    if (pred == truth) {
        atomicAdd(correct_count, 1ULL);
    }
}

float accuracy_cuda(Tensor* y_pred, Tensor* y_true) {
    int64_t N = tensor_numel(y_pred);
    if (N == 0) return 0.0f;

    const float* d_yp = (float*)y_pred->data;
    const float* d_yt = (float*)y_true->data;

    unsigned long long h_correct = 0ULL;
    unsigned long long* d_correct;
    cudaMalloc(&d_correct, sizeof(unsigned long long));
    cudaMemcpy(d_correct, &h_correct, sizeof(unsigned long long), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (int)((N + threads - 1) / threads);

    accuracy_kernel<<<blocks, threads>>>(d_yp, d_yt, N, d_correct);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_correct, d_correct, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaFree(d_correct);

    return (float)h_correct / (float)N;
}


//  MCC (multilabel) 
__global__ void mcc_confusion_kernel(
    const float* y_pred,
    const float* y_true,
    int batch,
    int labels,
    unsigned long long* tp,
    unsigned long long* tn,
    unsigned long long* fp,
    unsigned long long* fn
) {
    int sample = blockIdx.x;
    int label = threadIdx.x;

    if (sample >= batch || label >= labels) return;

    int idx = sample * labels + label;

    int pred = y_pred[idx] >= 0.5f;
    int truth = (int)y_true[idx];

    if (pred == 1 && truth == 1) atomicAdd(&tp[label], 1ULL);
    else if (pred == 1 && truth == 0) atomicAdd(&fp[label], 1ULL);
    else if (pred == 0 && truth == 1) atomicAdd(&fn[label], 1ULL);
    else atomicAdd(&tn[label], 1ULL);
}

float mcc_score_cuda(Tensor* y_pred, Tensor* y_true) {
    int batch = y_pred->shape[0];
    int labels = y_pred->shape[1];

    const float* d_yp = (float*)y_pred->data;
    const float* d_yt = (float*)y_true->data;

    // allocate counters (unsigned long long for atomicAdd)
    unsigned long long *tp, *tn, *fp, *fn;
    cudaMalloc(&tp, labels * sizeof(unsigned long long));
    cudaMalloc(&tn, labels * sizeof(unsigned long long));
    cudaMalloc(&fp, labels * sizeof(unsigned long long));
    cudaMalloc(&fn, labels * sizeof(unsigned long long));

    cudaMemset(tp, 0, labels * sizeof(unsigned long long));
    cudaMemset(tn, 0, labels * sizeof(unsigned long long));
    cudaMemset(fp, 0, labels * sizeof(unsigned long long));
    cudaMemset(fn, 0, labels * sizeof(unsigned long long));

    // Note: threadsPerBlock must not exceed device limit (usually 1024) #TODO.
    // If labels > maxThreadsPerBlock, not supported #TODO
    dim3 blocks(batch);
    dim3 threads(labels);
    mcc_confusion_kernel<<<blocks, threads>>>(
        d_yp, d_yt, batch, labels, tp, tn, fp, fn
    );
    cudaDeviceSynchronize();

    // copy counts back to host
    unsigned long long* h_tp = (unsigned long long*)malloc(labels * sizeof(unsigned long long));
    unsigned long long* h_tn = (unsigned long long*)malloc(labels * sizeof(unsigned long long));
    unsigned long long* h_fp = (unsigned long long*)malloc(labels * sizeof(unsigned long long));
    unsigned long long* h_fn = (unsigned long long*)malloc(labels * sizeof(unsigned long long));

    cudaMemcpy(h_tp, tp, labels * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_tn, tn, labels * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fp, fp, labels * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fn, fn, labels * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    // free device counters
    cudaFree(tp); cudaFree(tn); cudaFree(fp); cudaFree(fn);

    // compute MCC per label on CPU (use double for numeric stability)
    double mcc_sum = 0.0;

    for (int l = 0; l < labels; l++) {
        double TP = (double)h_tp[l];
        double TN = (double)h_tn[l];
        double FP = (double)h_fp[l];
        double FN = (double)h_fn[l];

        double num = TP * TN - FP * FN;
        double den = sqrt(
            (TP + FP) *
            (TP + FN) *
            (TN + FP) *
            (TN + FN)
        );

        double mcc = (den == 0.0 ? 0.0 : num / den);
        mcc_sum += mcc;
    }

    free(h_tp); free(h_tn); free(h_fp); free(h_fn);

    return (float)(mcc_sum / (double)labels);
}

__global__ void multiclass_confusion_kernel(
    const int64_t* y_pred,
    const int64_t* y_true,
    int64_t N,
    int num_classes,
    unsigned long long* conf  // flatten num_classes x num_classes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    int pred = y_pred[idx];
    int truev = y_true[idx];

    atomicAdd(&conf[pred * num_classes + truev], 1ULL);
}

float mcc_mc_score_cuda(Tensor* y_pred, Tensor* y_true, int num_classes) {
    int64_t N = y_pred->shape[0];
    int64_t* d_yp = (int64_t*)y_pred->data;
    int64_t* d_yt = (int64_t*)y_true->data;

    unsigned long long* d_conf;
    cudaMalloc(&d_conf, num_classes * num_classes * sizeof(unsigned long long));
    cudaMemset(d_conf, 0, num_classes * num_classes * sizeof(unsigned long long));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    multiclass_confusion_kernel<<<blocks, threads>>>(d_yp, d_yt, N, num_classes, d_conf);
    cudaDeviceSynchronize();

    unsigned long long* h_conf = (unsigned long long*)malloc(num_classes * num_classes * sizeof(unsigned long long));
    cudaMemcpy(h_conf, d_conf, num_classes * num_classes * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaFree(d_conf);

    unsigned long long c = 0;
    unsigned long long sum_pred[num_classes];
    unsigned long long sum_true[num_classes];
    for (int k = 0; k < num_classes; k++) { sum_pred[k] = 0; sum_true[k] = 0; }

    for (int i = 0; i < num_classes; i++) {
        for (int j = 0; j < num_classes; j++) {
            unsigned long long val = h_conf[i * num_classes + j];
            sum_pred[i] += val;
            sum_true[j] += val;
            if (i == j) c += val;
        }
    }

    unsigned long long s_pred = 0, s_true = 0, prod_sum = 0;
    for (int k = 0; k < num_classes; k++) {
        s_pred += sum_pred[k] * sum_pred[k];
        s_true += sum_true[k] * sum_true[k];
        prod_sum += sum_pred[k] * sum_true[k];
    }

    float numerator = (float)(c * N - prod_sum);
    float denominator = sqrtf((float)(N * N - s_pred) * (float)(N * N - s_true));

    free(h_conf);

    if (denominator < 1e-8f) return 0.0f;
    return numerator / denominator;
}

#ifdef __cplusplus
}
#endif
