#ifdef __cplusplus
extern "C" {
#endif

#ifndef METRICS_H
#define METRICS_H

#include "../include/tensor.h"


float accuracy_cuda(Tensor* y_pred, Tensor* y_true);
float mcc_score_cuda(Tensor* y_pred, Tensor* y_true);
float mcc_mc_score_cuda(Tensor* y_pred, Tensor* y_true, int num_classes);

#endif

#ifdef __cplusplus
}
#endif