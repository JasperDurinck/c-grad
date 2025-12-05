#ifdef __cplusplus
extern "C" {
#endif

#ifndef METRICS_H
#define METRICS_H

#include "../include/tensor.h"


float accuracy_cpu(Tensor* y_pred, Tensor* y_true);
float accuracy(Tensor* y_pred, Tensor* y_true);
float mcc_score_cpu(Tensor* y_pred, Tensor* y_true);
float mcc_score(Tensor* y_pred, Tensor* y_true);
float mcc_mc_score_cpu(Tensor* y_pred, Tensor* y_true, int num_classes); 
float mcc_mc_score(Tensor* y_pred, Tensor* y_true, int num_classes); 

#endif

#ifdef __cplusplus
}
#endif