#ifdef __cplusplus
extern "C" {
#endif

#ifndef LOSS_FN_CU_H
#define LOSS_FN_CU_H

#include "../include/tensor.h"

float mse_loss_cuda(Tensor* y_pred, Tensor* y_true, Tensor* grad_out);

float binary_cross_entropy_loss_cuda(Tensor* y_pred, Tensor* y_true, Tensor* grad_out);

#endif

#ifdef __cplusplus
}
#endif