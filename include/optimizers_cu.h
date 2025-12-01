#ifdef __cplusplus
extern "C" {
#endif

#ifndef OPTIM__CU_H
#define OPTIM__CU_H

#include "../include/optimizers.h"

void sgd_update_optimizer_cuda(Optimizer* opt, float lr);

#endif

#ifdef __cplusplus
}
#endif