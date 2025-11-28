#ifdef __cplusplus
extern "C" {
#endif

#ifndef OPTIM_H
#define OPTIM_H

#include "../include/nn_layers.h"

typedef struct Optimizer {
    Layer** layers;                 // array of layers to update
    int n_layers;                   // number of layers
    void (*update_fn)(struct Optimizer*, float lr);  // pointer to update function
    void* state;                    // optional state for optimizer (Adam)
} Optimizer;

typedef struct AdamState {
    Tensor* m;   // first moment
    Tensor* v;   // second moment
    int t;       // timestep
} AdamState;


void optimizer_step(Optimizer* opt, float lr);
void sgd_update_optimizer(Optimizer* opt, float lr);
void adam_update_optimizer(Optimizer* opt, float lr);


#endif

#ifdef __cplusplus
}
#endif