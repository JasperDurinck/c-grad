#include "../include/nn_layers.h"
#include "../include/tensor.h"
#include "../include/optimizers.h"
#include <math.h>

void optimizer_step(Optimizer* opt, float lr) {
    if (opt && opt->update_fn) {
        opt->update_fn(opt, lr);
    }
}

void sgd_update_optimizer(Optimizer* opt, float lr) {
    for (int i = 0; i < opt->n_layers; i++) {
        Layer* layer = opt->layers[i];
        for (int w = 0; w < layer->n_weights; w++) {
            Tensor* W = layer->weights[w];
            if (!W->grad) continue;

            Tensor* G = (Tensor*)W->grad;
            float* w_data = (float*)W->data;
            float* g_data = (float*)G->data;

            int64_t N = 1;
            for (int d = 0; d < W->ndim; d++) N *= W->shape[d];

            for (int64_t j = 0; j < N; j++) {
                w_data[j] -= lr * g_data[j];
            }
        }
    }
}

void adam_update_optimizer(Optimizer* opt, float lr) {
    AdamState** states = (AdamState**)opt->state; // array of AdamState per weight

    int state_idx = 0;
    for (int i = 0; i < opt->n_layers; i++) {
        Layer* layer = opt->layers[i];
        for (int w = 0; w < layer->n_weights; w++) {
            Tensor* W = layer->weights[w];
            if (!W->grad) continue;

            AdamState* st = states[state_idx++];
            if (!st) continue;

            float* w_data = (float*)W->data;
            Tensor* G = (Tensor*)W->grad;
            float* g_data = (float*)G->data;

            int64_t N = 1;
            for (int d = 0; d < W->ndim; d++) N *= W->shape[d];

            if (!st->m) st->m = tensor_create(W->ndim, W->shape, FLOAT32, W->device);
            if (!st->v) st->v = tensor_create(W->ndim, W->shape, FLOAT32, W->device);

            float* m = (float*)st->m->data;
            float* v = (float*)st->v->data;

            st->t += 1;

            float beta1 = 0.9f;
            float beta2 = 0.999f;
            float eps = 1e-8f;

            for (int64_t j = 0; j < N; j++) {
                m[j] = beta1 * m[j] + (1 - beta1) * g_data[j];
                v[j] = beta2 * v[j] + (1 - beta2) * g_data[j] * g_data[j];

                float m_hat = m[j] / (1 - powf(beta1, st->t));
                float v_hat = v[j] / (1 - powf(beta2, st->t));

                w_data[j] -= lr * m_hat / (sqrtf(v_hat) + eps);
            }
        }
    }
}
