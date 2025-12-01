#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdlib.h>  
#include <time.h>    
#include "../include/tensor.h"
#include "../include/tensor_cu.h"

size_t dtype_size(TensorType dtype) {
    switch (dtype) {
        case FLOAT32:  return sizeof(float);
        case FLOAT16:
        case BFLOAT16: return 2;
        case INT32:    return sizeof(int32_t);
        case INT64:    return sizeof(int64_t);
        default:       return 0;
    }
}

int64_t tensor_numel(const Tensor* t) {
    int64_t n = 1;
    for (int i = 0; i < t->ndim; i++) n *= t->shape[i];
    return n;
}

Tensor* tensor_create_cpu(int ndim, const int64_t* shape, TensorType dtype) {
    Tensor* t = calloc(1, sizeof(Tensor));
    t->ndim = ndim;
    t->dtype = dtype;
    t->device = CPU;
    t->requires_grad = 1;

    memcpy(t->shape, shape, ndim * sizeof(int64_t));

    if (ndim > 0) {
        t->stride[ndim - 1] = 1;
        for (int i = ndim - 2; i >= 0; i--) {
            t->stride[i] = t->stride[i + 1] * t->shape[i + 1];
        }
    }

    int64_t numel = tensor_numel(t);
    t->data = calloc(numel, dtype_size(dtype));  

    return t;
}

Tensor* tensor_create(int ndim, const int64_t* shape, TensorType dtype, Device dev) {
    if (dev == CPU) {
        return tensor_create_cpu(ndim, shape, dtype);
    } else if (dev == CUDA) {
        return tensor_create_cuda(ndim, shape, dtype);
    } else {
        return NULL; // unsupported
    }
}

Tensor* tensor_create_scalar_cpu(TensorType dtype, Device device) {
    Tensor* t = calloc(1, sizeof(Tensor));
    t->ndim = 0;              
    t->dtype = dtype;
    t->device = device;
    t->requires_grad = 0;     

    // numel = 1
    int64_t numel = 1;
    t->data = calloc(numel, dtype_size(dtype)); 
    t->grad = NULL;

    return t;
}

Tensor* tensor_create_scalar(TensorType dtype, Device device) {
    switch (device) {
        case CPU:
            return tensor_create_scalar_cpu(dtype, CPU);
        case CUDA:
            return tensor_create_scalar_cuda(dtype); 
        default:
            fprintf(stderr, "tensor_create_scalar_opt: unknown device\n");
            return NULL;
    }
}

void tensor_fill_cpu(Tensor* t, double value) {
    int64_t n = tensor_numel(t);
    if (t->dtype == FLOAT32) {
        float* d = (float*)t->data;
        float v = (float)value;
        for (int64_t i = 0; i < n; i++) d[i] = v;
    }
}

void tensor_fill(Tensor* t, double value) {
    if (t->device == CPU)
        tensor_fill_cpu(t, value);
    else if (t->device == CUDA)
        tensor_fill_cuda(t, value);
}

void tensor_fill_random_cpu(Tensor* t, float min_val, float max_val) {
    int64_t n = tensor_numel(t);

    if (t->dtype == FLOAT32) {
        float* d = (float*)t->data;
        for (int64_t i = 0; i < n; i++) {
            float r = (float)rand() / (float)RAND_MAX; // random in [0,1]
            d[i] = min_val + r * (max_val - min_val);  // scale to [min_val, max_val]
        }
    }
    else if (t->dtype == FLOAT32) {
        double* d = (double*)t->data;
        for (int64_t i = 0; i < n; i++) {
            double r = (double)rand() / (double)RAND_MAX;
            d[i] = min_val + r * (max_val - min_val);
        }
    }
    else if (t->dtype == INT32) {
        int* d = (int*)t->data;
        for (int64_t i = 0; i < n; i++) {
            d[i] = min_val + rand() % ((int)(max_val - min_val + 1));
        }
    }
}

void tensor_fill_random(Tensor* t,  float min_val, float max_val) {
    if (!t) return;

    switch (t->device) {
        case CPU:
            tensor_fill_random_cpu(t, min_val, max_val);
            break;
        case CUDA:
            tensor_fill_random_cuda(t, min_val, max_val);
            break;
        default:
            fprintf(stderr, "tensor_fill_random: unknown device\n");
            break;
    }
}

void tensor_free_cpu(Tensor* t) {
    if (t->data) free(t->data);
    if (t->grad) free(t->grad);
    free(t);
}

void tensor_free(Tensor* t) {
    if (!t) return;

    switch (t->device) {
        case CPU:
            tensor_free_cpu(t);
            break;
        case CUDA:
            tensor_free_cuda(t);
            break;
        default:
            fprintf(stderr, "tensor_free: unknown device\n");
            break;
    }
}

#define MAX_PRINT 5  // maximum elements per dimension to print

void tensor_print_recursive(const Tensor* t, int dim, int64_t offset) {
    if (dim == t->ndim) {
        // Base case: print a single element
        if (t->dtype == FLOAT32)
            printf("%6.2f", ((float*)t->data)[offset]);
        else if (t->dtype == INT32)
            printf("%6d", ((int*)t->data)[offset]);
        return;
    }

    printf("[");
    int64_t limit = t->shape[dim];

    for (int64_t i = 0; i < limit; i++) {
        // for large dimensions, skip middle elements
        if (limit > MAX_PRINT * 2 && i == MAX_PRINT) {
            printf(" ... ");
            i = limit - MAX_PRINT;
        }

        // new line for higher dims
        if (dim < t->ndim - 1 && i > 0)
            printf("\n ");

        tensor_print_recursive(t, dim + 1, offset + i * t->stride[dim]);

        if (i < limit - 1) printf(", ");
    }
    printf("]");
}

void tensor_print(const Tensor* t) {
    const Tensor* to_print = t;
    Tensor* tmp = NULL;

    printf("Tensor(");
    for (int i = 0; i < t->ndim; i++) {
        printf("%" PRId64, t->shape[i]);
        if (i < t->ndim - 1) printf(", ");
    }
    printf(")\n");

    // If the tensor is on CUDA make a CPU copy
    if (t->device == CUDA) {
        tmp = tensor_to_cpu(t);
        to_print = tmp;
    }

    tensor_print_recursive(to_print, 0, 0);
    printf("\n");

    if (tmp != NULL) {
        tensor_free(tmp);
    }
}

void tensor_print_shape(const Tensor* t) { // make copy to cpu if on gpu
    printf("Tensor(");
    for (int i = 0; i < t->ndim; i++) {
        printf("%" PRId64, t->shape[i]);
        if (i < t->ndim - 1) printf(", ");
    }
    printf(")\n");
}
