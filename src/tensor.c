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

Tensor* tensor_create_as(const Tensor* src) {
    Tensor* t = tensor_create(src->ndim, src->shape, src->dtype, src->device);
    t->requires_grad = src->requires_grad;
    return t;
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

// Returns a new tensor containing only the index-th element along dim 0
Tensor* tensor_slice_cpu(const Tensor* src, int index) {
    if (!src || index < 0 || index >= src->shape[0]) {
        fprintf(stderr, "tensor_slice: index %d out of bounds (0-%" PRId64 ")\n", index, src->shape[0]-1);
        exit(1);
    }

    // Create a new tensor like the src
    Tensor* slice = tensor_create_as(src);

    // first dimension = 1 (single slice)
    slice->shape[0] = 1;

    // Number of elements in one slice (excluding first dimension)
    int64_t slice_size = 1;
    for (int i = 1; i < src->ndim; i++) slice_size *= src->shape[i];

    // copy the slice data
    memcpy(slice->data,
           (char*)src->data + index * slice_size * dtype_size(src->dtype),
           slice_size * dtype_size(src->dtype));

    return slice;
}

Tensor* tensor_slice(const Tensor* src, int index) {

    switch (src->device) {
        case CPU:
            return tensor_slice_cpu(src, index);
        case CUDA:
            return tensor_slice_cuda(src, index);
        default:
            fprintf(stderr, "tensor_slice: unknown device \n");
            exit(1);
    }
}

// Copy a single slice into a batch tensor at position dest_index (CPU)
void tensor_copy_slice_cpu(Tensor* dest, const Tensor* src, int dest_index) {
    if (!dest || !src || dest->ndim != src->ndim) {
        fprintf(stderr, "tensor_copy_slice: invalid tensor(s)\n");
        exit(1);
    }
    if (dest_index < 0 || dest_index >= dest->shape[0]) {
        fprintf(stderr, "tensor_copy_slice: index %d out of bounds (0-%" PRId64 ")\n", dest_index, dest->shape[0]-1);
        exit(1);
    }

    // Check inner dimensions match
    for (int i = 1; i < dest->ndim; i++) {
        if (dest->shape[i] != src->shape[i]) {
            fprintf(stderr, "tensor_copy_slice: shape mismatch at dim %d\n", i);
            exit(1);
        }
    }

    // Number of elements in one slice
    int64_t slice_size = 1;
    for (int i = 1; i < src->ndim; i++) slice_size *= src->shape[i];

    memcpy((char*)dest->data + dest_index * slice_size * dtype_size(dest->dtype),
           src->data,
           slice_size * dtype_size(dest->dtype));
}

void tensor_copy_slice(Tensor* dest, const Tensor* src, int dest_index) {

    switch (src->device) {
        case CPU:
            tensor_copy_slice_cpu(dest, src, dest_index);
            break;
        case CUDA:
            tensor_copy_slice_cuda(dest, src, dest_index);
            break;
        default:
            fprintf(stderr, "tensor_copy_slice: unknown device\n");
            break;
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
    if (!t) return;

    // only free data if not a view
    if (!t->is_view && t->data) {
        free(t->data);
        t->data = NULL;
    }

    if (t->grad) {
        free(t->grad);
        t->grad = NULL;
    }

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

// void tensor_print_recursive(const Tensor* t, int dim, int64_t offset) {
//     if (dim == t->ndim) {
//         // Base case: print a single element
//         if (t->dtype == FLOAT32)
//             printf("%6.2f", ((float*)t->data)[offset]);
//         else if (t->dtype == INT32)
//             printf("%6d", ((int*)t->data)[offset]);
//         return;
//     }

//     printf("[");
//     int64_t limit = t->shape[dim];

//     for (int64_t i = 0; i < limit; i++) {
//         // for large dimensions, skip middle elements
//         if (limit > MAX_PRINT * 2 && i == MAX_PRINT) {
//             printf(" ... ");
//             i = limit - MAX_PRINT;
//         }

//         // new line for higher dims
//         if (dim < t->ndim - 1 && i > 0)
//             printf("\n ");

//         tensor_print_recursive(t, dim + 1, offset + i * t->stride[dim]);

//         if (i < limit - 1) printf(", ");
//     }
//     printf("]");
// }

void tensor_print_recursive(const Tensor* t, int dim, int64_t offset) {
    if (dim == t->ndim) {
        // Base case: print a single element
        if (t->dtype == FLOAT32)
            printf("%6.2f", ((float*)t->data)[offset]);
        else if (t->dtype == INT32)
            printf("%6d", ((int*)t->data)[offset]);
        else if (t->dtype == INT64)           // <--- added
            printf("%6" PRId64, ((int64_t*)t->data)[offset]);
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

void tensor_print_shape(const Tensor* t) { 
    printf("Tensor(");
    for (int i = 0; i < t->ndim; i++) {
        printf("%" PRId64, t->shape[i]);
        if (i < t->ndim - 1) printf(", ");
    }
    printf(")\n");
}

Tensor* tensor_reshape(Tensor* src, int new_ndim, const int64_t* new_shape) {
    int64_t old_numel = tensor_numel(src);

    int64_t known = 1;
    int neg1_index = -1;
    for (int i = 0; i < new_ndim; i++) {
        if (new_shape[i] == -1) {
            if (neg1_index != -1) {
                fprintf(stderr, "tensor_reshape: only one -1 dimension allowed\n");
                exit(1);
            }
            neg1_index = i;
        } else {
            known *= new_shape[i];
        }
    }

    int64_t inferred_shape[16];  // max dims < 16
    for (int i = 0; i < new_ndim; i++) inferred_shape[i] = new_shape[i];

    if (neg1_index != -1) {
        if (old_numel % known != 0) {
            fprintf(stderr, "tensor_reshape: cannot infer -1 dimension (not divisible)\n");
            exit(1);
        }
        inferred_shape[neg1_index] = old_numel / known;
    }

    int64_t new_numel = 1;
    for (int i = 0; i < new_ndim; i++) new_numel *= inferred_shape[i];
    if (new_numel != old_numel) {
        fprintf(stderr, "tensor_reshape: mismatch numel (%ld vs %ld)\n", old_numel, new_numel);
        exit(1);
    }

    Tensor* t = malloc(sizeof(Tensor));
    t->ndim = new_ndim;
    t->dtype = src->dtype;
    t->device = src->device;     
    t->requires_grad = src->requires_grad;
    t->is_view = 1;             

    for (int i = 0; i < new_ndim; i++) t->shape[i] = inferred_shape[i];

    // compute row major strides
    int64_t stride = 1;
    for (int i = new_ndim - 1; i >= 0; i--) {
        t->stride[i] = stride;
        stride *= t->shape[i];
    }

    t->data = src->data;
    t->grad = NULL;  

    return t;
}

Tensor* tensor_concat_cpu(const Tensor* a, const Tensor* b, int dim) {
    if (dim != 0) {
        fprintf(stderr, "tensor_concat: only dim=0 supported\n");
        exit(1);
    }

    if (a->ndim != b->ndim) {
        fprintf(stderr, "tensor_concat: tensors must have same ndim\n");
        exit(1);
    }

    for (int i = 1; i < a->ndim; i++) {
        if (a->shape[i] != b->shape[i]) {
            fprintf(stderr, "tensor_concat: shapes must match except dim 0\n");
            exit(1);
        }
    }

    // new shape
    int64_t new_shape[MAX_DIMS];
    new_shape[0] = a->shape[0] + b->shape[0];
    for (int i = 1; i < a->ndim; i++) new_shape[i] = a->shape[i];

    Tensor* out = tensor_create_cpu(a->ndim, new_shape, a->dtype);

    size_t elem_size = dtype_size(a->dtype);
    int64_t numel_a = tensor_numel(a);
    int64_t numel_b = tensor_numel(b);

    memcpy(out->data, a->data, numel_a * elem_size);
    memcpy((char*)out->data + numel_a * elem_size, b->data, numel_b * elem_size);

    return out;
}


Tensor* tensor_concat(const Tensor* a, const Tensor* b, int dim) {
    switch (a->device) {
        case CPU:
            return tensor_concat_cpu(a, b, dim);
        case CUDA:
            return tensor_concat_cuda(a, b); //TODO add more dims, only 0 now
        default:
            fprintf(stderr, "tensor_concat: unknown device\n");
            exit(1);
    }
}