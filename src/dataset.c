#include "dataset.h"
#include "tensor.h"
#include <stdint.h>
#include <stdio.h>
#include <time.h> 
#include <stdlib.h>


DataLoader* dataloader_create(Dataset* ds, int batch_size, int shuffle) {
    DataLoader* dl = malloc(sizeof(DataLoader));
    dl->dataset = ds;
    dl->batch_size = batch_size;
    dl->shuffle = shuffle;
    dl->index = 0;

    dl->order = malloc(sizeof(int) * ds->length);
    for (int i = 0; i < ds->length; i++) dl->order[i] = i;

    if (shuffle) shuffle_array(dl->order, ds->length);

    return dl;
}

int dataloader_next(DataLoader* dl, Tensor** X, Tensor** Y) {
    if (dl->index >= dl->dataset->length) return 0;

    int start = dl->index;
    int end = start + dl->batch_size;
    if (end > dl->dataset->length) end = dl->dataset->length;

    int batch = end - start;

    int64_t x_shape[4] = {batch, 1, 28, 28};
    int64_t y_shape[1] = {batch};

    *X = tensor_create(4, x_shape, FLOAT32, CPU);
    *Y = tensor_create(1, y_shape, INT64, CPU);

    for (int i = 0; i < batch; i++) {
        int idx = dl->order[start + i];

        Tensor* xi; 
        Tensor* yi;
        dl->dataset->get_item(dl->dataset->data, idx, &xi, &yi);

        tensor_copy_slice(*X, xi, i);
        tensor_copy_slice(*Y, yi, i);

        tensor_free(xi);
        tensor_free(yi);
    }

    dl->index = end;
    return 1;
}

void dataloader_reset(DataLoader* dl) {
    dl->index = 0;
    if (dl->shuffle) {
        shuffle_array(dl->order, dl->dataset->length);
    }
}

// MNIST 

void shuffle_array(int* arr, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }
}

Tensor* load_mnist_images(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        printf("Cannot open %s\n", path);
        return NULL;
    }

    uint32_t magic   = read_be_uint32(f);
    uint32_t n_imgs  = read_be_uint32(f);
    uint32_t n_rows  = read_be_uint32(f);
    uint32_t n_cols  = read_be_uint32(f);

    if (magic != 2051) {
        printf("Invalid MNIST image magic: %u\n", magic);
        fclose(f);
        return NULL;
    }

    int64_t shape[4] = {n_imgs, 1, n_rows, n_cols};
    Tensor* t = tensor_create(4, shape, FLOAT32, CPU);

    uint32_t pixels = n_rows * n_cols;
    uint8_t* buffer = malloc(pixels);

    for (uint32_t i = 0; i < n_imgs; i++) {
        size_t rd = fread(buffer, 1, pixels, f);
        if (rd != pixels) {
            printf("Unexpected EOF while reading MNIST images\n");
            free(buffer);
            fclose(f);
            return NULL;
        }

        float* dst = (float*)t->data + i * pixels;
        for (uint32_t j = 0; j < pixels; j++) {
            dst[j] = buffer[j] / 255.0f;
        }
    }

    free(buffer);
    fclose(f);

    return t;
}

Tensor* load_mnist_labels(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { printf("Cannot open %s\n", path); return NULL; }

    uint32_t magic  = read_be_uint32(f);
    uint32_t n_lbls = read_be_uint32(f);

    if (magic != 2049) {
        printf("Invalid MNIST label magic: %u\n", magic);
        fclose(f);
        return NULL;
    }

    int64_t shape[1] = {n_lbls};
    Tensor* t = tensor_create(1, shape, INT64, CPU);

    uint8_t* labels = malloc(n_lbls);
    fread(labels, 1, n_lbls, f);

    int64_t* dst = (int64_t*)t->data;
    for (uint32_t i = 0; i < n_lbls; i++) dst[i] = labels[i];

    free(labels);
    fclose(f);
    return t;
}


static uint32_t read_be_uint32(FILE *f) {
    uint8_t bytes[4];
    fread(bytes, 1, 4, f);
    return (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
}

int mnist_get_item(void* self, int index, Tensor** x, Tensor** y) {
    MNISTDatasetData* d = (MNISTDatasetData*)self;

    *x = tensor_slice(d->images, index);  // your own slice implementation
    *y = tensor_slice(d->labels, index);

    return 0;
}

Dataset* create_mnist_dataset(const char* images_path, const char* labels_path) {
    MNISTDatasetData* d = malloc(sizeof(MNISTDatasetData));
    
    d->images = load_mnist_images(images_path);
    d->labels = load_mnist_labels(labels_path);

    Dataset* ds = malloc(sizeof(Dataset));
    ds->length = d->images->shape[0];
    ds->get_item = mnist_get_item;
    ds->data = d;
    return ds;
}