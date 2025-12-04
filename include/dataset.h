#ifndef DATASET_H
#define DATASET_H

#include <stdint.h>
#include "../include/tensor.h"

typedef struct {
    int64_t length;          // number of samples
    int (*get_item)(void* self, int index, Tensor** x, Tensor** y);
    void* data;              // pointer to custom dataset struct
} Dataset;

#endif

#ifndef MNIST_DS_H
#define MNIST_DS_H

#include <stdint.h>
#include "../include/tensor.h"

typedef struct {
    Tensor* images;
    Tensor* labels;
} MNISTDatasetData;


#endif


#ifndef MNIST_DL_H
#define MNIST_DL_H

#include <stdint.h>
#include "../include/tensor.h"
#include <stdio.h>
#include <time.h> 

typedef struct {
    Dataset* dataset;
    int batch_size;
    int shuffle;
    int index;
    int* order;
} DataLoader;

DataLoader* dataloader_create(Dataset* ds, int batch_size, int shuffle);
int dataloader_next(DataLoader* loader, Tensor** X, Tensor** Y);
void dataloader_reset(DataLoader* loader);

// mnist


int mnist_get_item(void* self, int index, Tensor** x, Tensor** y);
Dataset* create_mnist_dataset(const char* images_path, const char* labels_path);
DataLoader* dataloader_create(Dataset* ds, int batch_size, int shuffle);
int dataloader_next(DataLoader* dl, Tensor** X, Tensor** Y);

void shuffle_array(int* arr, int n);
Tensor* load_mnist_images(const char* path);
Tensor* load_mnist_labels(const char* path);
static uint32_t read_be_uint32(FILE *f); 
int mnist_get_item(void* self, int index, Tensor** x, Tensor** y);


#endif

