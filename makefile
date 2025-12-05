# Compiler
CC = gcc
NVCC = nvcc
CFLAGS = -std=c11 -Wall -Wextra -pedantic -I./include
NVCCFLAGS = -std=c++14 -Xcompiler "-fPIC" -I./include

# CPU library sources 
LIB_SRC = src/tensor.c src/tensor_ops.c src/tensor_ops_cpu.c src/nn_layers.c src/nn.c src/optimizers.c src/loss_fns.c src/dataset.c src/metrics.c
LIB_OBJ = $(LIB_SRC:%.c=build/%.o)

# CUDA sources
CUDA_SRC = cuda/tensor_ops.cu cuda/tensor.cu cuda/nn_layers.cu cuda/loss_fns.cu cuda/optimizers.cu cuda/metrics.cu
CUDA_OBJ = $(CUDA_SRC:%.cu=build/%.o)

# Examples and tests
EXAMPLES = tests/demo.c examples/mnist/mnist_mlp.c
PROGRAMS = $(EXAMPLES:%.c=build/%)

# Default target
all: $(LIB_OBJ) $(CUDA_OBJ) $(PROGRAMS)

# Compile CPU library objects
build/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

# Compile CUDA objects
build/%.o: %.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Build each program (link everything with NVCC in correct order)
build/%: %.c $(LIB_OBJ) $(CUDA_OBJ)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $(CUDA_OBJ) $(LIB_OBJ) $< -o $@ -lcudart

# Clean
clean:
	rm -rf build