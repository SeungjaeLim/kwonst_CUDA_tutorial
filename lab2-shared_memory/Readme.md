# Lab 2: Shared Memory in CUDA

This lab focuses on using shared memory in CUDA to improve performance. The lab includes two main programs: matrix multiplication using shared memory and a 1D stencil computation. The goal is to understand how shared memory can reduce global memory accesses and speed up CUDA kernels.

## How to Run
To compile and run the programs in this lab, use the provided `Makefile`. The `Makefile` contains rules to build each individual program and provides a convenient way to clean up compiled binaries.

### Compilation
Use the following commands to compile the CUDA programs:

```
# Compile all programs
make all

# Compile 1D stencil computation
make stc
./stc

# Compile matrix multiplication with shared memory
make mmul
./mmul
```

Cleaning up
To remove the generated binary files:

```
make clean
```

## Brief Explanation of Each Assignment



### 1. 1D Stencil Computation with Shared Memory (`stencil_1d.cu`)
This program applies a 1D stencil operation to an input array using shared memory to improve performance.
Each thread computes the sum of its neighboring elements within a defined radius and writes the result to an output array.
Shared memory is used to store a block of input data plus halo elements (neighboring elements) for faster access.
You will implement TODOs to allocate shared memory, load data, synchronize threads, apply the stencil operation, and store the result.

### 2. Matrix Multiplication with Shared Memory (`matrix_mul_shared.cu`)
This program multiplies two square matrices A and B, storing the result in matrix C.
Shared memory is used to load sub-blocks of matrices A and B into shared memory, reducing global memory accesses.
You will implement TODOs to load data into shared memory, synchronize threads, perform the dot product calculation, and store the result in matrix C.
Each thread computes one element of the output matrix using tiles of data loaded into shared memory.