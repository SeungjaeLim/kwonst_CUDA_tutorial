# Lab 1: Introduction to CUDA Programming

This lab introduces basic CUDA programming concepts and explores three different programs: a simple "hello world" kernel, vector addition, and matrix multiplication. The objective is to get familiar with launching CUDA kernels, writing CUDA code, and understanding parallel execution.

## How to Run
To compile and run the programs in this lab, use the provided `Makefile`. The `Makefile` contains the rules to build each individual program and provides a convenient way to clean up compiled binaries.

### Compilation
Use the following commands to compile the CUDA programs:
```
# Compile all programs
make all

# Compile hello world kernel
make hello

# Compile vector addition
make vadd

# Compile matrix multiplication
make mmul
```
### Cleaning up
To remove the generated binary files:

```
make clean
```

## Brief Explanation of Each Assignment
### 1. Hello World Kernel (`hello.cu`)
This is a basic CUDA program that launches a kernel called hello, which prints the block and thread indices for each thread in the grid.
You will implement TODOs to print the block and thread indices (`blockIdx.x`, `threadIdx.x`) and set the grid configuration (`<<<blocks, threads>>>`).
The objective is to understand the structure of a CUDA kernel and how thread blocks and grids are organized.

Expected Output:
```
Hello from block: 0, thread: 0
Hello from block: 0, thread: 1
Hello from block: 1, thread: 0
Hello from block: 1, thread: 1
```
(the ordering of the above lines may vary; ordering differences do not indicate an incorrect result)

### 2. Vector Addition (`vector_add.cu`)
This program performs element-wise addition of two vectors, A and B, to produce the result in vector C.
The CUDA kernel computes each element in parallel by calculating the global thread index and performing the addition if within bounds.
You will implement TODOs to calculate the thread's global index and write the result to the output vector.

### 3. Matrix Multiplication (`matrix_mul.cu`)
This program multiplies two square matrices A and B, storing the result in matrix C.
The CUDA kernel computes the dot product of each row of A with each column of B, parallelizing the computation across threads.
You will implement TODOs to perform the dot product and store the result in matrix C.
