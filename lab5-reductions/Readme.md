# Lab 5: CUDA Reductions

This lab demonstrates various CUDA reduction techniques and matrix operations, focusing on profiling kernel execution times and memory efficiency. We will compare several reduction methods, implement a maximum-finding reduction, and improve the performance of matrix row and column sums.

## How to Run
To compile and run the programs in this lab, follow the instructions below. The Makefile includes rules to build the CUDA programs, and you can clean up the generated binaries after running.

### Compilation
Run the following commands to compile the programs:

```
# Compile all CUDA programs
make all

./reduct

./maxred

./msum
```
Cleaning Up
To remove compiled binary files:

```
make clean
```
## Brief Explanation of the Assignment
### 1. Sum Reduction (`reduction.cu`)
This program implements different types of sum reduction kernels:

- **Atomic Reduction**: This kernel performs reduction by accumulating partial sums using atomic operations, which ensures race condition-free updates but can result in high contention and lower performance.

- **Reduction with Atomic Finish**: This kernel uses parallel reduction within each block and atomic operations only at the final stage to update the global sum.

- **Warp-Shuffle Reduction**: This kernel leverages warp-shuffle instructions for efficient intra-warp communication, followed by atomic operations to finalize the reduction.
Profiling and Timing

### Profiling with Nsight Compute

For this experiment, you'll be running the reduction on arrays of different sizes (163840, 8M, and 32M) and recording the execution time for each kernel.

```
ncu ./reduct
```
### Profiling Results:

| Kernel                  | Array Size | Kernel Duration (ms) |
|-------------------------|------------|----------------------|
| Atomic Reduction         | 163840     | TODO                 |
| Atomic Reduction         | 8M         | TODO                 |
| Atomic Reduction         | 32M        | TODO                 |
| Parallel Reduction       | 163840     | TODO                 |
| Parallel Reduction       | 8M         | TODO                 |
| Parallel Reduction       | 32M        | TODO                 |
| Warp-Shuffle Reduction   | 163840     | TODO                 |
| Warp-Shuffle Reduction   | 8M         | TODO                 |
| Warp-Shuffle Reduction   | 32M        | TODO                 |


Key Questions:

- How do the kernel execution times change when N increases from 163840 to 8M to 32M?
- Why might there be little difference between the parallel reduction with atomics and the warp-shuffle reduction with atomics? 


### 2. Max Reduction (`max_reduction.cu`)
This task requires modifying the sum reduction kernel to implement a max-finding reduction. The kernel finds the maximum value in the dataset rather than the sum. A two-stage reduction process is employed, where each block first reduces its assigned elements, and then a second reduction finds the global maximum.

### 3. Matrix Row and Column Sums (`matrix_sums.cu`)
In this task, we revisit matrix sums from Lab 4 and improve the performance of the row sums kernel by utilizing a parallel reduction per row. This kernel now matches the efficiency of the column sum kernel by addressing memory access inefficiencies.

- **Lab 4 Row Sums**: 
    
    Basic row sum kernel using a grid-stride loop.
- **Lab 5 Row Sums**: 

    Optimized row sum kernel using parallel reduction within each block.

The strategy for optimizing the row sum kernel involves assigning one block per row, allowing each block to handle the reduction of a single row. Instead of using a separate kernel call for each row, a for-loop can be used to loop over all rows, or we can assign a warp or thread block to each row, performing the reduction in one kernel call. The kernel should be adapted from the parallel reduction method (without atomic operations) seen in the earlier exercises. Each block will perform a block-striding loop, which functions similarly to a grid-stride loop, traversing the row in strides to sum up the elements efficiently. Careful indexing will be needed to ensure each block correctly reduces its assigned row.

### Profiling with Nsight Compute

```
ncu ./msum
```

### Profiling Results:

| Kernel              | Matrix Size | Kernel Duration (ms) |
|---------------------|-------------|----------------------|
| Lab 4 Row Sums      | TODO        | TODO                 |
| Lab 5 Row Sums      | TODO        | TODO                 |
| Column Sums         | TODO        | TODO                 |

