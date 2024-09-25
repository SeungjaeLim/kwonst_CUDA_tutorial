# Lab 4: Matrix Row and Column Sums in CUDA

This lab demonstrates how to compute row and column sums of a matrix in CUDA. The lab includes both CUDA kernel implementations for row-wise and column-wise sum computations. We will also introduce the use of Nsight Compute to profile the performance of these kernels.

## How to Run
To compile and run the programs in this lab, use the provided `Makefile`. The `Makefile` contains rules to build the CUDA program and provides a convenient way to clean up compiled binaries.

### Compilation
Use the following commands to compile the CUDA programs:

```
# Compile the matrix sum program
make all

# Run the matrix sum program
./matrix_sums
```
### Cleaning Up
To remove the generated binary files:

```
make clean
```

## Brief Explanation of the Assignment
### 1. Row and Column Sums (`matrix_sums.cu`)

This program performs the following operations:

**Row Sums**: 
    Each thread is assigned to a row in the matrix and calculates the sum of all elements in that row. The result is stored in a vector representing the sums of each row.

**Column Sums**: 
    Each thread is assigned to a column in the matrix and calculates the sum of all elements in that column. The result is stored in a vector representing the sums of each column.

### 2. Profiling with Nsight Compute
In this section, we will use Nsight Compute to analyze the performance of our row and column sum kernels. We will first measure the kernel execution times and then gather additional metrics related to global memory load efficiency.

### Kernel Timing

Use Nsight Compute to profile and measure the kernel execution time:

```
ncu ./matrix_sums
```

Analyze the output to find the kernel durations for the row sums and column sums. Are the kernel durations similar or different? Consider whether these differences are expected based on the structure of the kernels.

### Memory Load Efficiency:

Next, gather metrics related to global memory load efficiency using the following command:

```
ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum ./matrix_sums
```

This will provide two key metrics:

`l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum`: 
    The number of sectors requested for global loads (transactions).

`l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum`: 
    The number of global memory load requests (requests).

The transactions per request can be computed by dividing these two metrics, providing insight into the global memory efficiency of the kernels.

### Profiling Results:
| Kernel        | Kernel Duration (ms) | GMEM Load Request | GMEM Load Sector (Transactions) | Transactions (Sector) per Request |
|---------------|----------------------|-------------------|------------------|-----------------------------------|
| Row Sums      | TODO                | TODO             | TODO            | TODO                             |
| Column Sums   | TODO               | TODO             | TODO            | TODO                             |


### Key Questions:
- Are the kernel durations for row sums and column sums different? Why or why not?
- How does the global memory load efficiency of the row sums kernel compare to the column sums kernel?
- Can we improve the memory access pattern or kernel efficiency for better performance?