# Lab 3: Grid-Stride Loops in CUDA
This lab demonstrates how to use a grid-stride loop to perform vector addition in CUDA. A grid-stride loop is a technique where each thread processes multiple elements in a data set, making better use of GPU resources when the number of threads is smaller than the total size of the input data.

## How to Run
To compile and run the programs in this lab, use the provided Makefile. The Makefile contains rules to build the CUDA program and provides a convenient way to clean up compiled binaries.

### Compilation
Use the following commands to compile the CUDA programs:

```
# Compile the vector addition program
make all

# Run the vector addition program
./vadd   
```
### Cleaning Up
To remove the generated binary files:

```
make clean
```
## Brief Explanation of the Assignment

### 1. Vector Addition with Grid-Stride Loop (`vadd.cu`)
This program performs element-wise addition of two vectors A and B, storing the result in vector C. To efficiently process large datasets, a grid-stride loop is used, allowing each thread to handle multiple elements.

**Grid-Stride Loop:**

In a typical CUDA kernel, each thread processes one element of a dataset. However, in cases where the total number of threads is smaller than the size of the dataset, many elements may remain unprocessed. To handle this, we use a grid-stride loop, where each thread can process multiple elements by stepping through the dataset in increments.

### Key Concepts

**Global Index**: 
    Each thread calculates a unique global index based on its block and thread indices. This index is used to determine which element in the vectors the thread will operate on.

**Grid-Stride**: 
    Each thread processes multiple elements by incrementing its index by the total number of threads in the grid. This ensures that all elements are processed, even if the number of threads is smaller than the size of the dataset.

### Benefits of Grid-Stride Loops:
**Improved Efficiency**: 
    The grid-stride loop allows threads to process multiple elements, improving the utilization of the GPU when there are fewer threads than elements in the vector.

**Simple and Flexible**: 
    This approach easily adapts to different grid and block configurations, ensuring that all data is processed efficiently.

### 2. Profiling Experiments
In this section, we will explore the impact of grid and block sizing (i.e., the number of blocks and threads per block) on the performance of our CUDA kernel. To do this, we will use Nsight Compute for profiling the kernel and collecting data on its duration and memory throughput. The goal is to analyze how different grid configurations affect the performance.

**Profiling Steps:**

We will perform three experiments with different configurations of blocks and threads, using the following command to run the profiler:

```
ncu --section SpeedOfLight --section MemoryWorkloadAnalysis ./vadd
```
This command will profile the kernel and display information about the kernel duration and memory throughput.


**Experiment A: 1 Block of 1 Thread**

In this experiment, we will use the smallest grid size possible: 1 block with 1 thread. This configuration is expected to yield poor performance due to low thread utilization and inefficient use of memory bandwidth.

**Experiment B: 1 Block of 1024 Threads**

Next, we will increase the number of threads per block to the maximum allowable limit (1024), while keeping the number of blocks at 1. This setup will utilize more threads but will still not make full use of the GPU’s SMs.

**Experiment C: 160 Blocks of 1024 Threads**

Finally, we will fully load the GPU by using 160 blocks, each containing 1024 threads. This configuration will maximize the occupancy of the GPU’s SMs and should result in the best performance in terms of both kernel duration and memory throughput.

### Grid Configuration and Profiling Results

| Experiment                        | Blocks | Threads per Block | Duration (useconds) | Memory Throughput (TB/s) |
|------------------------------------|--------|-------------------|--------------------|--------------------------|
| **A**        | 1      | 1                 | TODO              | TODO                    |
| **B**    | 1      | 1024              | TODO              | TODO                    |
| **C** | 160    | 1024              | TODO              | TODO                    |
