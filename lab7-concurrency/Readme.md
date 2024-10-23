# Lab 7: Concurrency

This lab focuses on concurrency in CUDA programming. You will implement three versions of a Gaussian PDF computation using CUDA: a default single-GPU implementation, an implementation using CUDA streams, and an implementation utilizing multiple GPUs. The goal of this lab is to explore the concurrency mechanisms available in CUDA, compare their performance, and use Nsight Systems for profiling.

### How to Run
The `Makefile` provided compiles all the necessary programs. You can run and clean the programs using the commands below:

### Compilation
```
# Compile all CUDA programs
make all

# Running the executables
./gpdf           # Run the default single-GPU implementation
./overlap        # Run the GPU implementation using streams
./multi          # Run the multi-GPU implementation
```

### Cleaning Up
```
make clean
```
## Brief Explanation of the Assignment
### 1. Gaussian PDF Computation (`gpdf.cu`)
This task involves the computation of the Gaussian probability density function (PDF) on the GPU. The goal is to implement the function without any concurrency optimization.

**Key Steps:**

1. Allocate memory on the host and device.
2. Copy input data from the host to the device.
3. Execute the Gaussian PDF kernel on the GPU.
4. Copy the result back to the host and check for correctness.

### 2. Gaussian PDF with Streams (`gpdf_overlap.cu`)
In this task, we use CUDA streams to parallelize memory transfers and kernel execution, improving performance through overlap of computation and data transfers.

**Key Steps:**
1. Allocate pinned host memory and device memory.
2. Use multiple CUDA streams to launch memory transfers and kernels concurrently.
3. Perform a depth-first launch of the kernel across chunks of data.

### 3. Multi-GPU Gaussian PDF (`gpdf_multi.cu`)

This task involves using multiple GPUs to distribute the workload and compute the Gaussian PDF in parallel across multiple devices.

**Key Steps:**
1. Allocate memory on each GPU.
2. Transfer data to each GPU.
3. Launch the Gaussian PDF kernel on each GPU in parallel.
4. Synchronize the results across all GPUs.

### 4. Profiling with Nsight Systems

You will use NVIDIA's Nsight Systems tool to profile each of the programs. Nsight Systems helps identify performance bottlenecks, data movement issues, and concurrency opportunities in CUDA programs.

To profile the programs, use the following command:

```
nsys profile -o <destination_dir>/<filename>.qdrep <filename>
```

This will generate a profiling report that can be opened with Nsight Systems. For more information on how to install Nsight Systems, visit the official Nsight Systems download page. [Download Nsight Systems](https://developer.nvidia.com/nsight-systems)


**Example:** 
```
nsys profile -o ./gpdf.qdrep ./gpdf
```

**Profiling Results**
| Program                    | Elapsed Time (ms)    |
|----------------------------|----------------------|
| Default Single-GPU (`gpdf`) | TODO                 |
| Streams (`gpdf_overlap`)    | TODO                 |
| Multi-GPU (`gpdf_multi`)    | TODO                 |
