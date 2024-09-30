# Lab 6: Managed Memory

This lab explores two main concepts in CUDA programming: linked list data structures and array operations, specifically focusing on manual data copy, Unified Memory (UM), and the benefits of prefetching. You will be implementing and profiling programs using CUDA's Unified Memory (UM) feature to simplify data management between the CPU and GPU.

### How to Run
The `Makefile` provided compiles all the necessary programs. You can run and clean the programs using the commands below:

### Compilation
```
# Compile all CUDA programs
make all

# Running the executables
./linked_list_manual
./linked_list_um
./arr_inc_manual
./arr_inc_um
./arr_inc_prefetch
```
### Cleaning Up
```
make clean
```

## Brief Explanation of the Assignment
### 1. Linked List Porting to GPU

This task involves porting a linked list structure to the GPU in two ways: manually copying memory and using Unified Memory.

**A. Manual Copy (`linked_list_manual.cu`)**

The program manually copies the linked list from host memory to device memory. This process requires allocating memory for each node on both the host and the device and copying the data.

Allocate memory for each node on both host and device using `cudaMalloc`.
Manually copy each node's key and next pointer to the device.
Test the linked list on the GPU by printing an element using a kernel.

**B. Unified Memory (`linked_list_um.cu`)**

The Unified Memory version simplifies memory management by using `cudaMallocManaged`. Here, the linked list resides in memory accessible to both the CPU and GPU, removing the need for explicit memory transfers.

Use `cudaMallocManaged` to allocate memory for the entire linked list.
The program automatically handles data movement between host and device.
Test the linked list on the GPU by printing an element using a kernel.

### 2. Array Increment
This task involves incrementing elements of an array on the GPU, first manually handling memory transfers and then using Unified Memory with and without prefetching.

**A. Manual Copy (`array_inc_manual.cu`)**

The program increments each element of the array, manually copying the array from host to device before the kernel launch and back after completion.

Allocate memory on both host and device.
Copy the array from host to device using `cudaMemcpy`.
Launch a CUDA kernel to increment array elements on the device.
Copy the incremented array back to host memory for verification.

**B. Unified Memory (`array_inc_um.cu`)**

Using Unified Memory simplifies memory management by eliminating manual `cudaMemcpy` calls. The array is allocated in Unified Memory and is automatically available to both host and device.

Use `cudaMallocManaged` to allocate the array.
No explicit `cudaMemcpy` required, as the system handles memory transfers automatically.

**C. Prefetching with Unified Memory (`array_inc_prefetch.cu`)**

Prefetching is used to explicitly migrate the memory pages to the GPU before kernel execution to improve performance.

Use `cudaMemPrefetchAsync` to move the memory to the GPU before kernel execution.
Use `cudaMemPrefetchAsync` again to migrate the memory back to the CPU after kernel execution.

**D. Profiling**

To collect performance data, use the following command:

```
nsys profile --force-overwrite=true --cuda-um-gpu-page-faults=true --cuda-um-cpu-page-faults=true --stats=true --show-output=true ./program_name
```

**D1. Duration Profiling**

Record the kernel execution time for the following:

| Program                  | Kernel Duration (ms) |
|--------------------------|----------------------|
| A. Manual Copy              | TODO                 |
| B. Unified Memory           | TODO                 |
| C. Unified Memory + Prefetch| TODO                 |


**D2. Migration and Page Fault Profiling**

For the Unified Memory version (B), check for migration and page faults. The command output should include the following information:

| Program          | HtoD Migration (MB) | DtoH Migration (MB) | CPU Page Faults | GPU Page Faults |
|------------------|---------------------|---------------------|-----------------|-----------------|
| Unified Memory   | TODO                | TODO                | TODO            | TODO            |

**Key Metrics:**

- Total HtoD (Host to Device) Migration Size (MB): 

    The total amount of data migrated from host to device during execution.
- Total DtoH (Device to Host) Migration Size (MB): 

    The total amount of data migrated from device to host during execution.
- Total CPU Page Faults: 

    The number of page faults that occurred on the CPU.
- Total GPU Page Faults: 

    The number of page faults that occurred on the GPU.