# Lab 8: Optimization Lab

This lab focuses on optimizing a CUDA matrix transpose operation through a series of progressively more optimized implementations. The goal is to understand the impact of different optimizations, including the use of shared memory and the elimination of bank conflicts, and how these affect performance. The lab tasks include implementing naive matrix transpose, shared memory transpose, and shared memory transpose with bank conflict mitigation, followed by profiling each implementation.

### How to Run

Use the provided `Makefile` to compile the different versions of the matrix transpose implementation.

### Compilation

```
# Compile all versions of transpose
make all

# Compile a specific version of transpose
make naive # task1
make sm # task2
make bc # task3
```

### Cleaning Up
```
make clean
```

## Brief Explanation of the Assignment

### 1. Transpose Naive (`transpose_naive.cu`)

This task involves a straightforward implementation of matrix transpose using global memory without any optimization.

**Key Steps:**

1. Calculate row and column index for each thread.

2. Ensure the thread is within the matrix bounds.

3. Transpose the matrix element by switching row and column indices.


### 2. Transpose Shared Memory (`transpose_sm.cu`)

This version uses shared memory to optimize memory access patterns. Shared memory improves memory coalescing by allowing both coalesced loads and coalesced stores to global memory.

**Key Steps:**

1. Declare a shared memory array for a tile of the matrix.
2. Load the matrix tile from global memory into shared memory.
3. Synchronize threads to ensure all data is loaded.
4. Write the transposed tile back to global memory.

### 3. Transpose Without Bank Conflict (`transpose_bc.cu`)

This version mitigates shared memory bank conflicts by modifying the shared memory indexing. Adding an extra column to the shared memory array avoids conflicts during access.

**Key Steps:**

1. Modify shared memory to avoid bank conflicts by adding an additional column.

### 4. Profiling

Profile each implementation using NVIDIA Nsight Compute to gather performance metrics and identify bottlenecks.

To profile, use the following command:

```
./profile.sh <filename>
```

This will generate performance metrics for the given implementation. Complete the profiling of each version using the `./profile.sh` script and fill in the TODO sections in the table above. Analyze the profiling results to determine the effectiveness of each optimization, particularly focusing on memory access patterns and bank conflicts.

### Profiling Results


 **Metric Name**                                                 | **Description**                                                | **Metric Unit** | **Task 1** | **Task 2** | **Task 3** |
| -------------------------------------------------------------------------------- | -------------------------------------------------------------- | --------------- | ---------- | ---------- | ---------- |
| l1tex\_\_average\_t\_sectors\_per\_request\_pipe\_lsu\_mem\_global\_op\_ld.ratio | Average sectors per load request from global memory            | sector/request  | TODO       | TODO       | TODO       |
| l1tex\_\_average\_t\_sectors\_per\_request\_pipe\_lsu\_mem\_global\_op\_st.ratio | Average sectors per store request to global memory             | sector/request  | TODO       | TODO       | TODO       |
| l1tex\_\_data\_bank\_conflicts\_pipe\_lsu\_mem\_shared\_op\_ld.sum               | Number of shared memory bank conflicts during load operations  | -               | TODO       | TODO       | TODO       |
| l1tex\_\_data\_bank\_conflicts\_pipe\_lsu\_mem\_shared\_op\_st.sum               | Number of shared memory bank conflicts during store operations | -               | TODO       | TODO       | TODO       |
| l1tex\_\_data\_pipe\_lsu\_wavefronts\_mem\_shared\_op\_ld.sum                    | Number of load wavefronts in shared memory                     | -               | TODO       | TODO       | TODO       |
| l1tex\_\_data\_pipe\_lsu\_wavefronts\_mem\_shared\_op\_st.sum                    | Number of store wavefronts in shared memory                    | -               | TODO       | TODO       | TODO       |
| l1tex\_\_t\_bytes\_pipe\_lsu\_mem\_global\_op\_ld.sum.per\_second                | Global memory load bandwidth (bytes per second)                | Gbyte/s         | TODO       | TODO       | TODO       |
| l1tex\_\_t\_bytes\_pipe\_lsu\_mem\_global\_op\_st.sum.per\_second                | Global memory store bandwidth (bytes per second)               | Gbyte/s         | TODO       | TODO       | TODO       |
| l1tex\_\_t\_requests\_pipe\_lsu\_mem\_global\_op\_ld.sum                         | Total number of global memory load requests                    | request         | TODO       | TODO       | TODO       |
| l1tex\_\_t\_requests\_pipe\_lsu\_mem\_global\_op\_st.sum                         | Total number of global memory store requests                   | request         | TODO       | TODO       | TODO       |
| l1tex\_\_t\_sectors\_pipe\_lsu\_mem\_global\_op\_ld.sum                          | Total number of sectors accessed for global memory loads       | sector          | TODO       | TODO       | TODO       |
| l1tex\_\_t\_sectors\_pipe\_lsu\_mem\_global\_op\_st.sum                          | Total number of sectors accessed for global memory stores      | sector          | TODO       | TODO       | TODO       |
| smsp\_\_cycles\_active.avg.pct\_of\_peak\_sustained\_elapsed                     | Percentage of peak performance sustained by active cycles      | %               | TODO       | TODO       | TODO       |
| smsp\_\_sass\_average\_data\_bytes\_per\_sector\_mem\_global\_op\_ld.pct         | Percentage of maximum data bytes per sector for global loads   | %               | TODO       | TODO       | TODO       |
| smsp\_\_sass\_average\_data\_bytes\_per\_sector\_mem\_global\_op\_st.pct         | Percentage of maximum data bytes per sector for global stores  | %               | TODO       | TODO       | TODO       |

### Metric Descriptions

- `l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio`: Average number of sectors accessed per load request. Higher values may indicate inefficient memory access.

- `l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio`: Average number of sectors accessed per store request.

- `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum`: Total number of bank conflicts for shared memory load operations.

- `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum`: Total number of bank conflicts for shared memory store operations.

- `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum`: Number of load operations performed on shared memory.

- `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum`: Number of store operations performed on shared memory.

- `l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second`: Effective bandwidth for global memory loads.

- `l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum.per_second`: Effective bandwidth for global memory stores.

- `l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum`: Total number of global memory load requests made.

- `l1tex__t_requests_pipe_lsu_mem_global_op_st.sum`: Total number of global memory store requests made.

- `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum`: Number of memory sectors accessed during load operations.

- `l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum`: Number of memory sectors accessed during store operations.

- `smsp__cycles_active.avg.pct_of_peak_sustained_elapsed`: Percentage of cycles that achieve peak sustained performance.

- `smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct`: Efficiency of memory load operations, measured as a percentage of the theoretical peak.

- `smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct`: Efficiency of memory store operations.