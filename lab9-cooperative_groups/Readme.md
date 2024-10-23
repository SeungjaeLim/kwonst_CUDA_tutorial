# Lab 9: Cooperative Groups Lab

This lab focuses on exploring CUDA cooperative groups to perform reductions and stream compaction in a more efficient way using thread-level and grid-level synchronization. The lab tasks include implementing reduction operations with cooperative groups and performing stream compaction with grid-wide synchronization, followed by profiling each implementation.

### How to Run

Use the provided `Makefile` to compile the implementations.

### Compilation

```
# Compile all 
make all

# Compile a specific task
make reduce # reduction task
make stream_compaction # stream compaction task
```

### Cleaning Up
```
make clean
```

### Brief Explanation of the Assignment

### 1. Reduction Using Cooperative Groups (`reduce.cu`)

This task involves implementing reduction operations across different levels of thread groups (full block, 32-thread tile, and 16-thread tile) using CUDA cooperative groups. Cooperative groups provide fine-grained control over thread synchronization, allowing for efficient reduction operations.

**Key Steps:**

1. Create a thread block group for full block-level reduction.

2. Create 32-thread and 16-thread tiles within the block for partial reduction.

### 2. Stream Compaction with Cooperative Grid-Wide Synchronization (`stream_compaction.cu`)

This task performs a stream compaction using cooperative groups to synchronize threads within the block and across the grid. The task demonstrates the use of cooperative grid-wide synchronization to achieve efficient stream compaction without multiple kernel launches.

**Key Steps:**

1. Use cooperative groups to create thread block and grid-wide groups.

2. Perform prefix sum (scan) over the predicate values to determine which elements to keep.

3. Synchronize across the entire grid to ensure prefix sum completion before moving data.

4. Move input data to output locations based on the prefix sum results.

**Stream Compaction Example:**

Consider an input array with both elements to keep and elements to remove, for example:

```
Input:  [3, 4, 3, 7, 0, 5, 0, 8, 0, 0, 0, 4]
Filter: [1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1]
```

Here, the filter array represents the predicate, where 1 indicates that the value should be kept, and 0 means it should be removed. The exclusive prefix sum of the filter array would look like:

```
Prefix Sum: [0, 1, 2, 3, 4, 4, 5, 5, 6, 6, 6, 6]
```

Using the prefix sum, we can determine the final positions of the elements to keep, resulting in the output array:
```
Output: [3, 4, 3, 7, 5, 8, 4]
```
The prefix sum helps in determining the exact indices to place the values that should be kept, allowing for efficient parallel compaction of the array.