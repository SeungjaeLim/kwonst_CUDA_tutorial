#include <cooperative_groups.h>
#include <stdio.h>
#include <cstdlib>  // for rand()
#include <ctime>   // for time()
using namespace cooperative_groups;

const int nTPB = 256;


/**
 * @brief Performs a reduction operation across the threads in the group.
 *
 * This function performs a reduction within a thread group to sum values across all
 * threads. It uses shared memory for storing partial results and synchronizes the threads
 * to ensure proper reduction.
 * 
 * @param g   The thread group (block, tile, etc.) used for synchronization.
 * @param x   Shared memory pointer for partial results.
 * @param val The value for each thread to contribute to the reduction.
 * 
 * @return The reduced value, if the calling thread is rank 0.
 */
__device__ int reduce(thread_group g, int *x, int val) { 
 /* 
  * TODO: Retrieve the current thread's rank within the group.
  * This rank (lane) is used to determine the position of the thread within the group
  * for performing reduction operations.
  */
  int lane = ;

  /* TODO: Use a loop to iteratively reduce values in a binary tree fashion */
  for (int i = ; ; ) {
    /* TODO: Store the current value in shared memory */
      
    /* TODO: Synchronize all threads within the group.*/

    /* TODO: Perform the reduction.*/

    /* TODO: Synchronize again to ensure all threads see the result.*/

  }

  if (g.thread_rank() == 0) {
    return val;
  }
  return 0;
}

/**
 * @brief CUDA kernel that performs multiple levels of reduction using different thread group sizes.
 *
 * This kernel reduces the input data in three stages:
 * - A full block reduction (g1)
 * - A 32-thread tile reduction (g2)
 * - A 16-thread tile reduction (g3)
 * It uses cooperative groups to partition threads and perform reductions with different synchronization scopes.
 * 
 * @param data        Input data to be reduced.
 * @param total_sum   Output array to store the reduced sums for each group.
 * @param g1_counter  Pointer to counter for the number of times g1 is executed.
 * @param g2_counter  Pointer to counter for the number of times g2 is executed.
 * @param g3_counter  Pointer to counter for the number of times g3 is executed.
 */
__global__ void my_reduce_kernel(int *data, int *total_sum, int *g1_counter, int *g2_counter, int *g3_counter) {
  __shared__ int sdata[nTPB];

  /* TODO: Create a thread block group (g1) for full block-level reduction */
  auto g1 = ;
  size_t gindex = g1.group_index().x * nTPB + g1.thread_index().x;

  /* TODO: Create a 32-thread tile within the block (g2) for partial reduction */
  auto g2 = ;

  /* TODO: Create a 16-thread tile within g2 for more fine-grained reduction (g3) */
  auto g3 = ;

  // g1 sum reduction
  int sdata_offset = (g1.thread_index().x / g1.size()) * g1.size();
  atomicAdd(&total_sum[0], reduce(g1, sdata + sdata_offset, data[gindex]));
  g1.sync();
  
  // g1 counter
  if (g1.thread_rank() == 0) atomicAdd(g1_counter, 1);

  // g2 sum reduction
  sdata_offset = (g1.thread_index().x / g2.size()) * g2.size();
  atomicAdd(&total_sum[1], reduce(g2, sdata + sdata_offset, data[gindex]));
  g2.sync();
  
  // g2 counter
  if (g2.thread_rank() == 0) atomicAdd(g2_counter, 1);

  // g3 sum reduction
  sdata_offset = (g1.thread_index().x / g3.size()) * g3.size();
  atomicAdd(&total_sum[2], reduce(g3, sdata + sdata_offset, data[gindex]));
  g3.sync();
  
  // g3 counter
  if (g3.thread_rank() == 0) atomicAdd(g3_counter, 1);
}

int main() {
  int *data, *total_sum, *g1_counter, *g2_counter, *g3_counter;
  cudaMallocManaged(&data, nTPB * sizeof(int));
  cudaMallocManaged(&total_sum, 3 * sizeof(int));
  cudaMallocManaged(&g1_counter, sizeof(int));
  cudaMallocManaged(&g2_counter, sizeof(int));
  cudaMallocManaged(&g3_counter, sizeof(int));

  // Initialize random data and counters
  std::srand(std::time(0));
  int host_sum = 0;
  for (int i = 0; i < nTPB; i++) {
    data[i] = std::rand() % 10;  // Random number between 0 and 9
    host_sum += data[i];         // Host sum for validation
  }
  
  // Initialize total_sum and counters
  for (int i = 0; i < 3; i++) total_sum[i] = 0;
  *g1_counter = 0;
  *g2_counter = 0;
  *g3_counter = 0;

  // Launch kernel
  my_reduce_kernel<<<1, nTPB>>>(data, total_sum, g1_counter, g2_counter, g3_counter);
  cudaError_t err = cudaDeviceSynchronize();

  // Output total sums and counters
  printf("Host sum: %d\n", host_sum);
  printf("Total sum 1 (g1): %d\n", total_sum[0]);
  printf("Total sum 2 (g2): %d\n", total_sum[1]);
  printf("Total sum 3 (g3): %d\n", total_sum[2]);

  // Output group counters
  printf("g1 executed %d times\n", *g1_counter);
  printf("g2 executed %d times\n", *g2_counter);
  printf("g3 executed %d times\n", *g3_counter);

  // Check if the results are correct
  bool pass = true;

  // Check if the sums are correct
  if (total_sum[0] != host_sum) {
    printf("Fail: g1 sum does not match host sum\n");
    pass = false;
  } 

  if (total_sum[1] != host_sum) {
    printf("Fail: g2 sum does not match host sum\n");
    pass = false;
  } 

  if (total_sum[2] != host_sum) {
    printf("Fail: g3 sum does not match host sum\n");
    pass = false;
  } 

  // Check if the group counters are correct
  if (*g1_counter != 1) {
    printf("Fail: g1 did not execute exactly 1 time\n");
    pass = false;
  } 

  if (*g2_counter != 8) {
    printf("Fail: g2 did not execute 8 times\n");
    pass = false;
  } 

  if (*g3_counter != 16) {
    printf("Fail: g3 did not execute 16 times\n");
    pass = false;
  } 

  if (pass) {
    printf("All tests passed!\n");
  } else {
    printf("Some tests failed!\n");
  }

  // Error handling
  if (err != cudaSuccess) 
    printf("cuda error: %s\n", cudaGetErrorString(err));

  // Free memory
  cudaFree(data);
  cudaFree(total_sum);
  cudaFree(g1_counter);
  cudaFree(g2_counter);
  cudaFree(g3_counter);

  return 0;
}
