#include <stdio.h>

/**
 * @brief A simple CUDA kernel to print block and thread indices.
 */
__global__ void hello(){
  /* TODO: Use `blockIdx.x` to print the block index.
   * `blockIdx.x` gives the index of the block within the grid. 
   *  Use `threadIdx.x` to print the thread index within the block.
   * `threadIdx.x` provides the index of the thread within the block. */
  printf("Hello from block: %u, thread: %u\n", /*TODO*/, /*TODO*/);
}

int main(){
  /* TODO: Specify the grid and block configuration.
   * Here, we are launching the kernel with 2 blocks, each containing 2 threads.
   * The configuration `<<<2, 2>>>` represents a grid of 2 blocks, and each block has 2 threads. */
  hello<<</*TODO*/>>>();
  cudaDeviceSynchronize();
}

