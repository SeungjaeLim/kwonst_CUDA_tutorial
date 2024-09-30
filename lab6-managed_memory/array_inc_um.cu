#include <cstdio>
#include <cstdlib>
// error checking macro
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

/**
 * @brief Allocate unified memory using cudaMallocManaged.
 * 
 * This template function allocates unified memory that is accessible by both
 * the host and device using cudaMallocManaged.
 * 
 * @param ptr       Reference to the pointer to allocate memory for.
 * @param num_bytes Number of bytes to allocate.
 */
template <typename T>
void alloc_bytes(T &ptr, size_t num_bytes){
 /*
  * TODO: Allocate memory using cudaMallocManaged.
  */ 
}

/**
 * @brief CUDA kernel to increment elements of an array.
 * 
 * This kernel increments each element of the input array by 1.
 * It uses a grid-stride loop to handle large arrays that are larger than the total number of threads.
 * 
 * @param array Pointer to the array in device memory.
 * @param n     Size of the array.
 */
__global__ void inc(int *array, size_t n){
  size_t idx = threadIdx.x+blockDim.x*blockIdx.x;
  /* 
   * TODO: Loop through the array using a grid-stride loop.
   * This ensures all elements are processed even if the array is larger than the number of threads.
   */
}

const size_t  ds = 32ULL*1024ULL*1024ULL;

int main(){

  int *h_array;
  alloc_bytes(h_array, ds*sizeof(h_array[0]));
  cudaCheckErrors("cudaMalloc Error");
  memset(h_array, 0, ds*sizeof(h_array[0]));
  cudaCheckErrors("cudaMemcpy H->D Error");
  /* 
   * TODO: Launch the CUDA kernel to increment each element of the array by 1.
   * The kernel is configured with 256 blocks and 256 threads per block.
   */
  inc<<<256, 256>>>(/*TODO*/, ds);
  /* 
   * TODO: Synchronize the device to ensure the kernel execution is complete.
   */

  cudaCheckErrors("kernel launch error");
  for (int i = 0; i < ds; i++) 
    if (h_array[i] != 1) {printf("mismatch at %d, was: %d, expected: %d\n", i, h_array[i], 1); return -1;}
  printf("success!\n"); 
  return 0;
}
