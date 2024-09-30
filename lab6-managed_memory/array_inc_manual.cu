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
 * @brief Allocate memory using malloc.
 * 
 * This template function allocates memory on the host using malloc.
 * It takes a reference to a pointer and allocates the specified number of bytes.
 * 
 * @param ptr       Reference to the pointer to allocate memory for.
 * @param num_bytes Number of bytes to allocate.
 */
template <typename T>
void alloc_bytes(T &ptr, size_t num_bytes){

  ptr = (T)malloc(num_bytes);
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
   * This loop ensures that all elements are processed, even if the array is larger than the number of threads.
   */
}

const size_t  ds = 32ULL*1024ULL*1024ULL;

int main(){

  int *h_array, *d_array;
  alloc_bytes(h_array, ds*sizeof(h_array[0]));
  /* 
   * TODO: Allocate memory for the device array using cudaMalloc.
   * This allocates memory on the GPU for the array.
   */

  cudaCheckErrors("cudaMalloc Error");
  memset(h_array, 0, ds*sizeof(h_array[0]));
  /* 
   * TODO: Copy the initialized host array to the device array using cudaMemcpy.
   * This transfers data from host to device before launching the kernel.
   */

  cudaCheckErrors("cudaMemcpy H->D Error");
  inc<<<256, 256>>>(d_array, ds);
  cudaCheckErrors("kernel launch error");
  /* 
   * TODO: Copy the modified array from the device back to the host using cudaMemcpy.
   * This transfers the incremented array from device memory to host memory for verification.
   */

  cudaCheckErrors("kernel execution or cudaMemcpy D->H Error");
  for (int i = 0; i < ds; i++) 
    if (h_array[i] != 1) {printf("mismatch at %d, was: %d, expected: %d\n", i, h_array[i], 1); return -1;}
  printf("success!\n"); 
  return 0;
}
