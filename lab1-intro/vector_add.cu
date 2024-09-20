#include <stdio.h>
#include <stdlib.h>

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

const int DSIZE = 4096;
const int block_size = 256;  // CUDA maximum is 1024

/**
 * @brief CUDA kernel for vector addition: C = A + B.
 * 
 * This kernel performs element-wise addition of two input vectors A and B, and stores 
 * the result in the output vector C. The operation is parallelized across multiple threads, 
 * with each thread handling one element of the vector.
 * 
 * @param A   Pointer to the input vector A in device memory (size: ds elements).
 * @param B   Pointer to the input vector B in device memory (size: ds elements).
 * @param C   Pointer to the output vector C in device memory (size: ds elements).
 * @param ds  The size of the vectors (number of elements in A, B, and C).
 * 
 * Details:
 * - Each thread computes the sum of corresponding elements from A and B and writes 
 *   the result to the same index in C.
 * - The global index of each thread is calculated using the block and thread indices.
 * - The kernel checks if the global index is within bounds before performing the addition.
 *
 * Grid and Block Dimensions:
 * - Block size is defined by `block_size`, typically set based on hardware limits (e.g., 256).
 * - Grid size is determined by dividing the total size (ds) by block_size, ensuring all elements are processed.
 */
__global__ void vadd(const float *A, const float *B, float *C, int ds){

  /* TODO: Calculate the global thread index.
   * This combines the block and thread indices to calculate the global index in the arrays. */
  int idx = /*TODO*/

  /* TODO: Check if the index is within bounds, then perform the addition.
   * Each thread adds one element from A and B, and stores the result in C. */
  if (idx < ds)
    /*TODO*/
}

// CPU implementation of vector addition
void cpu_vadd(const float *A, const float *B, float *C, int ds) {
    for (int i = 0; i < ds; i++) {
        C[i] = A[i] + B[i];
    }
}

int main(){

  float *h_A, *h_B, *h_C, *h_C_CPU, *d_A, *d_B, *d_C;
  h_A = new float[DSIZE];  // allocate space for vectors in host memory
  h_B = new float[DSIZE];
  h_C = new float[DSIZE];
  h_C_CPU = new float[DSIZE];  // allocate space for CPU comparison result
  for (int i = 0; i < DSIZE; i++){  // initialize vectors in host memory
    h_A[i] = rand()/(float)RAND_MAX;
    h_B[i] = rand()/(float)RAND_MAX;
    h_C[i] = 0;
    h_C_CPU[i] = 0;  // initialize CPU result vector
  }

  /* TODO: Allocate memory on the device for vectors A, B, and C.
   * These will hold the input and output data for the GPU computation. */
  /*TODO*/
  cudaCheckErrors("cudaMalloc failure"); // error checking

  /* TODO: Copy the initialized vectors from host to device memory.
   * This transfers input vectors A and B from CPU memory to GPU memory. */
  /*TODO*/
  cudaCheckErrors("cudaMemcpy H2D failure");

  /* TODO: Launch the vector addition kernel.
   * The grid size is calculated by dividing DSIZE by the block size to ensure all elements are processed. */
  /*TODO*/
  cudaCheckErrors("kernel launch failure");

  /* TODO: Copy the result vector C from device to host memory.
   * This transfers the result of the vector addition back to CPU memory for validation. */
  /*TODO*/
  cudaCheckErrors("kernel execution failure or cudaMemcpy H2D failure");

  // Perform CPU vector addition for comparison
  cpu_vadd(h_A, h_B, h_C_CPU, DSIZE);

  // Compare the results of GPU and CPU calculations
  bool success = true;
  for (int i = 0; i < DSIZE; i++) {
      if (abs(h_C[i] - h_C_CPU[i]) > 1e-5) {  // Allow a small floating-point tolerance
          printf("Mismatch at index %d, GPU result: %f, CPU result: %f\n", i, h_C[i], h_C_CPU[i]);
          success = false;
          break;
      }
  }

  if (success) {
      printf("Success!\n");
  } else {
      printf("Failure!\n");
  }

  // Cleanup
  delete[] h_A;
  delete[] h_B;
  delete[] h_C;
  delete[] h_C_CPU;
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  return 0;
}
