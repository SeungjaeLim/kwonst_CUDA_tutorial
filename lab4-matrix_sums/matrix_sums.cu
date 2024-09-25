#include <stdio.h>
#include <cstdlib>  // for rand()
#include <ctime>    // for seeding rand()
#include <cmath>    // for abs

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

const size_t DSIZE = 16384;      // matrix side dimension
const int block_size = 256;  // CUDA maximum is 1024

/**
 * @brief CUDA kernel for calculating row sums of a matrix.
 * 
 * Each thread calculates the sum of one row in the matrix and stores the result
 * in the corresponding index of the output vector `sums`.
 * 
 * @param A     Pointer to the input matrix (1D array representing a 2D matrix) in device memory.
 * @param sums  Pointer to the output vector (sums of each row) in device memory.
 * @param ds    The dimension size of the matrix (number of rows and columns).
 *
 * Details:
 * - Each thread is assigned to a single row of the matrix.
 * - The thread iterates through all columns of its assigned row, summing up the elements.
 */
__global__ void row_sums(const float *A, float *sums, size_t ds){

  /* TODO: Calculate the global thread index to map threads to matrix rows. */
  int idx = /*TODO*/; 

  if (idx < ds) {
    float sum = 0.0f;
    /* TODO: Iterate through the columns of the matrix, accumulating the sum for the row. */
    /*TODO*/
  }
}

/**
 * @brief CUDA kernel for calculating column sums of a matrix.
 * 
 * Each thread calculates the sum of one column in the matrix and stores the result
 * in the corresponding index of the output vector `sums`.
 * 
 * @param A     Pointer to the input matrix (1D array representing a 2D matrix) in device memory.
 * @param sums  Pointer to the output vector (sums of each column) in device memory.
 * @param ds    The dimension size of the matrix (number of rows and columns).
 *
 * Details:
 * - Each thread is assigned to a single column of the matrix.
 * - The thread iterates through all rows of its assigned column, summing up the elements.
 */
__global__ void column_sums(const float *A, float *sums, size_t ds){

  /* TODO: Calculate the global thread index to map threads to matrix columns. */
  int idx = /*TODO*/; 

  if (idx < ds) {
    float sum = 0.0f;
    /* TODO: Iterate through the rows of the matrix, accumulating the sum for the column. */
    /*TODO*/
  }
}

// CPU implementation for row sums
void cpu_row_sums(const float *A, float *sums, size_t ds) {
    for (size_t row = 0; row < ds; row++) {
        float sum = 0.0f;
        for (size_t col = 0; col < ds; col++) {
            sum += A[row * ds + col];  // Accumulate sum across columns in a row
        }
        sums[row] = sum;
    }
}

// CPU implementation for column sums
void cpu_column_sums(const float *A, float *sums, size_t ds) {
    for (size_t col = 0; col < ds; col++) {
        float sum = 0.0f;
        for (size_t row = 0; row < ds; row++) {
            sum += A[col + ds * row];  // Accumulate sum across rows in a column
        }
        sums[col] = sum;
    }
}

bool validate(const float *gpu_results, const float *cpu_results, size_t sz){
  for (size_t i = 0; i < sz; i++) {
    if (abs(gpu_results[i] - cpu_results[i]) > 1e-2) {  // Allow small floating-point tolerance
      printf("Results mismatch at index %lu, GPU result: %f, CPU result: %f\n", i, gpu_results[i], cpu_results[i]);
      return false;
    }
  }
  return true;
}

int main() {
  float *h_A, *h_sums, *h_sums_CPU, *d_A, *d_sums;
  h_A = new float[DSIZE * DSIZE];  // allocate space for matrix in host memory
  h_sums = new float[DSIZE]();     // allocate space for GPU results in host memory
  h_sums_CPU = new float[DSIZE](); // allocate space for CPU validation sums

  // Seed the random number generator
  srand(time(0));  // Use current time to seed the random number generator

  // Initialize matrix in host memory with random values
  for (int i = 0; i < DSIZE * DSIZE; i++) {
    h_A[i] = static_cast<float>(rand()) / RAND_MAX;  // Random float between 0 and 1
  }

  cudaMalloc(&d_A, DSIZE * DSIZE * sizeof(float));  // allocate device space for matrix A
  cudaMalloc(&d_sums, DSIZE * sizeof(float));       // allocate device space for sums vector
  cudaCheckErrors("cudaMalloc failure"); // error checking

  // copy matrix A to device:
  cudaMemcpy(d_A, h_A, DSIZE * DSIZE * sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy H2D failure");

  //cuda processing sequence step 1 is complete
  row_sums<<<(DSIZE + block_size - 1) / block_size, block_size>>>(d_A, d_sums, DSIZE);
  cudaCheckErrors("kernel launch failure");

  // copy row sums from device to host:
  cudaMemcpy(h_sums, d_sums, DSIZE * sizeof(float), cudaMemcpyDeviceToHost);
  cudaCheckErrors("kernel execution failure or cudaMemcpy D2H failure");

  // CPU row sums calculation for validation
  cpu_row_sums(h_A, h_sums_CPU, DSIZE);

  if (!validate(h_sums, h_sums_CPU, DSIZE)) return -1;
  printf("Row sums correct!\n");

  cudaMemset(d_sums, 0, DSIZE * sizeof(float));  // reset sums on device

  column_sums<<<(DSIZE + block_size - 1) / block_size, block_size>>>(d_A, d_sums, DSIZE);
  cudaCheckErrors("kernel launch failure");

  // copy column sums from device to host:
  cudaMemcpy(h_sums, d_sums, DSIZE * sizeof(float), cudaMemcpyDeviceToHost);
  cudaCheckErrors("kernel execution failure or cudaMemcpy D2H failure");

  // CPU column sums calculation for validation
  cpu_column_sums(h_A, h_sums_CPU, DSIZE);

  if (!validate(h_sums, h_sums_CPU, DSIZE)) return -1;
  printf("Column sums correct!\n");

  // Cleanup
  delete[] h_A;
  delete[] h_sums;
  delete[] h_sums_CPU;
  cudaFree(d_A);
  cudaFree(d_sums);

  return 0;
}
