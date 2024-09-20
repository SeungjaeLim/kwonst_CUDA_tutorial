#include <stdio.h>
#include <stdlib.h>  // For rand() and srand()
#include <time.h>    // For timing and srand()

// Error checking macro
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

const int DSIZE = 1024;      // Matrix size is DSIZE x DSIZE
const int block_size = 16;   // CUDA maximum is 1024 total threads in a block

/**
 * @brief CUDA kernel for matrix multiplication: C = A * B.
 * 
 * This kernel performs matrix multiplication of two input matrices A and B, storing 
 * the result in matrix C. The operation is performed in a parallelized manner, where 
 * each thread computes one element of the resulting matrix by performing a dot product 
 * of a row from matrix A and a column from matrix B.
 * 
 * @param A   Pointer to matrix A in device memory (size: ds x ds).
 * @param B   Pointer to matrix B in device memory (size: ds x ds).
 * @param C   Pointer to matrix C in device memory (size: ds x ds, output).
 * @param ds  Size of the matrices (assuming square matrices of dimension ds x ds).
 * 
 * Details:
 * - Each thread calculates one element in the resulting matrix C.
 * - The global thread indices (idx and idy) determine which element the thread will compute.
 * - The kernel computes the dot product of the corresponding row from A and column from B 
 *   for the given (idx, idy) location in the output matrix C.
 * 
 * Grid and Block Dimensions:
 * - Block size is defined by `block_size`, typically a 2D block (e.g., 16x16 threads).
 * - Grid size is calculated based on the matrix size (ds) divided by the block size.
 */
__global__ void mmul(const float *A, const float *B, float *C, int ds) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;  // 1D thread index
    int idy = threadIdx.y + blockDim.y * blockIdx.y;  // 1D thread index
    if ((idx < ds) && (idy < ds)) {
        float temp = 0;
        for (int i = 0; i < ds; i++)
            /* TODO: Perform the dot product for the corresponding row in A and column in B.
            * Each thread will iterate over the entire row from A and column from B, multiplying
            * the corresponding elements and accumulating the result in temp. */
            /*TODO*/
        /* TODO: Store the computed result in matrix C.
         * Once the dot product is computed, it is stored in the appropriate element of matrix C. */
        /*TODO*/
    }
}

// CPU matrix multiplication for comparison
void cpu_mmul(const float *A, const float *B, float *C, int ds) {
    for (int i = 0; i < ds; i++) {
        for (int j = 0; j < ds; j++) {
            float temp = 0;
            for (int k = 0; k < ds; k++) {
                temp += A[i * ds + k] * B[k * ds + j];
            }
            C[i * ds + j] = temp;
        }
    }
}

int main() {
    float *h_A, *h_B, *h_C, *h_C_CPU, *d_A, *d_B, *d_C;

    // For timing
    clock_t t0, t1, t2, t3;
    double t1sum = 0.0, t2sum = 0.0, t3sum = 0.0;

    // Start timing
    t0 = clock();

    // Allocate host memory for matrices A, B, C, and C_CPU
    h_A = new float[DSIZE * DSIZE];
    h_B = new float[DSIZE * DSIZE];
    h_C = new float[DSIZE * DSIZE];
    h_C_CPU = new float[DSIZE * DSIZE];  // For CPU comparison

    // Fill matrices A and B with random values
    srand(time(NULL));  // Seed the random number generator
    for (int i = 0; i < DSIZE * DSIZE; i++) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;  // Random float between 0 and 1
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;  // Random float between 0 and 1
        h_C[i] = 0;
        h_C_CPU[i] = 0;
    }

    // Initialization timing
    t1 = clock();
    t1sum = ((double)(t1 - t0)) / CLOCKS_PER_SEC;
    printf("Init took %f seconds. Begin compute\n", t1sum);

    // Allocate device memory for matrices A, B, C
    cudaMalloc(&d_A, DSIZE * DSIZE * sizeof(float));
    cudaMalloc(&d_B, DSIZE * DSIZE * sizeof(float));
    cudaMalloc(&d_C, DSIZE * DSIZE * sizeof(float));
    cudaCheckErrors("cudaMalloc failure");

    // Copy host matrices A and B to device
    cudaMemcpy(d_A, h_A, DSIZE * DSIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, DSIZE * DSIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy H2D failure");

    // Launch matrix multiplication kernel
    dim3 block(block_size, block_size);  // dim3 variable holds 3 dimensions
    dim3 grid((DSIZE + block.x - 1) / block.x, (DSIZE + block.y - 1) / block.y);
    mmul<<<grid, block>>>(d_A, d_B, d_C, DSIZE);
    cudaCheckErrors("kernel launch failure");

    // Copy results back from device to host
    cudaMemcpy(h_C, d_C, DSIZE * DSIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckErrors("kernel execution failure or cudaMemcpy D2H failure");

    // GPU timing
    t2 = clock();
    t2sum = ((double)(t2 - t1)) / CLOCKS_PER_SEC;
    printf("GPU Done. GPU Compute took %f seconds\n", t2sum);

    // Perform matrix multiplication on the CPU for comparison
    // CPU timing

    cpu_mmul(h_A, h_B, h_C_CPU, DSIZE);
    t3 = clock();
    t3sum = ((double)(t3 - t2)) / CLOCKS_PER_SEC;
    printf("CPU Done. CPU Compute took %f seconds\n", t3sum);
    
    // Verify the results by comparing the GPU result with the CPU result
    bool success = true;
    for (int i = 0; i < DSIZE * DSIZE; i++) {
        if (abs(h_C[i] - h_C_CPU[i]) > 1e-1) {  // Allow for a small tolerance in floating-point comparison
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
