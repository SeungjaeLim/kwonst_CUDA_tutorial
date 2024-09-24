#include <stdio.h>
#include <stdlib.h>  // For rand()
#include <math.h>    // For abs()

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

const int DSIZE = 32 * 1048576;  // Total size of the vectors

/**
 * @brief CUDA kernel for vector addition: C = A + B, using a grid-stride loop.
 * 
 * This kernel performs element-wise addition of two input vectors A and B, and stores 
 * the result in vector C. The computation is parallelized across multiple threads, with 
 * each thread processing multiple elements by using a grid-stride loop.
 *
 * @param A   Pointer to the input vector A in device memory (size: ds elements).
 * @param B   Pointer to the input vector B in device memory (size: ds elements).
 * @param C   Pointer to the output vector C in device memory (size: ds elements).
 * @param ds  The size of the vectors (number of elements in A, B, and C).
 *
 * Grid-Stride Loop:
 * - Each thread calculates a starting index based on its global index (`idx`), which is 
 *   derived from the block and thread indices.
 * - Instead of each thread processing only one element, it processes multiple elements 
 *   in a loop by advancing the index by the total number of threads in the grid.
 * - This allows for better utilization of GPU resources when the number of threads is 
 *   smaller than the size of the vector.
 *
 * Details:
 * - The grid-stride loop allows each thread to process multiple elements, improving efficiency.
 * - Threads keep processing elements until they cover the entire vector length (`ds`).
 *
 * Grid and Block Dimensions:
 * - Grid size is specified by the number of blocks.
 * - Block size is determined by the number of threads per block.
 * 
 * @note The grid-stride loop ensures all elements are covered even if the number of threads 
 *       is less than the total size of the vector.
 */
__global__ void vadd(const float *A, const float *B, float *C, int ds) {
    /* TODO: Calculate the global index for the current thread.
     * The index is based on the block and thread indices, and it determines which element
     * of the vectors A, B, and C the thread will operate on. */
    /*TODO*/

    /* TODO: Implement a grid-stride loop.
     * In a grid-stride loop, each thread processes multiple elements by looping over the 
     * vector in steps equal to the total number of threads in the grid. This ensures that 
     * even if the number of threads is smaller than the vector size, all elements are covered. */
    /*TODO*/

    /* TODO: Each thread processes multiple elements.
     * The thread starts at its initial index and continues to add the stride to process 
     * the next set of elements, until it has processed the entire vector. 
     * Each thread adds the corresponding elements from A and B and stores the result in C. */
    /*TODO*/
}

// CPU implementation of vector addition
void cpu_vadd(const float *A, const float *B, float *C, int ds) {
    for (int i = 0; i < ds; i++) {
        C[i] = A[i] + B[i];
    }
}

int main() {

    // Allocate host memory for vectors A, B, C, and C_CPU for comparison
    float *h_A = new float[DSIZE];
    float *h_B = new float[DSIZE];
    float *h_C = new float[DSIZE];  // GPU result
    float *h_C_CPU = new float[DSIZE];  // CPU result for comparison

    // Initialize vectors A and B with random values
    for (int i = 0; i < DSIZE; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
        h_C[i] = 0;
        h_C_CPU[i] = 0;  // Initialize CPU result vector to 0
    }

    // Allocate device memory for vectors A, B, and C
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, DSIZE * sizeof(float));
    cudaMalloc(&d_B, DSIZE * sizeof(float));
    cudaMalloc(&d_C, DSIZE * sizeof(float));
    cudaCheckErrors("cudaMalloc failure");

    // Copy vectors A and B from host to device
    cudaMemcpy(d_A, h_A, DSIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, DSIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy H2D failure");

    /* TODO: Launch the vector addition kernel on the GPU.
     * Specify the number of blocks and threads, and call the kernel with the correct grid configuration. */
    int blocks = 160;   // TODO: Modify this line for experimentation
    int threads = 1024; // TODO: Modify this line for experimentation
    
    vadd<<<blocks, threads>>>(d_A, d_B, d_C, DSIZE);
    cudaCheckErrors("kernel launch failure");

    // Copy result from device to host
    cudaMemcpy(h_C, d_C, DSIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckErrors("cudaMemcpy D2H failure");

    // Perform vector addition on the CPU for comparison
    cpu_vadd(h_A, h_B, h_C_CPU, DSIZE);

    // Compare the GPU result with the CPU result
    bool success = true;
    for (int i = 0; i < DSIZE; i++) {
        if (fabs(h_C[i] - h_C_CPU[i]) > 1e-5) {  // Allow small floating-point tolerance
            printf("Mismatch at index %d: GPU result = %f, CPU result = %f\n", i, h_C[i], h_C_CPU[i]);
            success = false;
            break;
        }
    }

    // Print result of comparison
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
