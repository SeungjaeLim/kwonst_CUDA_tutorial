#include <stdio.h>
#include <stdlib.h>
#include <time.h>

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

const int DSIZE = 1024;
const int block_size = 32;  // CUDA maximum is 1024 *total* threads in block

/**
 * @brief Matrix multiplication kernel using shared memory. Computes C = A * B.
 * 
 * This CUDA kernel performs matrix multiplication by dividing the matrix into blocks and
 * using shared memory for improved performance. Each thread calculates one element of the
 * output matrix by loading sub-blocks of the input matrices into shared memory, performing
 * the dot product of corresponding rows and columns, and accumulating the result.
 *
 * @param A   Pointer to matrix A on the device (input).
 * @param B   Pointer to matrix B on the device (input).
 * @param C   Pointer to matrix C on the device (output).
 * @param ds  The size of the matrix (assuming square matrix of size ds x ds).
 * 
 * Block dimensions: block_size x block_size
 * Grid dimensions: (DSIZE / block_size) x (DSIZE / block_size)
 * 
 * Threads in each block collaborate by loading sub-blocks of matrices A and B into shared
 * memory and calculating partial dot products. Each thread computes one element of matrix C.
 * Synchronization (`__syncthreads`) is used to ensure that all threads have finished reading
 * their portion of the matrices before starting the dot product calculation.
 */

__global__ void mmul(const float *A, const float *B, float *C, int ds) {
    __shared__ float As[block_size][block_size];
    __shared__ float Bs[block_size][block_size];

    int idx = threadIdx.x + blockDim.x * blockIdx.x;  // thread x index
    int idy = threadIdx.y + blockDim.y * blockIdx.y;  // thread y index

    if ((idx < ds) && (idy < ds)) {
        float temp = 0;
        for (int i = 0; i < ds / block_size; i++) {
            /* TODO: Load one tile of matrix A and B into shared memory
             * Each thread loads a single element from A and B into the shared memory arrays As and Bs. */
            As[threadIdx.y][threadIdx.x] = /*TODO*/
            Bs[threadIdx.y][threadIdx.x] = /*TODO*/

            /* TODO: Synchronize all threads to ensure complete data load into shared memory
             * All threads must finish loading their part of the sub-tile before moving on. */
            /*TODO*/

            for (int k = 0; k < block_size; k++) {
                /* TODO: Perform the dot product for the loaded tile
                 * Each thread computes a portion of the dot product between rows of A and columns of B. */
                temp += /*TODO*/
            }
            /* TODO: Synchronize threads again to avoid conflicts when moving to the next tile
             * After computing the dot product, all threads must wait for others to finish before
             * loading the next tile. */
            /*TODO*/
        }
        /* TODO: Store the final computed value into the output matrix C
         * After all tiles have been processed, the accumulated result is stored in the global matrix C. */
        /*TODO*/
    }
}

void cpu_mmul(const float *A, const float *B, float *C, int ds) {
    for (int i = 0; i < ds; i++) {
        for (int j = 0; j < ds; j++) {
            float sum = 0;
            for (int k = 0; k < ds; k++) {
                sum += A[i * ds + k] * B[k * ds + j];
            }
            C[i * ds + j] = sum;
        }
    }
}

int main() {
    float *h_A, *h_B, *h_C, *h_C_CPU, *d_A, *d_B, *d_C;

    // seed the random number generator
    srand(time(NULL));

    clock_t t0, t1, t2, t3;
    double t1sum = 0.0;
    double t2sum = 0.0;
    double t3sum = 0.0;

    // start timing
    t0 = clock();

    h_A = new float[DSIZE * DSIZE];
    h_B = new float[DSIZE * DSIZE];
    h_C = new float[DSIZE * DSIZE];
    h_C_CPU = new float[DSIZE * DSIZE];

    // Initialize A and B with random values
    for (int i = 0; i < DSIZE * DSIZE; i++) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
        h_C[i] = 0;
        h_C_CPU[i] = 0;
    }

    // Initialization timing
    t1 = clock();
    t1sum = ((double)(t1 - t0)) / CLOCKS_PER_SEC;
    printf("Init took %f seconds.  Begin compute\n", t1sum);

    // Allocate device memory and copy input data over to GPU
    cudaMalloc(&d_A, DSIZE * DSIZE * sizeof(float));
    cudaMalloc(&d_B, DSIZE * DSIZE * sizeof(float));
    cudaMalloc(&d_C, DSIZE * DSIZE * sizeof(float));
    cudaCheckErrors("cudaMalloc failure");

    cudaMemcpy(d_A, h_A, DSIZE * DSIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, DSIZE * DSIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy H2D failure");

    // Launch kernel
    dim3 block(block_size, block_size);
    dim3 grid((DSIZE + block.x - 1) / block.x, (DSIZE + block.y - 1) / block.y);
    mmul<<<grid, block>>>(d_A, d_B, d_C, DSIZE);
    cudaCheckErrors("kernel launch failure");

    // Copy results back to host
    cudaMemcpy(h_C, d_C, DSIZE * DSIZE * sizeof(float), cudaMemcpyDeviceToHost);
    t2 = clock();
    t2sum = ((double)(t2 - t1)) / CLOCKS_PER_SEC;
    printf("GPU compute took %f seconds\n", t2sum);

    // Perform matrix multiplication on the CPU
    cpu_mmul(h_A, h_B, h_C_CPU, DSIZE);
    t3= clock();
    t3sum = ((double)(t3 - t2)) / CLOCKS_PER_SEC;
    printf("CPU compute took %f seconds\n", t3sum);
    
    // Verify the results
    bool success = true;
    for (int i = 0; i < DSIZE * DSIZE; i++) {
        if (abs(h_C[i] - h_C_CPU[i]) > 1) {
            printf("Mismatch at index %d: GPU = %f, CPU = %f\n", i, h_C[i], h_C_CPU[i]);
            success = false;
        }
    }

    if (success) {
        printf("Success!\n");
    } else {
        printf("Failure!\n");
    }

    // Clean up
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_CPU;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
