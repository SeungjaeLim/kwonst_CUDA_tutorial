#include <stdio.h>
#include <algorithm>
#include <stdlib.h>
#include <time.h>

using namespace std;

#define N 4096
#define RADIUS 3
#define BLOCK_SIZE 16

/**
 * @brief 1D stencil computation using shared memory.
 * 
 * This CUDA kernel applies a 1D stencil operation to an input array. 
 * Each thread computes the sum of its neighboring elements (defined by the RADIUS) 
 * and writes the result into an output array. Shared memory is used to reduce the number 
 * of global memory accesses by loading a block of data plus a halo region into shared memory.
 *
 * @param in   Pointer to the input array in device memory. The input array is expected 
 *             to have halo regions on both sides of size RADIUS to accommodate boundary conditions.
 * @param out  Pointer to the output array in device memory. The result of the stencil computation
 *             will be written to this array.
 *
 * Details:
 * - Shared memory is allocated to hold a block of input elements plus additional elements 
 *   for the halo regions (left and right of the block). 
 * - Each thread loads one element from the input array into shared memory.
 * - Threads at the boundaries of the block also load the halo elements into shared memory.
 * - After loading, all threads synchronize to ensure that shared memory is fully populated.
 * - Each thread then computes the sum of the element at its position and its neighbors 
 *   within the RADIUS. The result is written into the output array at the corresponding position.
 *
 * Block and Grid Dimensions:
 * - Block size is defined by BLOCK_SIZE (e.g., 16 threads per block).
 * - Grid size is determined by dividing the array size (N) by BLOCK_SIZE.
 *
 * @note The input array `in` should have extra elements to account for the halo regions. 
 * The kernel operates on the non-halo region of the array (in + RADIUS to out + RADIUS).
 */
__global__ void stencil_1d(int *in, int *out) {

    /* TODO: Allocate shared memory with enough space for the block and the halo region.
     * This includes elements for the block itself and an extra RADIUS elements on each side. */
    __shared__ int temp[/*TODO*/];
    
    /* TODO: Calculate global and local indices.
     * The global index corresponds to the position in the input array, while the local
     * index corresponds to the position in the shared memory buffer, with an offset for RADIUS. */
    int gindex = /*TODO*/
    int lindex = /*TODO*/

    /* TODO: Load the central part of the block into shared memory.
     * Each thread loads one element from the global input array into the appropriate position
     * in shared memory. */
    temp[/*TODO*/] = in[/*TODO*/];

    /* TODO: Load the halo elements into shared memory.
     * Threads at the beginning and end of the block load the necessary RADIUS elements
     * from the left and right halo regions, respectively. */
    if (threadIdx.x < RADIUS) {
        temp[/*TODO*/] = in[/*TODO*/];
        temp[/*TODO*/] = in[/*TODO*/];
    }

    /* TODO: Synchronize threads to ensure all data is loaded before applying the stencil.
     * This is crucial to avoid race conditions where some threads are still loading
     * while others start computing. */
    /*TODO*/

    /* TODO: Apply the stencil operation.
     * Each thread computes the sum of its neighboring elements as defined by the RADIUS.
     * The sum is calculated using the values stored in shared memory. */
    int result = 0;
    for (int offset = -RADIUS; offset <= RADIUS; offset++)
        result += temp[/*TODO*/];

    // Store the result
    out[gindex] = result;
}

// Function to generate random integers in the input array
void fill_random_ints(int *x, int n) {
    for (int i = 0; i < n; i++) {
        x[i] = rand() % 10;  // Random integer between 0 and 9
    }
}

// CPU-based stencil computation
void cpu_stencil_1d(int *in, int *out) {
    for (int i = RADIUS; i < N + RADIUS; i++) {
        int result = 0;
        for (int j = -RADIUS; j <= RADIUS; j++) {
            result += in[i + j];
        }
        out[i] = result;
    }
}

int main(void) {
    srand(time(NULL));  // Seed the random number generator

    int *in, *out, *cpu_out;  // host copies
    int *d_in, *d_out;        // device copies

    /* TODO: Allocate space for host arrays.
     * Each array needs to accommodate N elements plus 2*RADIUS for the halo region. */
    int size = /*TODO*/
    in = /*TODO*/
    out = /*TODO*/
    cpu_out = (int *)malloc(size);

    // Fill input array with random values
    fill_random_ints(in, N + 2 * RADIUS);

    // Allocate space for device copies
    cudaMalloc((void **)&d_in, size);
    cudaMalloc((void **)&d_out, size);

    // Copy to device
    cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice);

    /* TODO: Launch the stencil_1d kernel on the GPU.
     * The kernel is launched with N/BLOCK_SIZE blocks and BLOCK_SIZE threads per block.
     * The +RADIUS is used to skip the halo region in the device input array. */
    stencil_1d<<<N / BLOCK_SIZE, BLOCK_SIZE>>>(/*TODO*/, /*TODO*/);

    // Copy result back to host
    cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);

    // Perform the same operation on the CPU
    cpu_stencil_1d(in, cpu_out);

    // Check the results
    bool success = true;
    printf("Checking the result\n");
    for (int i = RADIUS; i < N + RADIUS; i++) {
        if (out[i] != cpu_out[i]) {
            printf("Mismatch at index %d, GPU result: %d, CPU result: %d\n", i, out[i], cpu_out[i]);
            success = false;
        }
    }

    // Final success message
    printf("%s\n", success ? "Success!" : "Failure!");

    // Cleanup
    free(in); free(out); free(cpu_out);
    cudaFree(d_in); cudaFree(d_out);

    return 0;
}
