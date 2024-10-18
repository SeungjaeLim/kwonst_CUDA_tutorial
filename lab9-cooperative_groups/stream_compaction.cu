#include <stdio.h>
#include <stdlib.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/remove.h>
#include <cooperative_groups.h>

typedef int mytype;
const int test_dsize = 256;
const int nTPB = 256;

/**
 * @brief Predicate function to test if a value should be removed.
 * 
 * This function returns 0 if the input data matches the `testval`, indicating that
 * the value should be removed. Otherwise, it returns 1.
 * 
 * @tparam T Type of the data to be tested.
 * @param data The input data to be tested.
 * @param testval The value to check against.
 * @return unsigned 0 if the value should be removed, 1 otherwise.
 */
template <typename T>
__device__ unsigned predicate_test(T data, T testval) {
    if (data == testval) return 0;
    return 1;
}

using namespace cooperative_groups;

/**
 * @brief CUDA kernel to remove elements from an array based on a predicate.
 * 
 * This kernel performs two main tasks:
 * 1. Computes a prefix sum (scan) over the predicate values (1 if the value should be kept, 0 otherwise).
 * 2. Moves the values that pass the predicate test to their correct locations in the output array.
 * 
 * The kernel uses cooperative groups to synchronize threads within the block and across the grid.
 * (Assume dsize is divisible by nTPB)
 *
 * @tparam T Type of the input and output data.
 * @param idata The input data array.
 * @param remove_val The value to be removed from the array.
 * @param odata The output data array.
 * @param idxs An array to store the prefix sum results (indices).
 * @param dsize The size of the input data array.
 */
template <typename T>
__global__ void my_remove_if(const T* __restrict__ idata, 
                             const T remove_val, 
                             T* __restrict__ odata, 
                             unsigned* __restrict__ idxs, 
                             const unsigned dsize) {

    __shared__ unsigned sidxs[nTPB];
    /* TODO: Define the cooperative group for the current thread block.
     * This will be used for synchronization within the block.
     */
    auto g = ;
    
    /* TODO: Define the grid-wide cooperative group for synchronization across all blocks.
     * This allows synchronizing across the entire grid.
     */
    auto gg = ;

    /* TODO: Calculate the thread's local ID (thread index within the block).
     * This value will be used for accessing shared memory within the block.
     */
    unsigned tidx = ;

    /* TODO: Calculate the global index for each thread in the grid.
     * This index is used to access the input data array in a grid-stride loop.
     */
    unsigned gidx = ;

    unsigned gridSize = g.size() * gridDim.x;

    // First use grid-stride loop to have each block do a prefix sum over data set
    for (unsigned i = gidx; i < dsize; i += gridSize) {
        /* TODO: Evaluate the predicate (whether the data matches the remove_val) 
         * and store the result in shared memory.
         */

        /* TODO: Perform an in-block prefix sum (scan) using a binary tree approach.
         * Synchronize threads between each step.
         */
        for (int j = 1; j < g.size(); j <<= 1) {
            
        }

        /* TODO: Store the final result of the scan in the global memory (idxs array),
         * which will be used later to compute final indices.
         */

    }

    /* TODO: Synchronize all blocks in the grid to ensure the prefix sum is complete.
     * This is necessary before proceeding to the next step.
     */
    

    // Then compute final index, and move input data to output location
    unsigned stride = 0;
    for (unsigned i = gidx; i < dsize; i += gridSize) {
        T temp = idata[i];
        if (predicate_test(temp, remove_val)) {
            unsigned my_idx = idxs[i];

            /* TODO: Adjust the index by summing the final results of the prefix sum from previous blocks.
             * This ensures that the values are placed in the correct position in the output array.
             */
            
            /* TODO: Write the value to the output array at the calculated index (my_idx - 1).
             * This completes the removal operation.
             */
        }
        stride++;
    }
}

int main() {
    // Data setup
    mytype* d_idata, *d_odata, *h_data;
    unsigned* d_idxs;
    size_t tsize = ((size_t)test_dsize) * sizeof(mytype);
    h_data = (mytype*)malloc(tsize);
    cudaMalloc(&d_idata, tsize);
    cudaMalloc(&d_odata, tsize);
    cudaMemset(d_odata, 0, tsize);
    cudaMalloc(&d_idxs, test_dsize * sizeof(unsigned));

    // Check for support and device configuration
    // And calculate maximum grid size
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) {
        printf("cuda error: %s\n", cudaGetErrorString(err));
        return 0;
    }
    if (prop.cooperativeLaunch == 0) {
        printf("cooperative launch not supported\n");
        return 0;
    }

    int numSM = prop.multiProcessorCount;
    printf("number of SMs = %d\n", numSM);

    /* TODO: Determine the number of active blocks per SM (Streaming Multiprocessor).
     * This call uses cudaOccupancyMaxActiveBlocksPerMultiprocessor to determine the 
     * optimal number of blocks per SM for the given kernel (my_remove_if).
     */
    int numBlkPerSM;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlkPerSM, , , 0);
    printf("number of blocks per SM = %d\n", numBlkPerSM);

    // Test 1: No remove values
    for (int i = 0; i < test_dsize; i++) h_data[i] = i;
    cudaMemcpy(d_idata, h_data, tsize, cudaMemcpyHostToDevice);
    cudaStream_t str;
    cudaStreamCreate(&str);
    mytype remove_val = -1;
    unsigned ds = test_dsize;
    void* args[] = { (void*)&d_idata, (void*)&remove_val, (void*)&d_odata, (void*)&d_idxs, (void*)&ds };
    dim3 grid(numBlkPerSM * numSM);
    dim3 block(nTPB);

    /* TODO: Launch the cooperative kernel using cudaLaunchCooperativeKernel. 
     * This ensures that the kernel is executed in cooperative mode with synchronization 
     * across all thread blocks.
     */
    cudaLaunchCooperativeKernel(,,,,,);

    err = cudaMemcpy(h_data, d_odata, tsize, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("cuda error: %s\n", cudaGetErrorString(err));
        return 0;
    }

    // Validate
    for (int i = 0; i < test_dsize; i++) {
        if (h_data[i] != i) {
            printf("mismatch 1 at %d, was: %d, should be: %d\n", i, h_data[i], i);
            return 1;
        }
    }

    // Test 2: With remove values
    int val = 0;
    for (int i = 0; i < test_dsize; i++) {
        if ((rand() / (float)RAND_MAX) > 0.5) {
            h_data[i] = val++;
        } else {
            h_data[i] = -1;
        }
    }

    thrust::device_vector<mytype> t_data(h_data, h_data + test_dsize);
    cudaMemcpy(d_idata, h_data, tsize, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    /* TODO: Launch the cooperative kernel using cudaLaunchCooperativeKernel. 
     * This ensures that the kernel is executed in cooperative mode with synchronization 
     * across all thread blocks.
     */
    cudaLaunchCooperativeKernel(,,,,,);
    
    cudaEventRecord(stop);

    float et;
    cudaMemcpy(h_data, d_odata, tsize, cudaMemcpyDeviceToHost);
    cudaEventElapsedTime(&et, start, stop);

    // Validate
    bool pass = true;
    for (int i = 0; i < val; i++) {
        if (h_data[i] != i) {
            printf("mismatch 2 at %d, was: %d, should be: %d\n", i, h_data[i], i);
            pass = false;
            break;
        }
    }
    printf("Test 1: %s\n", pass ? "PASS" : "FAIL");

    printf("kernel time: %fms\n", et);
    cudaEventRecord(start);
    thrust::remove(t_data.begin(), t_data.end(), -1);
    cudaEventRecord(stop);

    thrust::host_vector<mytype> th_data = t_data;

    // Validate
    pass = true;
    for (int i = 0; i < val; i++) {
        if (h_data[i] != th_data[i]) {
            printf("mismatch 3 at %d, was: %d, should be: %d\n", i, th_data[i], h_data[i]);
            pass = false;
            break;
        }
    }
    printf("Test 2: %s\n", pass ? "PASS" : "FAIL");

    cudaEventElapsedTime(&et, start, stop);
    printf("thrust time: %fms\n", et);

    return 0;
}
