#include <math.h>
#include <iostream>
#include <time.h>
#include <sys/time.h>
#include <stdio.h>

// modifiable
typedef float ft;
const int chunks = 64;
const size_t ds = 1024*1024*chunks;
const int count = 22;
const int num_gpus = 2;

// not modifiable
const float sqrt_2PIf = 2.5066282747946493232942230134974f;
const double sqrt_2PI = 2.5066282747946493232942230134974;

/**
 * @brief Compute the Gaussian PDF for a given value.
 * 
 * This function calculates the Gaussian probability density function (PDF)
 * for a given value and standard deviation.
 * 
 * @param val   The value for which the PDF is computed.
 * @param sigma The standard deviation for the Gaussian distribution.
 * @return float The computed Gaussian PDF value.
 */
__device__ float gpdf(float val, float sigma) {
    return expf(-0.5f * val * val) / (sigma * sqrt_2PIf);
}

/**
 * @brief Compute the Gaussian PDF for a given double value.
 * 
 * Similar to the float version, but works with double precision.
 * 
 * @param val   The value for which the PDF is computed.
 * @param sigma The standard deviation for the Gaussian distribution.
 * @return double The computed Gaussian PDF value.
 */
__device__ double gpdf(double val, double sigma) {
    return exp(-0.5 * val * val) / (sigma * sqrt_2PI);
}

/**
 * @brief CUDA kernel to compute the average Gaussian PDF over a window.
 * 
 * This kernel computes the average Gaussian PDF value over a window of values
 * around each point. It processes the data in parallel using CUDA threads.
 * 
 * @param x     Pointer to the input data array.
 * @param y     Pointer to the output data array.
 * @param mean  The mean of the Gaussian distribution.
 * @param sigma The standard deviation of the Gaussian distribution.
 * @param n     The number of data points.
 */
__global__ void gaussian_pdf(const ft * __restrict__ x, ft * __restrict__ y, const ft mean, const ft sigma, const int n) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < n) {
        ft in = x[idx] - (count / 2) * 0.01f;
        ft out = 0;
        for (int i = 0; i < count; i++) {
            ft temp = (in - mean) / sigma;
            out += gpdf(temp, sigma);
            in += 0.01f;
        }
        y[idx] = out / count;
    }
}

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

// host-based timing
#define USECPSEC 1000000ULL

/**
 * @brief Measure elapsed time in microseconds.
 * 
 * This function calculates the elapsed time in microseconds since the provided
 * start time.
 * 
 * @param start  The start time in microseconds.
 * @return unsigned long long The elapsed time in microseconds.
 */
unsigned long long dtime_usec(unsigned long long start) {
    timeval tv;
    gettimeofday(&tv, 0);
    return ((tv.tv_sec * USECPSEC) + tv.tv_usec) - start;
}

int main() {
    ft *h_x, *d_x[num_gpus], *d_y[num_gpus];

    
    h_x = (ft *)malloc(ds * sizeof(ft));
    for (int i = 0; i < num_gpus; i++) {
     /* 
      * TODO: Allocate device memory using cudaMalloc for each GPU.
      * Allocate input (d_x) and output (d_y) memory on both GPUs.
      */;
    }
    cudaCheckErrors("allocation error");

    
    for (int i = 0; i < num_gpus; i++) {
        for (size_t j = 0; j < ds; j++) {
            h_x[j] = rand() / (ft)RAND_MAX;
        }
       /* 
        * TODO: Perform memory copies to transfer input data from the host (h_x) to each GPU.
        */
        
    }
    cudaCheckErrors("copy error");
    unsigned long long et1 = dtime_usec(0);

    /* 
     * TODO: Launch the gaussian_pdf kernel on each GPU in parallel.
     * This loop sets the current device and launches the kernel.
     */
    
    cudaCheckErrors("execution error");
    et1 = dtime_usec(et1);
    std::cout << "elapsed time: " << et1 / (float)USECPSEC << std::endl;

    return 0;
}
