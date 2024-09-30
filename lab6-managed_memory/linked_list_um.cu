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

struct list_elem {
  int key;
  list_elem *next;
};

/**
 * @brief Allocate memory using cudaMallocManaged.
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
 * @brief Print the key of a specific element in the linked list.
 * 
 * This function takes a pointer to the head of a linked list and prints the key value of the nth element.
 * It is designed to work on both host and device (with __host__ __device__ qualifiers).
 * 
 * @param list     Pointer to the base of the linked list.
 * @param ele_num  Index of the element whose key is to be printed.
 */
__host__ __device__
void print_element(list_elem *list, int ele_num){
  list_elem *elem = list;
  for (int i = 0; i < ele_num; i++)
    elem = elem->next;
  printf("key = %d\n", elem->key);
}

/**
 * @brief CUDA kernel to print a specific element in the linked list.
 * 
 * This CUDA kernel calls the print_element function to print the key of the nth element
 * in the linked list on the GPU.
 * 
 * @param list     Pointer to the base of the linked list in device memory.
 * @param ele_num  Index of the element to print.
 */
__global__ void gpu_print_element(list_elem *list, int ele_num){
  print_element(list, ele_num);
}

const int num_elem = 256;
const int ele = 255;
int main(){

  list_elem *list_base, *list;
  alloc_bytes(list_base, sizeof(list_elem));
  list = list_base;
  for (int i = 0; i < num_elem; i++) {
    list->key = i;
    alloc_bytes(list->next, sizeof(list_elem));
    list = list->next;
  }
  print_element(list_base, ele);
  /* 
   * TODO: Synchronize the device to ensure the kernel execution completes.
   * Check for any errors during execution using cudaCheckErrors.
   */
  cudaCheckErrors("cuda error!");
}
