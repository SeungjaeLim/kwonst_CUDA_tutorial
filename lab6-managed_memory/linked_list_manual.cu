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
 * @brief Allocate memory dynamically for a pointer using malloc.
 * 
 * This template function is used to allocate memory for the pointer passed as a reference. It takes
 * the pointer and the number of bytes to be allocated.
 * 
 * @param ptr        Reference to the pointer to allocate memory for.
 * @param num_bytes  Number of bytes to allocate.
 */
template <typename T>
void alloc_bytes(T &ptr, size_t num_bytes) {
    ptr = (T)malloc(num_bytes);
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
void print_element(list_elem *list, int ele_num) {
    list_elem *elem = list;
    for (int i = 0; i < ele_num; i++)
        elem = elem->next;
    printf("key = %d\n", elem->key);
}

/**
 * @brief CUDA kernel to print a specific element from a linked list on the device.
 * 
 * The kernel calls the print_element function to print the key value of the nth element from the list.
 * 
 * @param list     Pointer to the base of the linked list (on device).
 * @param ele_num  Index of the element to print.
 */
__global__ void gpu_print_element(list_elem *list, int ele_num) {
    print_element(list, ele_num);
}

const int num_elem = 256;
const int ele = 255;

int main() {
    list_elem *list_base, *list;

    // Allocate and initialize the linked list on the host
    alloc_bytes(list_base, sizeof(list_elem));
    list = list_base;
    for (int i = 0; i < num_elem; i++) {
        list->key = i;
        if (i != num_elem - 1) {
            alloc_bytes(list->next, sizeof(list_elem));
        } else {
            list->next = nullptr;
        }
        list = list->next;
    }

    // Print a specific element from the list on the host
    print_element(list_base, ele);

    /* 
     * TODO: Allocate memory for the first element of the list on the GPU using cudaMalloc.
     */

    /* 
     * TODO: Copy the entire linked list structure from the host to the GPU.
     * For each element in the list, memory is allocated on the GPU, and the data is copied.
     * Additionally, the next pointer for each node is updated on the GPU to maintain the linked structure.
     */

    /* 
     * TODO: Launch a CUDA kernel to test the GPU implementation by printing an element from the list.
     * This will use the gpu_print_element kernel to print a specific element from the list on the device.
     */
    gpu_print_element<<<1, 1>>>(/*TODO*/, ele);
    cudaDeviceSynchronize();
    cudaCheckErrors("cuda error!");

    // Free the host and device memory (optional but good practice)
    list_elem *temp = list_base;
    while (temp != nullptr) {
        list_elem *next_temp = temp->next;
        free(temp);
        temp = next_temp;
    }

    /* 
     * TODO: Free the memory allocated on the device for the linked list.
     * This is done by iterating through the list on the device and freeing each node using cudaFree.
     */

    return 0;
}
