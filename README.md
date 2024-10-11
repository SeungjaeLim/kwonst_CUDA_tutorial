# CASYS Kwonst CUDA Tutorial
This repository contains tutorials and labs for learning CUDA programming, including concepts like shared memory, parallelism, and kernel optimization. The repository is structured into different labs with corresponding Makefiles for easy compilation and execution.

## Lab
- **lab1-intro/**: Contains Lab 1, which introduces basic CUDA programming with examples like "hello world", vector addition, and matrix multiplication.
- **lab2-shared_memory/**: Contains Lab 2, focusing on using shared memory in CUDA to optimize matrix multiplication and 1D stencil computations.
- **lab3-grid_stride_loop/**: Contains Lab 3, which demonstrates the use of grid-stride loops to efficiently perform vector addition and maximize GPU resource utilization
- **lab4-matrix_sums/**: Contains Lab 4, which focuses on calculating row and column sums of a matrix using CUDA, and using Nsight Compute to analyze kernel performance, focusing on memory load requests and transactions.
- **lab5-reductions/**: Contains Lab 5, which covers advanced CUDA reduction techniques, comparing atomic reduction, parallel reduction with atomic finish, and warp-shuffle reduction.
- **lab6-managed_memory/**: Contains Lab 6, which demonstrates porting linked lists and array operations to the GPU using manual memory management, Unified Memory (UM), and prefetching, with a focus on profiling GPU page faults and memory migrations.
- **lab7-concurrency/**: Contains Lab 7, which explores concurrency in CUDA programming through three tasks: Gaussian PDF computation on a single GPU, using CUDA streams to overlap computation and memory transfers, and distributing the workload across multiple GPUs.

## How to Start
To get started with this CUDA tutorial, follow these steps:
```
# Clone the repository
git clone https://github.com/SeungjaeLim/kwonst_CUDA_tutorial.git

# Navigate into the project directory
cd kwonst_CUDA_tutorial

# Build and run the Docker container with the tutorial environment
make up
```
### Docker Setup
This project uses Docker to create a containerized environment with CUDA support. You can build and run the container using the provided Makefile commands.

### Makefile Commands
The Makefile provides various management commands for setting up and running the project. Here's how to use the Makefile:

```
make build            # Build the cu_tutorial project.
make preprocess       # Preprocess step.
make run              # Boot up Docker container.
make up               # Build and run the project.
make rm               # Remove Docker container.
make stop             # Stop Docker container.
make reset            # Stop and remove Docker container.
make docker-setup     # Setup Docker permissions for the user.
```