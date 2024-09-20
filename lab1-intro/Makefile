# Makefile

# Default target to build all
.PHONY: all
all: hello vadd mmul

# Rule for hello
hello: hello.cu
	nvcc -o hello hello.cu

# Rule for vector addition
vadd: vector_add.cu
	nvcc -o vadd vector_add.cu

# Rule for matrix multiplication
mmul: matrix_mul.cu
	nvcc -o mmul matrix_mul.cu

# Help target
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  make hello    - Compile hello.cu"
	@echo "  make vadd     - Compile vector_add.cu"
	@echo "  make mmul     - Compile matrix_mul.cu"
	@echo "  make all      - Compile all (hello, vadd, mmul)"
	@echo "  make clean    - Clean up generated binaries"

# Clean target
.PHONY: clean
clean:
	rm -f hello vadd mmul
