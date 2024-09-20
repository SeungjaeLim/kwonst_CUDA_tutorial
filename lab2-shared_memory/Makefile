# Makefile

# Default target to build all
.PHONY: all
all: mmul stc

# Rule for matrix multiplication (shared memory)
mmul: matrix_mul_shared.cu
	nvcc -o mmul matrix_mul_shared.cu

# Rule for 1D stencil
stc: stencil_1d.cu
	nvcc -o stc stencil_1d.cu

# Help target
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  make mmul     - Compile matrix_mul_shared.cu"
	@echo "  make stc      - Compile stencil_1d.cu"
	@echo "  make all      - Compile all (mmul, stc)"
	@echo "  make clean    - Clean up generated binaries"

# Clean target
.PHONY: clean
clean:
	rm -f mmul stc
