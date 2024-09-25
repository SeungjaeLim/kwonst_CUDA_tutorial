# Makefile

# Default target to build all
.PHONY: all
all: msum

# Rule for Matrix Sum
msum: matrix_sums.cu
	nvcc -o msum matrix_sums.cu

# Help target
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  make msum     - Compile matrix_sums.cu"
	@echo "  make clean    - Clean up generated binaries"

# Clean target
.PHONY: clean
clean:
	rm -f msum
