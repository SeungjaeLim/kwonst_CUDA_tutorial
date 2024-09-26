# Makefile

# Default target to build all
.PHONY: all
all: reduct maxred msum

# Rule for Reductions
reduct: reductions.cu
	nvcc -o reduct reductions.cu

# Rule for Max Reductions
maxred: max_reduction.cu
	nvcc -o maxred max_reduction.cu		

# Rule for Matrix Sum
msum: matrix_sums.cu
	nvcc -o msum matrix_sums.cu

# Help target
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  make reduct     - Compile reductions.cu"
	@echo "  make maxred     - Compile max_reduction.cu"
	@echo "  make msum       - Compile matrix_sums.cu"
	@echo "  make clean      - Clean up generated binaries"
	@echo "  make help       - Print this help"

# Clean target
.PHONY: clean
clean:
	rm -f reduct maxred msum
