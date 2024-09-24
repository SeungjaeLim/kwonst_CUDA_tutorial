# Makefile

# Default target to build all
.PHONY: all
all: vadd

# Rule for vector addition
vadd: vector_add.cu
	nvcc -o vadd vector_add.cu

# Help target
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  make vadd     - Compile vector_add.cu"
	@echo "  make clean    - Clean up generated binaries"

# Clean target
.PHONY: clean
clean:
	rm -f vadd
