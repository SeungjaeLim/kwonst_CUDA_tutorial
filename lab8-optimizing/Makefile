# Makefile

# Default target to build only transpose-related files
.PHONY: all
all: transpose_naive transpose_sm transpose_bc

# Rule for naive (which compiles transpose_naive)
naive: transpose_naive

# Rule for sm (which compiles transpose_sm)
sm: transpose_sm

# Rule for bc (which compiles transpose_bc)
bc: transpose_bc

# Rule for transpose_naive
transpose_naive: transpose_naive.cu
	nvcc -o transpose_naive transpose_naive.cu

# Rule for transpose_sm
transpose_sm: transpose_sm.cu
	nvcc -o transpose_sm transpose_sm.cu

# Rule for transpose_bc
transpose_bc: transpose_bc.cu
	nvcc -o transpose_bc transpose_bc.cu

# Help target
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  make clean            - Clean up generated binaries"
	@echo "  make naive            - Compile transpose_naive.cu"
	@echo "  make sm               - Compile transpose_sm.cu"
	@echo "  make bc               - Compile transpose_bc.cu"
	@echo "  make help             - Print this help"

# Clean target
.PHONY: clean
clean:
	rm -f naive sm bc transpose_naive transpose_sm transpose_bc
