# Makefile

# Default target to build all
.PHONY: all
all: gpdf overlap multi

# Rule for gpdf
gpdf: gpdf.cu
	nvcc -o gpdf gpdf.cu

# Rule for overlap
overlap: gpdf_overlap.cu
	nvcc -o overlap gpdf_overlap.cu

# Rule for multi
multi: gpdf_multi.cu
	nvcc -o multi gpdf_multi.cu

# Help target
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  make clean      - Clean up generated binaries"
	@echo "  make gpdf       - Compile gpdf.cu"
	@echo "  make overlap    - Compile gpdf_overlap.cu"
	@echo "  make multi      - Compile gpdf_multi.cu"
	@echo "  make help       - Print this help"

# Clean target
.PHONY: clean
clean:
	rm -f gpdf overlap multi
