# Makefile

# Default target to build all
.PHONY: all
all: linked_list_manual linked_list_um arr_inc_manual arr_inc_um arr_inc_prefetch

# Rule for linked_list_manual
linked_list_manual: linked_list_manual.cu
	nvcc -o linked_list_manual linked_list_manual.cu

# Rule for linked_list_um
linked_list_um: linked_list_um.cu
	nvcc -o linked_list_um linked_list_um.cu

# Rule for arr_inc_manual
arr_inc_manual: array_inc_manual.cu
	nvcc -o arr_inc_manual array_inc_manual.cu

# Rule for arr_inc_um
arr_inc_um: array_inc_um.cu
	nvcc -o arr_inc_um array_inc_um.cu

# Rule for arr_inc_prefetch
arr_inc_prefetch: array_inc_prefetch.cu
	nvcc -o arr_inc_prefetch array_inc_prefetch.cu


# Help target
.PHONY: help
help:
	@echo "Available targets:"
	@
	@echo "  make clean      - Clean up generated binaries"
	@echo "  make linked_list_manual    - Compile linked_list_manual.cu"
	@echo "  make linked_list_um        - Compile linked_list_um.cu"
	@echo "  make arr_inc_manual		- Compile array_inc_manual.cu"
	@echo "  make arr_inc_um		- Compile array_inc_um.cu"
	@echo "  make arr_inc_prefetch	- Compile array_inc_prefetch.cu"
	@echo "  make help       - Print this help"

# Clean target
.PHONY: clean
clean:
	rm -f linked_list_manual linked_list_um arr_inc_manual arr_inc_um arr_inc_prefetch
