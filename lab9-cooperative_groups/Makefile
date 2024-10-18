# Makefile

# Default target to build both reduce and stream_compaction
.PHONY: all
all: reduce stream_compaction

# Rule for reduce
reduce: reduce.cu
	nvcc -o reduce reduce.cu

# Rule for stream_compaction
stream_compaction: stream_compaction.cu
	nvcc -o stream_compaction stream_compaction.cu

# Help target
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  make clean              - Clean up generated binaries"
	@echo "  make reduce             - Compile reduce.cu"
	@echo "  make stream_compaction  - Compile stream_compaction.cu"
	@echo "  make help               - Print this help"

# Clean target
.PHONY: clean
clean:
	rm -f reduce stream_compaction
