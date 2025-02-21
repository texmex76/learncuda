# Compiler
NVCC := nvcc

# Executable name
TARGET := main

# Source file
SRC := main.cu

# Compilation flags
NVCC_FLAGS := -g -G -Wno-deprecated-gpu-targets

# Default rule
all: $(TARGET)

# Compile CUDA source file
$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $(SRC)

# Clean up generated files
clean:
	rm -f $(TARGET)

.PHONY: all clean
