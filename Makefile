# Compiler
NVCC = nvcc

# Source and Output
SRC = src/polaroid_effect.cu
EXECUTABLE = polaroid_effect

# Default target
build: $(EXECUTABLE)

# Build the executable
$(EXECUTABLE): $(SRC)
	$(NVCC) -o $@ $^

# Run the executable
run: $(EXECUTABLE)
	./$(EXECUTABLE)

# Clean up build artifacts
clean:
	rm -f $(EXECUTABLE)

.PHONY: clean all run 
