# Matrix Multiplication Acceleration on AMD GPUs

This repository contains implementations and benchmarks of matrix multiplication algorithms optimized for AMD GPUs using HIP (Heterogeneous-Computing Interface for Portability).

## Project Structure

```
.
├── src/
│   ├── cpu/          # CPU baseline implementation
│   ├── naive/        # Naive GPU implementation
│   ├── tiled/        # Tiled GPU implementation with shared memory
│   └── optimized/    # Optimized GPU implementation
├── include/          # Header files
├── benchmarks/       # Benchmarking code and scripts
├── data/            # Benchmark results and data
└── docs/            # Documentation and analysis
```

## Prerequisites

- AMD ROCm platform
- HIP (Heterogeneous-Computing Interface for Portability)
- CMake (version 3.10 or higher)
- C++ compiler with C++11 support

## Building the Project

1. Create a build directory:
```bash
mkdir build && cd build
```

2. Configure with CMake:
```bash
cmake ..
```

3. Build:
```bash
make
```

## Running the Benchmarks

Each implementation can be run separately:

```bash
# Run CPU baseline
./cpu_matmul

# Run naive GPU implementation
./naive_matmul

# Run tiled GPU implementation
./tiled_matmul

# Run optimized GPU implementation
./optimized_matmul
```

## Implementations

1. **CPU Baseline**: Standard CPU implementation using triple nested loops
2. **Naive GPU**: Basic GPU implementation with global memory access
3. **Tiled GPU**: Improved implementation using shared memory tiling
4. **Optimized GPU**: Further optimized version with:
   - Loop unrolling
   - Register blocking
   - Tuned block dimensions

## Results

Detailed benchmarking results and analysis can be found in the [docs/ANALYSIS.md](docs/ANALYSIS.md) file.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 