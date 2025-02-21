# Matrix Multiplication on AMD GPUs

Hey! This is my implementation of matrix multiplication comparing CPU vs GPU approaches using AMD's HIP framework. Quick heads up: This is a theoretical implementation since I don't have access to AMD GPU hardware, but the code structure is all there and ready to go if you've got the right setup.

## What's In Here

I've organized the code into different implementations, each one getting progressively more optimized:
```
.
├── src/
│   ├── cpu/          # Basic CPU version (the classic triple loop)
│   ├── naive/        # Simple GPU version (just to get it working)
│   ├── tiled/        # Smarter GPU version with shared memory
│   └── optimized/    # Full-on optimized version with all the tricks
├── include/          # Common header stuff
├── benchmarks/       # Code to measure performance
├── data/            # Where benchmark results would go
└── docs/            # Detailed writeup of how it all works
```

## What You'd Need to Run This

I couldn't run this myself (Mac problems 😅), but here's what you'd need:
- Linux system (AMD GPU stuff doesn't work on Mac)
- AMD GPU
- ROCm platform installed
- HIP toolkit
- CMake (version 3.10+)
- C++ compiler

## How to Build (If You've Got the Hardware)

1. Make a build directory:
```bash
mkdir build && cd build
```

2. Set up with CMake:
```bash
cmake ..
```

3. Build it:
```bash
make
```

## Running the Different Versions

Each version can be run separately:

```bash
# Try the CPU version
./cpu_matmul

# Basic GPU version
./naive_matmul

# Fancy shared memory version
./tiled_matmul

# Super optimized version
./optimized_matmul
```

## What Each Version Does

1. **CPU Version**: Just your basic triple-nested loop implementation
2. **Basic GPU**: Gets it running on the GPU, nothing fancy
3. **Tiled Version**: Uses shared memory to speed things up
4. **Optimized Version**: Goes all out with:
   - Loop unrolling
   - Register blocking
   - Tuned block sizes

## Results?

Since I couldn't actually run this (no AMD GPU access), check out [docs/ANALYSIS.md](docs/ANALYSIS.md) for a theoretical discussion of how these different versions should perform and why.

## Important Note

Just to be super clear - this is all theoretical code that I wrote to demonstrate the progression from basic to optimized GPU programming on AMD hardware. I wasn't able to run it myself (stuck with a Mac), but the code structure is solid and should work if you've got the right setup. Think of it as a "here's how you'd do it" guide rather than a "here's how it performs" benchmark. 