#pragma once

#include <vector>
#include <random>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <hip/hip_runtime.h>

// Error checking macro for HIP calls
#define HIP_CHECK(call)                                                    \
{                                                                         \
    hipError_t err = call;                                               \
    if (err != hipSuccess)                                               \
    {                                                                    \
        printf("HIP error %s:%d: '%s'\n", __FILE__, __LINE__,           \
               hipGetErrorString(err));                                  \
        exit(EXIT_FAILURE);                                              \
    }                                                                    \
}

// Timer class for benchmarking
class Timer {
public:
    Timer() : start_(std::chrono::high_resolution_clock::now()) {}
    
    double elapsed() {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(now - start_).count();
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};

// Matrix initialization functions
inline void initializeMatrix(std::vector<float>& matrix, int rows, int cols) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    matrix.resize(rows * cols);
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = dis(gen);
    }
}

// Matrix verification function
inline bool verifyResults(const std::vector<float>& expected, 
                        const std::vector<float>& actual,
                        float tolerance = 1e-5) {
    if (expected.size() != actual.size()) {
        std::cout << "Size mismatch!" << std::endl;
        return false;
    }
    
    for (size_t i = 0; i < expected.size(); ++i) {
        if (std::abs(expected[i] - actual[i]) > tolerance) {
            std::cout << "Mismatch at index " << i 
                      << ": Expected " << expected[i] 
                      << ", got " << actual[i] << std::endl;
            return false;
        }
    }
    return true;
}

// Print matrix dimensions and performance metrics
inline void printPerformanceMetrics(const char* implementation,
                                  int M, int N, int K,
                                  double time_ms) {
    double flops = 2.0 * M * N * K;  // Multiply-add operations
    double gflops = (flops * 1e-9) / (time_ms * 1e-3);  // Convert to GFLOPS
    
    std::cout << std::fixed << std::setprecision(2)
              << implementation << " Performance:\n"
              << "Matrix Dimensions: " << M << "x" << K << " * " 
              << K << "x" << N << "\n"
              << "Execution Time: " << time_ms << " ms\n"
              << "Performance: " << gflops << " GFLOPS\n"
              << std::endl;
} 