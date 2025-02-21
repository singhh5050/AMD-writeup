#include "matrix_utils.hpp"
#include <fstream>
#include <iomanip>

// Function declarations from other files
void cpuMatMul(const std::vector<float>&, const std::vector<float>&, std::vector<float>&, int, int, int);
void naiveGPUMatMul(const std::vector<float>&, const std::vector<float>&, std::vector<float>&, int, int, int);
void tiledGPUMatMul(const std::vector<float>&, const std::vector<float>&, std::vector<float>&, int, int, int);
void optimizedGPUMatMul(const std::vector<float>&, const std::vector<float>&, std::vector<float>&, int, int, int);

struct BenchmarkResult {
    int size;
    double cpu_time;
    double naive_gpu_time;
    double tiled_gpu_time;
    double optimized_gpu_time;
    double naive_speedup;
    double tiled_speedup;
    double optimized_speedup;
};

BenchmarkResult runBenchmark(int size) {
    BenchmarkResult result;
    result.size = size;
    
    // Initialize matrices
    std::vector<float> A, B, C_cpu, C_naive, C_tiled, C_optimized;
    initializeMatrix(A, size, size);
    initializeMatrix(B, size, size);
    C_cpu.resize(size * size);
    C_naive.resize(size * size);
    C_tiled.resize(size * size);
    C_optimized.resize(size * size);
    
    // Run CPU version
    Timer timer;
    cpuMatMul(A, B, C_cpu, size, size, size);
    result.cpu_time = timer.elapsed();
    
    // Run naive GPU version
    timer = Timer();
    naiveGPUMatMul(A, B, C_naive, size, size, size);
    result.naive_gpu_time = timer.elapsed();
    
    // Run tiled GPU version
    timer = Timer();
    tiledGPUMatMul(A, B, C_tiled, size, size, size);
    result.tiled_gpu_time = timer.elapsed();
    
    // Run optimized GPU version
    timer = Timer();
    optimizedGPUMatMul(A, B, C_optimized, size, size, size);
    result.optimized_gpu_time = timer.elapsed();
    
    // Calculate speedups
    result.naive_speedup = result.cpu_time / result.naive_gpu_time;
    result.tiled_speedup = result.cpu_time / result.tiled_gpu_time;
    result.optimized_speedup = result.cpu_time / result.optimized_gpu_time;
    
    // Verify results
    if (!verifyResults(C_cpu, C_naive) ||
        !verifyResults(C_cpu, C_tiled) ||
        !verifyResults(C_cpu, C_optimized)) {
        std::cerr << "Result verification failed for size " << size << std::endl;
    }
    
    return result;
}

void saveResults(const std::vector<BenchmarkResult>& results, const std::string& filename) {
    std::ofstream file(filename);
    file << "Matrix_Size,CPU_Time_ms,Naive_GPU_Time_ms,Tiled_GPU_Time_ms,Optimized_GPU_Time_ms,"
         << "Naive_Speedup,Tiled_Speedup,Optimized_Speedup\n";
         
    for (const auto& result : results) {
        file << result.size << ","
             << std::fixed << std::setprecision(2)
             << result.cpu_time << ","
             << result.naive_gpu_time << ","
             << result.tiled_gpu_time << ","
             << result.optimized_gpu_time << ","
             << result.naive_speedup << ","
             << result.tiled_speedup << ","
             << result.optimized_speedup << "\n";
    }
}

int main() {
    // Matrix sizes to test
    std::vector<int> sizes = {128, 256, 512, 1024, 2048, 4096};
    std::vector<BenchmarkResult> results;
    
    std::cout << "Running benchmarks...\n";
    for (int size : sizes) {
        std::cout << "Testing size " << size << "x" << size << "...\n";
        results.push_back(runBenchmark(size));
        
        // Print current results
        const auto& result = results.back();
        std::cout << "Results for " << size << "x" << size << ":\n"
                  << std::fixed << std::setprecision(2)
                  << "  CPU time: " << result.cpu_time << " ms\n"
                  << "  Naive GPU time: " << result.naive_gpu_time 
                  << " ms (speedup: " << result.naive_speedup << "x)\n"
                  << "  Tiled GPU time: " << result.tiled_gpu_time 
                  << " ms (speedup: " << result.tiled_speedup << "x)\n"
                  << "  Optimized GPU time: " << result.optimized_gpu_time 
                  << " ms (speedup: " << result.optimized_speedup << "x)\n\n";
    }
    
    // Save results to CSV file
    saveResults(results, "../data/benchmark_results.csv");
    std::cout << "Results saved to benchmark_results.csv\n";
    
    return 0;
} 