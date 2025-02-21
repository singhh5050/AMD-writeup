#include "matrix_utils.hpp"

__global__ void naiveMatMulKernel(const float* A,
                                 const float* B,
                                 float* C,
                                 int M, int N, int K) {
    int row = hipBlockIdx.y * hipBlockDim.y + hipThreadIdx.y;
    int col = hipBlockIdx.x * hipBlockDim.x + hipThreadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

void naiveGPUMatMul(const std::vector<float>& A,
                    const std::vector<float>& B,
                    std::vector<float>& C,
                    int M, int N, int K) {
    float *d_A, *d_B, *d_C;
    
    // Allocate device memory
    HIP_CHECK(hipMalloc(&d_A, M * K * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_B, K * N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_C, M * N * sizeof(float)));
    
    // Copy input matrices to device
    HIP_CHECK(hipMemcpy(d_A, A.data(), M * K * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, B.data(), K * N * sizeof(float), hipMemcpyHostToDevice));
    
    // Set up grid and block dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (M + blockDim.y - 1) / blockDim.y);
    
    // Launch kernel
    hipLaunchKernelGGL(naiveMatMulKernel,
                       gridDim,
                       blockDim,
                       0, 0,
                       d_A, d_B, d_C,
                       M, N, K);
    
    // Check for kernel launch errors
    HIP_CHECK(hipGetLastError());
    
    // Copy result back to host
    HIP_CHECK(hipMemcpy(C.data(), d_C, M * N * sizeof(float), hipMemcpyDeviceToHost));
    
    // Free device memory
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));
}

int main(int argc, char* argv[]) {
    // Matrix dimensions
    const int M = 1024;  // Rows of A
    const int N = 1024;  // Cols of B
    const int K = 1024;  // Cols of A / Rows of B
    
    // Initialize matrices
    std::vector<float> A, B, C, C_cpu;
    initializeMatrix(A, M, K);
    initializeMatrix(B, K, N);
    C.resize(M * N);
    C_cpu.resize(M * N);
    
    // Compute reference result on CPU
    cpuMatMul(A, B, C_cpu, M, N, K);
    
    // Perform GPU matrix multiplication and measure time
    Timer timer;
    naiveGPUMatMul(A, B, C, M, N, K);
    double elapsed = timer.elapsed();
    
    // Verify results
    if (verifyResults(C_cpu, C)) {
        std::cout << "Results verified successfully!" << std::endl;
    } else {
        std::cout << "Results verification failed!" << std::endl;
        return 1;
    }
    
    // Print performance metrics
    printPerformanceMetrics("Naive GPU", M, N, K, elapsed);
    
    return 0;
} 