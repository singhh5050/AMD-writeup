#include "matrix_utils.hpp"

#define TILE_SIZE 16
#define BLOCK_SIZE 16
#define UNROLL_FACTOR 4

__global__ void optimizedMatMulKernel(const float* A,
                                    const float* B,
                                    float* C,
                                    int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int row = hipBlockIdx.y * TILE_SIZE + hipThreadIdx.y;
    int col = hipBlockIdx.x * TILE_SIZE + hipThreadIdx.x;
    
    // Register blocking for accumulation
    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tile from A into shared memory
        if (row < M && t * TILE_SIZE + hipThreadIdx.x < K) {
            As[hipThreadIdx.y][hipThreadIdx.x] = A[row * K + t * TILE_SIZE + hipThreadIdx.x];
        } else {
            As[hipThreadIdx.y][hipThreadIdx.x] = 0.0f;
        }
        
        // Load tile from B into shared memory
        if (t * TILE_SIZE + hipThreadIdx.y < K && col < N) {
            Bs[hipThreadIdx.y][hipThreadIdx.x] = B[(t * TILE_SIZE + hipThreadIdx.y) * N + col];
        } else {
            Bs[hipThreadIdx.y][hipThreadIdx.x] = 0.0f;
        }
        
        // Synchronize to make sure the tiles are loaded
        __syncthreads();
        
        // Compute partial dot product over this tile with loop unrolling
        #pragma unroll UNROLL_FACTOR
        for (int i = 0; i < TILE_SIZE; i += UNROLL_FACTOR) {
            sum += As[hipThreadIdx.y][i] * Bs[i][hipThreadIdx.x];
            sum += As[hipThreadIdx.y][i + 1] * Bs[i + 1][hipThreadIdx.x];
            sum += As[hipThreadIdx.y][i + 2] * Bs[i + 2][hipThreadIdx.x];
            sum += As[hipThreadIdx.y][i + 3] * Bs[i + 3][hipThreadIdx.x];
        }
        
        // Synchronize before loading the next tile
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void optimizedGPUMatMul(const std::vector<float>& A,
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
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // Launch kernel
    hipLaunchKernelGGL(optimizedMatMulKernel,
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
    optimizedGPUMatMul(A, B, C, M, N, K);
    double elapsed = timer.elapsed();
    
    // Verify results
    if (verifyResults(C_cpu, C)) {
        std::cout << "Results verified successfully!" << std::endl;
    } else {
        std::cout << "Results verification failed!" << std::endl;
        return 1;
    }
    
    // Print performance metrics
    printPerformanceMetrics("Optimized GPU", M, N, K, elapsed);
    
    return 0;
} 