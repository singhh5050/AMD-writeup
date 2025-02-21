#include "matrix_utils.hpp"

void cpuMatMul(const std::vector<float>& A,
               const std::vector<float>& B,
               std::vector<float>& C,
               int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main(int argc, char* argv[]) {
    // Matrix dimensions
    const int M = 1024;  // Rows of A
    const int N = 1024;  // Cols of B
    const int K = 1024;  // Cols of A / Rows of B
    
    // Initialize matrices
    std::vector<float> A, B, C;
    initializeMatrix(A, M, K);
    initializeMatrix(B, K, N);
    C.resize(M * N);
    
    // Perform CPU matrix multiplication and measure time
    Timer timer;
    cpuMatMul(A, B, C, M, N, K);
    double elapsed = timer.elapsed();
    
    // Print performance metrics
    printPerformanceMetrics("CPU Baseline", M, N, K, elapsed);
    
    return 0;
} 