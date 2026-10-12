#include <iostream>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>

__global__
void matmulKernel(
    const float* A,
    const float* B,
    float* C,
    int M,
    int K,
    int N)
{
    int row =
        blockIdx.y * blockDim.y
        + threadIdx.y;

    int col =
        blockIdx.x * blockDim.x
        + threadIdx.x;

    if (row < M && col < N)
    {
        float sum = 0.0f;

        for (int k = 0; k < K; ++k)
        {
            sum +=
                A[row * K + k]
                *
                B[k * N + col];
        }

        C[row * N + col] = sum;
    }
}

void cpuMatmul(
    const float* A,
    const float* B,
    float* C,
    int M,
    int K,
    int N)
{
    for (int row = 0; row < M; ++row)
    {
        for (int col = 0; col < N; ++col)
        {
            float sum = 0.0f;

            for (int k = 0; k < K; ++k)
            {
                sum +=
                    A[row * K + k]
                    *
                    B[k * N + col];
            }

            C[row * N + col] = sum;
        }
    }
}

int main()
{
    const int M = 512;
    const int K = 512;
    const int N = 512;

    const size_t bytesA =
        M * K * sizeof(float);

    const size_t bytesB =
        K * N * sizeof(float);

    const size_t bytesC =
        M * N * sizeof(float);

    float* h_A =
        new float[M * K];

    float* h_B =
        new float[K * N];

    float* h_C_cpu =
        new float[M * N];

    float* h_C_gpu =
        new float[M * N];

    for (int i = 0; i < M * K; ++i)
    {
        h_A[i] =
            static_cast<float>(i % 10);
    }

    for (int i = 0; i < K * N; ++i)
    {
        h_B[i] =
            static_cast<float>(i % 7);
    }

    // -----------------------------
    // CPU benchmark
    // -----------------------------

    auto cpu_start =
        std::chrono::high_resolution_clock::now();

    cpuMatmul(
        h_A,
        h_B,
        h_C_cpu,
        M,
        K,
        N
    );

    auto cpu_end =
        std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli>
        cpu_time =
            cpu_end - cpu_start;

    // -----------------------------
    // GPU memory
    // -----------------------------

    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, bytesA);
    cudaMalloc(&d_B, bytesB);
    cudaMalloc(&d_C, bytesC);

    cudaMemcpy(
        d_A,
        h_A,
        bytesA,
        cudaMemcpyHostToDevice
    );

    cudaMemcpy(
        d_B,
        h_B,
        bytesB,
        cudaMemcpyHostToDevice
    );

    // -----------------------------
    // Kernel configuration
    // -----------------------------

    dim3 threadsPerBlock(16, 16);

    dim3 blocks(
        (N + threadsPerBlock.x - 1)
            / threadsPerBlock.x,

        (M + threadsPerBlock.y - 1)
            / threadsPerBlock.y
    );

    // -----------------------------
    // GPU timing
    // -----------------------------

    cudaEvent_t start;
    cudaEvent_t stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    matmulKernel<<<blocks, threadsPerBlock>>>(
        d_A,
        d_B,
        d_C,
        M,
        K,
        N
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_time = 0.0f;

    cudaEventElapsedTime(
        &gpu_time,
        start,
        stop
    );

    // -----------------------------
    // GPU -> CPU
    // -----------------------------

    cudaMemcpy(
        h_C_gpu,
        d_C,
        bytesC,
        cudaMemcpyDeviceToHost
    );

    // -----------------------------
    // Correctness check
    // -----------------------------

    bool correct = true;

    const float tolerance = 1e-3f;

    for (int i = 0; i < M * N; ++i)
    {
        float difference =
            std::fabs(
                h_C_cpu[i]
                -
                h_C_gpu[i]
            );

        if (difference > tolerance)
        {
            correct = false;

            std::cout
                << "Mismatch at index "
                << i
                << ": CPU="
                << h_C_cpu[i]
                << ", GPU="
                << h_C_gpu[i]
                << ", diff="
                << difference
                << '\n';

            break;
        }
    }

    // -----------------------------
    // Results
    // -----------------------------

    std::cout
        << "Matrix: "
        << M << "x" << K
        << " * "
        << K << "x" << N
        << "\n\n";

    std::cout
        << "CPU time: "
        << cpu_time.count()
        << " ms\n";

    std::cout
        << "GPU kernel time: "
        << gpu_time
        << " ms\n";

    std::cout
        << "Results match: "
        << (correct ? "YES" : "NO")
        << '\n';

    // -----------------------------
    // Cleanup
    // -----------------------------

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    delete[] h_A;
    delete[] h_B;
    delete[] h_C_cpu;
    delete[] h_C_gpu;

    return 0;
}