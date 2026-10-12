#include <iostream>
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

int main()
{
    const int M = 2;
    const int K = 2;
    const int N = 2;

    float h_A[] =
    {
        1, 2,
        3, 4
    };

    float h_B[] =
    {
        5, 6,
        7, 8
    };

    float h_C[M * N] = {};

    size_t bytesA =
        M * K * sizeof(float);

    size_t bytesB =
        K * N * sizeof(float);

    size_t bytesC =
        M * N * sizeof(float);

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

    dim3 threadsPerBlock(16, 16);

    dim3 blocks(
        (N + threadsPerBlock.x - 1)
            / threadsPerBlock.x,

        (M + threadsPerBlock.y - 1)
            / threadsPerBlock.y
    );

    matmulKernel<<<blocks, threadsPerBlock>>>(
        d_A,
        d_B,
        d_C,
        M,
        K,
        N
    );

    cudaDeviceSynchronize();

    cudaMemcpy(
        h_C,
        d_C,
        bytesC,
        cudaMemcpyDeviceToHost
    );

    std::cout << "Result:\n";

    for (int row = 0; row < M; ++row)
    {
        for (int col = 0; col < N; ++col)
        {
            std::cout
                << h_C[row * N + col]
                << " ";
        }

        std::cout << '\n';
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}