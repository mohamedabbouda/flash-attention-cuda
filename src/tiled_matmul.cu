#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

#define TILE 16

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

__global__
void tiledMatmulKernel(
    const float* A,
    const float* B,
    float* C,
    int M,
    int K,
    int N)
{
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row =
        blockIdx.y * TILE
        + threadIdx.y;

    int col =
        blockIdx.x * TILE
        + threadIdx.x;

    float sum = 0.0f;

    int numTiles =
        (K + TILE - 1) / TILE;

    for (int tile = 0; tile < numTiles; ++tile)
    {
        int aCol =
            tile * TILE
            + threadIdx.x;

        int bRow =
            tile * TILE
            + threadIdx.y;

        if (row < M && aCol < K)
        {
            As[threadIdx.y][threadIdx.x] =
                A[row * K + aCol];
        }
        else
        {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (bRow < K && col < N)
        {
            Bs[threadIdx.y][threadIdx.x] =
                B[bRow * N + col];
        }
        else
        {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE; ++k)
        {
            sum +=
                As[threadIdx.y][k]
                *
                Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N)
    {
        C[row * N + col] = sum;
    }
}

int main()
{
    const int M = 512;
    const int K = 512;
    const int N = 512;

    size_t bytesA =
        M * K * sizeof(float);

    size_t bytesB =
        K * N * sizeof(float);

    size_t bytesC =
        M * N * sizeof(float);

    float* h_A = new float[M * K];
    float* h_B = new float[K * N];

    float* h_C_cpu =
        new float[M * N];

    float* h_C_gpu =
        new float[M * N];

    for (int i = 0; i < M * K; ++i)
        h_A[i] = static_cast<float>(i % 10);

    for (int i = 0; i < K * N; ++i)
        h_B[i] = static_cast<float>(i % 7);

    cpuMatmul(
        h_A,
        h_B,
        h_C_cpu,
        M,
        K,
        N
    );

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

    dim3 threadsPerBlock(TILE, TILE);

    dim3 blocks(
        (N + TILE - 1) / TILE,
        (M + TILE - 1) / TILE
    );

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    tiledMatmulKernel<<<blocks, threadsPerBlock>>>(
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

    cudaMemcpy(
        h_C_gpu,
        d_C,
        bytesC,
        cudaMemcpyDeviceToHost
    );

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
                << '\n';

            break;
        }
    }

    std::cout
        << "Tiled GPU kernel time: "
        << gpu_time
        << " ms\n";

    std::cout
        << "Sample C[0]: "
        << h_C_gpu[0]
        << '\n';

    std::cout
        << "Results match: "
        << (correct ? "YES" : "NO")
        << '\n';

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