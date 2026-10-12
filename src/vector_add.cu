#include <iostream>
#include <cuda_runtime.h>

__global__
void vectorAdd(
    const float* a,
    const float* b,
    float* c,
    int n)
{
    int i =
        blockIdx.x * blockDim.x
        + threadIdx.x;

    if (i < n)
    {
        c[i] = a[i] + b[i];
    }
}

int main()
{
    const int N = 1024;

    const size_t bytes =
        N * sizeof(float);

    float* h_a = new float[N];
    float* h_b = new float[N];
    float* h_c = new float[N];

    for (int i = 0; i < N; ++i)
    {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(2 * i);
    }

    float* d_a;
    float* d_b;
    float* d_c;

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(
        d_a,
        h_a,
        bytes,
        cudaMemcpyHostToDevice
    );

    cudaMemcpy(
        d_b,
        h_b,
        bytes,
        cudaMemcpyHostToDevice
    );

    const int threadsPerBlock = 256;

    const int blocks =
        (N + threadsPerBlock - 1)
        / threadsPerBlock;

    vectorAdd<<<blocks, threadsPerBlock>>>(
        d_a,
        d_b,
        d_c,
        N
    );

    cudaDeviceSynchronize();

    cudaMemcpy(
        h_c,
        d_c,
        bytes,
        cudaMemcpyDeviceToHost
    );

    for (int i = 0; i < 10; ++i)
    {
        std::cout
            << h_a[i]
            << " + "
            << h_b[i]
            << " = "
            << h_c[i]
            << '\n';
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    delete[] h_a;
    delete[] h_b;
    delete[] h_c;

    return 0;
}