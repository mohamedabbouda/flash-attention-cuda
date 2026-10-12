#include <iostream>
#include <chrono>
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
        c[i] = a[i] + b[i];
}

int main()
{
    const int N = 10'000'000;

    const size_t bytes =
        N * sizeof(float);

    float* h_a = new float[N];
    float* h_b = new float[N];
    float* h_c_cpu = new float[N];
    float* h_c_gpu = new float[N];

    for (int i = 0; i < N; ++i)
    {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(2 * i);
    }

    auto cpu_start =
        std::chrono::high_resolution_clock::now();

    for (int i = 0; i < N; ++i)
        h_c_cpu[i] = h_a[i] + h_b[i];

    auto cpu_end =
        std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli>
        cpu_time = cpu_end - cpu_start;

    float *d_a, *d_b, *d_c;

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    auto gpu_total_start =
        std::chrono::high_resolution_clock::now();

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

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    vectorAdd<<<blocks, threadsPerBlock>>>(
        d_a,
        d_b,
        d_c,
        N
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float kernel_time = 0.0f;

    cudaEventElapsedTime(
        &kernel_time,
        start,
        stop
    );

    cudaMemcpy(
        h_c_gpu,
        d_c,
        bytes,
        cudaMemcpyDeviceToHost
    );

    auto gpu_total_end =
        std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli>
        gpu_total_time =
            gpu_total_end - gpu_total_start;

    bool correct = true;

    for (int i = 0; i < N; ++i)
    {
        if (h_c_cpu[i] != h_c_gpu[i])
        {
            correct = false;
            break;
        }
    }

    std::cout
        << "CPU time: "
        << cpu_time.count()
        << " ms\n";

    std::cout
        << "GPU kernel only: "
        << kernel_time
        << " ms\n";

    std::cout
        << "GPU total including copies: "
        << gpu_total_time.count()
        << " ms\n";

    std::cout
        << "Correct: "
        << (correct ? "YES" : "NO")
        << '\n';

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    delete[] h_a;
    delete[] h_b;
    delete[] h_c_cpu;
    delete[] h_c_gpu;

    return 0;
}