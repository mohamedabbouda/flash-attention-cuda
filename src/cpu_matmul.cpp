#include <iostream>

void matmul(
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

void printMatrix(
    const float* matrix,
    int rows,
    int cols)
{
    for (int row = 0; row < rows; ++row)
    {
        for (int col = 0; col < cols; ++col)
        {
            std::cout
                << matrix[row * cols + col]
                << " ";
        }

        std::cout << '\n';
    }
}

int main()
{
    const int M = 2;
    const int K = 2;
    const int N = 2;

    float A[M * K] =
    {
        1, 2,
        3, 4
    };

    float B[K * N] =
    {
        5, 6,
        7, 8
    };

    float C[M * N] = {};

    matmul(A, B, C, M, K, N);

    std::cout << "Result:\n";

    printMatrix(C, M, N);

    return 0;
}