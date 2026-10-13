#include <iostream>
#include <cmath>
#include <iomanip>

// --------------------------------------------------
// Compute S = Q * K^T / sqrt(d)
// --------------------------------------------------

void computeScores(
    const float* Q,
    const float* K,
    float* scores,
    int sequenceLength,
    int d)
{
    const float scale = 1.0f / std::sqrt(static_cast<float>(d));

    for (int i = 0; i < sequenceLength; ++i)
    {
        for (int j = 0; j < sequenceLength; ++j)
        {
            float sum = 0.0f;

            for (int k = 0; k < d; ++k)
            {
                sum +=
                    Q[i * d + k] *
                    K[j * d + k];
            }

            scores[i * sequenceLength + j] =
                sum * scale;
        }
    }
}

// --------------------------------------------------
// Row-wise stable softmax
// --------------------------------------------------

void softmaxRows(
    float* matrix,
    int rows,
    int cols)
{
    for (int row = 0; row < rows; ++row)
    {
        // Find maximum value in this row
        float maxValue =
            matrix[row * cols];

        for (int col = 1; col < cols; ++col)
        {
            float value =
                matrix[row * cols + col];

            if (value > maxValue)
            {
                maxValue = value;
            }
        }

        // exponentiate and calculate denominator
        float sum = 0.0f;

        for (int col = 0; col < cols; ++col)
        {
            float expValue =
                std::exp(
                    matrix[row * cols + col]
                    - maxValue
                );

            matrix[row * cols + col] =
                expValue;

            sum += expValue;
        }

        // normalize
        for (int col = 0; col < cols; ++col)
        {
            matrix[row * cols + col] /= sum;
        }
    }
}

// --------------------------------------------------
// O = attentionWeights * V
// --------------------------------------------------

void computeOutput(
    const float* attentionWeights,
    const float* V,
    float* output,
    int sequenceLength,
    int d)
{
    for (int row = 0; row < sequenceLength; ++row)
    {
        for (int col = 0; col < d; ++col)
        {
            float sum = 0.0f;

            for (int k = 0; k < sequenceLength; ++k)
            {
                sum +=
                    attentionWeights[
                        row * sequenceLength + k
                    ]
                    *
                    V[
                        k * d + col
                    ];
            }

            output[row * d + col] = sum;
        }
    }
}

// --------------------------------------------------
// Print helper
// --------------------------------------------------

void printMatrix(
    const float* matrix,
    int rows,
    int cols,
    const char* name)
{
    std::cout << "\n" << name << ":\n";

    for (int row = 0; row < rows; ++row)
    {
        for (int col = 0; col < cols; ++col)
        {
            std::cout
                << std::fixed
                << std::setprecision(4)
                << matrix[row * cols + col]
                << " ";
        }

        std::cout << '\n';
    }
}

// --------------------------------------------------
// Main
// --------------------------------------------------

int main()
{
    const int sequenceLength = 3;
    const int d = 2;

    // 3 tokens, each with embedding dimension 2

    float Q[sequenceLength * d] =
    {
        1.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 1.0f
    };

    float K[sequenceLength * d] =
    {
        1.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 1.0f
    };

    float V[sequenceLength * d] =
    {
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f
    };

    float scores[
        sequenceLength * sequenceLength
    ];

    float output[
        sequenceLength * d
    ];

    // 1. QK^T / sqrt(d)
    computeScores(
        Q,
        K,
        scores,
        sequenceLength,
        d
    );

    printMatrix(
        scores,
        sequenceLength,
        sequenceLength,
        "Scaled attention scores"
    );

    // 2. Softmax
    softmaxRows(
        scores,
        sequenceLength,
        sequenceLength
    );

    printMatrix(
        scores,
        sequenceLength,
        sequenceLength,
        "Attention weights"
    );

    // 3. softmax(...) * V
    computeOutput(
        scores,
        V,
        output,
        sequenceLength,
        d
    );

    printMatrix(
        output,
        sequenceLength,
        d,
        "Attention output"
    );

    return 0;
}