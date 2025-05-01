#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <chrono>
#include <cuda_runtime.h>
#include "cxxopts.hpp"

using namespace std::chrono;
using myclock = steady_clock;

// CUDA error checking macro
#define CU_ASSERT(code)                                                                                                \
    {                                                                                                                  \
        cudaError_t err = code;                                                                                        \
        if (err != cudaSuccess)                                                                                        \
        {                                                                                                              \
            fprintf(stderr, "CUDA error <<%s>> on line %d of file %s\n", cudaGetErrorString(err), __LINE__, __FILE__); \
            exit(1);                                                                                                   \
        }                                                                                                              \
    }
#define BLOCKSIZE 32
#define BM 128
#define BN 128
#define BK 8
#define NR 16 // number of results calculated per thread

// A: M x K matrix
// B: K x N matrix
// C: M x N matrix
__global__ void kernel1(int M, int N, int K, const float *A, const float *B, float *C)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N)
    {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k)
        {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
__global__ void transpose(int M, int K,
                          const float *A,
                          float *A_t)
{
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < M && c < K)
    {
        // write A[r,c] into A_t[c,r]
        A_t[c * M + r] = A[r * K + c];
    }
}
__global__ void kernel10(int M, int N, int K, const float *A, const float *B, float *C)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N)
    {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k)
        {
            sum += A[k * K + row] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
__global__ void kernel0(int M, int N, int K, const float *A, const float *B, float *C)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N)
    {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k)
        {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void kernel2(int M, int N, int K,
                        const float *A,
                        const float *B,
                        float *C)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float As[BLOCKSIZE][BLOCKSIZE+1];
    __shared__ float Bs[BLOCKSIZE][BLOCKSIZE+1];

    float sum = 0.0f;

    // slide over K in BLOCKSIZE-wide panels
    for (int bk = 0; bk < K; bk += BLOCKSIZE)
    {
        if (row < M && bk + threadIdx.x < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + (bk + threadIdx.x)];
        else 
            As[threadIdx.y][threadIdx.x] = 0.0f;

        if (bk + threadIdx.y < K && col < N)
            Bs[threadIdx.y][threadIdx.x] = B[(bk + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // compute the BLOCKSIZE-length dot product
        for (int t = 0; t < BLOCKSIZE; ++t)
            // sum += As[t][threadIdx.x] * Bs[t][threadIdx.y];
            sum += As[threadIdx.y][t] * Bs[t][threadIdx.x];

        __syncthreads();
    }
    // write back
    if (row < M && col < N)
        C[row * N + col] = sum;
}

__global__ void kernel3(int M, int N, int K,
                        const float *A,
                        const float *B,
                        float *C)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float As[BLOCKSIZE][BLOCKSIZE+1];
    __shared__ float Bs[BLOCKSIZE][BLOCKSIZE+1];

    float sum = 0.0f;

    // slide over K in BLOCKSIZE-wide panels
    for (int bk = 0; bk < K; bk += BLOCKSIZE)
    {
        // transposed loads of A and B to coalesce
        if (row < M && bk + threadIdx.x < K)
            As[threadIdx.x][threadIdx.y] = A[row * K + (bk + threadIdx.x)];
        else 
            As[threadIdx.x][threadIdx.y] = 0.0f;

        if (bk + threadIdx.y < K && col < N)
            Bs[threadIdx.x][threadIdx.y] = B[(bk + threadIdx.y) * N + col];
        else
            Bs[threadIdx.x][threadIdx.y] = 0.0f;

        __syncthreads();

        // compute the BLOCKSIZE-length dot product
        for (int t = 0; t < BLOCKSIZE; ++t)
            sum += As[t][threadIdx.y] * Bs[threadIdx.x][t];

        __syncthreads();
    }
    // write back
    if (row < M && col < N)
        C[row * N + col] = sum;
}

__global__ void kernel4(int M, int N, int K,
    const float *A,
    const float *B,
    float *C)
{

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    float sums[NR] = {0.0};

    // slide along the K dimension 
    for (int bk = 0; bk < K; bk += BK)
    {
        if (row < M && bk + threadIdx.x < K)
            As[threadIdx.x][threadIdx.y] = A[col * K + (bk + threadIdx.y)];
        
        if (bk + threadIdx.y < K && col < N)
            Bs[threadIdx.y][threadIdx.x] = B[(bk + threadIdx.y) * N + col]; 
        
        __syncthreads();
        
        for (int t = 0; t < BK; ++t) {
            float tmp = Bs[t][threadIdx.x];
            for (uint i = 0; i < NR; ++i) {
                sums[i] += As[threadIdx.y*BK+i][t] * tmp; 
            }
        }
        __syncthreads();
    }
    
    // write the final result
    for (uint i = 0; i < NR; ++i) {
        if (row*NR+i < M && col < N)
            C[(row*NR+i) * N + col] = sums[i];
    }
}

int main(int argc, char **argv)
{

    cxxopts::Options options("matrix_mul_cuda", "Matrix Multiplication using CUDA Managed Memory");
    options.add_options()("h,help", "usage information")("k,kernel", "kernel number", cxxopts::value<int>()->default_value("1"))
    ("s,size", "Dimension of square matrices (M=N=K)", cxxopts::value<int>()->default_value("64")); // default for warming up
    auto result = options.parse(argc, argv);

    if (result.count("help") > 0)
    {
        std::cout << options.help() << '\n';
        return 0;
    }

    int DIM = result["size"].as<int>();
    int M = DIM;
    int N = DIM;
    int K = DIM;

    printf("Running Matrix Multiplication for %d x %d matrices\n", M, N);

    float *A, *B, *C;                              // Pointers for matrices on host/device (managed memory)
    size_t bytesA = (size_t)M * K * sizeof(float); // Size of matrix A
    size_t bytesB = (size_t)K * N * sizeof(float); // Size of matrix B
    size_t bytesC = (size_t)M * N * sizeof(float); // Size of matrix C

    // Allocate managed memory accessible from both host and device
    CU_ASSERT(cudaMallocManaged(&A, bytesA));
    CU_ASSERT(cudaMallocManaged(&B, bytesB));
    CU_ASSERT(cudaMallocManaged(&C, bytesC));

    // A = 1.0f everywhere
    // B = 2.0f everywhere
    for (int i = 0; i < M; ++i)
    {
        for (int k = 0; k < K; ++k)
        {
            A[i * K + k] = 1.0f;
        }
    }
    for (int k = 0; k < K; ++k)
    {
        for (int j = 0; j < N; ++j)
        {
            B[k * N + j] = 2.0f;
        }
    }
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            C[i * N + j] = 0.0f;
        }
    }

    
    dim3 threadsPerBlock(BLOCKSIZE, BLOCKSIZE);
    
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // for tiling
    dim3 threadsPerBlock2(BN, BK);
    dim3 numBlocks2((N + BN - 1) / BN, (M + BM - 1) / BM);

    myclock::time_point before = myclock::now();
    int kernel = result["kernel"].as<int>();
    switch (kernel)
    {
    case 1:
        kernel1<<<numBlocks, threadsPerBlock>>>(M, N, K, A, B, C);
        break;
    case 0:
        kernel0<<<numBlocks, threadsPerBlock>>>(M, N, K, A, B, C);
        break;
    case 10:
    {
        float *A_t;
        size_t bytesA_t = (size_t)M * K * sizeof(float);
        CU_ASSERT(cudaMallocManaged(&A_t, bytesA_t));
        dim3 gridT(
            (K + threadsPerBlock.x - 1) / threadsPerBlock.x,
            (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
        transpose<<<gridT, threadsPerBlock>>>(M, K, A, A_t);
        cudaDeviceSynchronize();
        kernel10<<<numBlocks, threadsPerBlock>>>(M, N, K, A_t, B, C);
        break;
    }
    case 2:
        kernel2<<<numBlocks, threadsPerBlock>>>(M, N, K, A, B, C);
        break;
    case 3:
        kernel3<<<numBlocks, threadsPerBlock>>>(M, N, K, A, B, C);
        break;
    case 4: 
        kernel4<<<numBlocks2, threadsPerBlock2>>>(M, N, K, A, B, C);
        break;
    default:
        std::cout << "Invalid kernel\n";
        return 1;
    }

    // Wait for the GPU kernel to complete execution
    CU_ASSERT(cudaDeviceSynchronize());

    auto after = myclock::now();
    std::cout << "Matrix multiplication took: "
              << duration_cast<milliseconds>(after - before).count()
              << " ms\n";

    bool error_found = false;
    float expected_value = (float)K * 1.0f * 2.0f;
    float tolerance = 1e-5f; // Tolerance for floating point comparison

    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            float error = fabs(C[i * N + j] - expected_value);

            // Check if the error exceeds the tolerance
            if (error > tolerance && !error_found)
            {
                std::cout << "Error in Matrix Multiplication: " << error << "\n";
                error_found = true;
            }
        }
    }

    if (!error_found) {
        std::cout << "Matrix Multiplication correct\n";
    }

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}
