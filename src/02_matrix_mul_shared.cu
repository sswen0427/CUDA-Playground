#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <chrono>

#define CHECK(call)                                                            \
    {                                                                          \
        const cudaError_t error = call;                                        \
        if (error != cudaSuccess)                                              \
        {                                                                      \
            fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);             \
            fprintf(stderr, "code: %d, reason: %s\n", error,                   \
                    cudaGetErrorString(error));                                \
            exit(1);                                                           \
        }                                                                      \
    }

#define BLOCK_SIZE 16

__global__ void matrixMulShared(float* A, float* B, float* C, int M, int N, int K)
{
    __shared__ float tileA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float tileB[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    float value = 0.0f;
    for (int t = 0; t < (N + BLOCK_SIZE - 1) / BLOCK_SIZE; t++)
    {
        int a_row = row;
        int a_col = t * BLOCK_SIZE + threadIdx.x;
        int b_row = t * BLOCK_SIZE + threadIdx.y;
        int b_col = col;

        if (a_row < M && a_col < N)
            tileA[threadIdx.y][threadIdx.x] = A[a_row * N + a_col];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;

        if (b_row < N && b_col < K)
            tileB[threadIdx.y][threadIdx.x] = B[b_row * K + b_col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int i = 0; i < BLOCK_SIZE; i++)
        {
            value += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < K)
    {
        C[row * K + col] = value;
    }
}


void matrixMulCPU(float* A, float* B, float* C, int M, int N, int K)
{
    for (int row = 0; row < M; row++)
    {
        for (int col = 0; col < K; col++)
        {
            float sum = 0.0f;
            for (int i = 0; i < N; i++)
            {
                sum += A[row * N + i] * B[i * K + col];
            }
            C[row * K + col] = sum;
        }
    }
}


/**
 * CPU matrix multiplication took 8169.75022 milliseconds
 * Host to device memory transfer took 2.64373 milliseconds
 * GPU kernel execution took 86.37453 milliseconds
 * Device to host memory transfer took 6.93015 milliseconds
 * Result verification passed!
 */
int main()
{
    int M = 1024, N = 1024, K = 1024;
    size_t sizeA = M * N * sizeof(float);
    size_t sizeB = N * K * sizeof(float);
    size_t sizeC = M * K * sizeof(float);

    float* h_A = (float*)malloc(sizeA);
    float* h_B = (float*)malloc(sizeB);
    float* h_C = (float*)malloc(sizeC);
    float* h_C_ref = (float*)malloc(sizeC);

    for (int i = 0; i < M * N; i++)
        h_A[i] = rand() / (float)RAND_MAX;
    for (int i = 0; i < N * K; i++)
        h_B[i] = rand() / (float)RAND_MAX;

    auto start_cpu = std::chrono::high_resolution_clock::now();
    matrixMulCPU(h_A, h_B, h_C_ref, M, N, K);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = end_cpu - start_cpu;
    printf("CPU matrix multiplication took %.5f milliseconds\n", cpu_duration.count());

    float* d_A, * d_B, * d_C;
    CHECK(cudaMalloc(&d_A, sizeA));
    CHECK(cudaMalloc(&d_B, sizeB));
    CHECK(cudaMalloc(&d_C, sizeC));

    auto host_to_device_start = std::chrono::high_resolution_clock::now();
    CHECK(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));
    auto host_to_device_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> host_to_device_duration = host_to_device_end - host_to_device_start;
    printf("Host to device memory transfer took %.5f milliseconds\n", host_to_device_duration.count());

    dim3 blockSize(16, 16);
    dim3 gridSize((K + blockSize.x - 1) / blockSize.x,
                  (M + blockSize.y - 1) / blockSize.y);
    auto kernel_start = std::chrono::high_resolution_clock::now();
    matrixMulShared<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);
    CHECK(cudaGetLastError());
    auto kernel_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> kernel_duration = kernel_end - kernel_start;
    printf("GPU kernel execution took %.5f milliseconds\n", kernel_duration.count());

    auto device_to_host_start = std::chrono::high_resolution_clock::now();
    CHECK(cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost));
    auto device_to_host_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> device_to_host_duration = device_to_host_end - device_to_host_start;
    printf("Device to host memory transfer took %.5f milliseconds\n", device_to_host_duration.count());

    // Verify results
    for (int i = 0; i < M * K; i++)
    {
        if (abs(h_C[i] - h_C_ref[i]) > 1e-4)
        {
            fprintf(stderr, "Result verification failed at element %d: %f != %f\n",
                    i, h_C[i], h_C_ref[i]);
            exit(1);
        }
    }
    printf("Result verification passed!\n");

    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));
    return 0;
}