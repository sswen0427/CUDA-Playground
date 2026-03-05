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


__global__ void matrixTranspose(float* input, float* output, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int inputIndex = y * width + x;
        int outputIndex = x * height + y;
        output[outputIndex] = input[inputIndex];
    }
}

void matrixTransposeCPU(float* input, float* output, int width, int height)
{
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int inputIndex = y * width + x;
            int outputIndex = x * height + y;
            output[outputIndex] = input[inputIndex];
        }
    }
}

/**
 * CPU matrix multiplication took 19.65142 milliseconds
 * Host to device memory transfer took 1.05203 milliseconds
 * GPU kernel execution took 0.23781 milliseconds
 * Device to host memory transfer took 2.91458 milliseconds
 * Result verification passed!
 */

int main()
{
    int M = 1024, N = 1024;
    size_t size = M * N * sizeof(float);

    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);

    for (int i = 0; i < M * N; i++)
        h_A[i] = rand() / (float)RAND_MAX;

    auto start_cpu = std::chrono::high_resolution_clock::now();
    matrixTransposeCPU(h_A, h_B, N, M);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = end_cpu - start_cpu;
    printf("CPU matrix multiplication took %.5f milliseconds\n", cpu_duration.count());


    float* d_A, * d_B;
    CHECK(cudaMalloc(&d_A, size)); 
    CHECK(cudaMalloc(&d_B, size));

    auto host_to_device_start = std::chrono::high_resolution_clock::now();
    CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    auto host_to_device_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> host_to_device_duration = host_to_device_end - host_to_device_start;
    printf("Host to device memory transfer took %.5f milliseconds\n", host_to_device_duration.count());

    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x,
                  (M + blockSize.y - 1) / blockSize.y);
    auto kernel_start = std::chrono::high_resolution_clock::now();
    matrixTranspose<<<gridSize, blockSize>>>(d_A, d_B, N, M);
    cudaDeviceSynchronize();
    CHECK(cudaGetLastError());
    auto kernel_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> kernel_duration = kernel_end - kernel_start;
    printf("GPU kernel execution took %.5f milliseconds\n", kernel_duration.count());

    auto device_to_host_start = std::chrono::high_resolution_clock::now();
    CHECK(cudaMemcpy(h_C, d_B, size, cudaMemcpyDeviceToHost));
    auto device_to_host_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> device_to_host_duration = device_to_host_end - device_to_host_start;
    printf("Device to host memory transfer took %.5f milliseconds\n", device_to_host_duration.count());

    for (int i = 0; i < M * N; i++)
    {
        if (fabs(h_C[i] - h_B[i]) > 1e-5)
        {
            printf("Result verification failed at index %d: %f != %f\n", i, h_C[i], h_B[i]);
            exit(1);
        }
    }
    printf("Result verification passed!\n");

    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    free(h_A);
    free(h_B);
    free(h_C);
    return 0;
}