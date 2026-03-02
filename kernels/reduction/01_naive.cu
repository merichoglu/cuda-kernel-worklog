// kernels/reduction/01_naive.cu

#include <cuda_runtime.h>
#include <numeric>
#include <cmath>
#include "common.h"

#define BLOCK_SIZE 256

/*
    CPU Reference for correctness
*/
float cpu_sum(float *arr, int n) {
    return std::accumulate(arr, arr + n, 0.0f);
}

/*
    GPU Kernel
*/
__global__ void reduce_naive(float *input, float *output, int n) {
    /* index computation */
    int thread_id = threadIdx.x;
    int global_id = blockIdx.x * blockDim.x + thread_id;

    __shared__ float shared_data[BLOCK_SIZE];
    shared_data[thread_id] = (global_id < n) ? input[global_id] : 0.0f;
    __syncthreads();

    int stride = BLOCK_SIZE / 2;
    while (stride > 0) {
        if (thread_id < stride) {
            shared_data[thread_id] += shared_data[thread_id + stride];
        }
        __syncthreads();
        stride /= 2;
    }

    /*  main thread to write */
    if (thread_id == 0) {
        output[blockIdx.x] = shared_data[0];
    }
}

int main(void) {
    int n = 1 << 24;
    size_t bytes = n * sizeof(float);

    /* get the input array and run CPU reference */
    float *h_input = 0;
    h_input = (float*)malloc(bytes);
    if (h_input == 0) {
        printf("Memory allocation on host side failed.");
        return 1;
    }
 
    for (size_t i = 0; i < n; i++) {
        h_input[i] = (float)rand() / RAND_MAX;
    }

    float cpu_result = cpu_sum(h_input, n);

    /* handle device */
    float *d_input = 0;
    CUDA_CHECK(cudaMalloc((void**)&d_input, bytes));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

    /* handle output part */
    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    float *h_output = 0;
    h_output = (float*)malloc(num_blocks * sizeof(float));
    if(h_output == 0) {
        printf("Memory allocation on host side failed.");
        return 1;
    }

    float *d_output = 0;
    CUDA_CHECK(cudaMalloc((void**)&d_output, num_blocks * sizeof(float)));
    /* warmup */
    reduce_naive<<<num_blocks, BLOCK_SIZE>>>(d_input, d_output, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    /* benchmark */
    GPU_TIMER timer;
    int iters = 100;
    start(timer);
    for (int i = 0; i < iters; i++) {
        reduce_naive<<<num_blocks, BLOCK_SIZE>>>(d_input, d_output, n);
    }
    float ms = stop(timer);
    float avg_ms = ms / iters;

    /* copy output back and get final sum */
    CUDA_CHECK(cudaMemcpy(h_output, d_output, num_blocks * sizeof(float), cudaMemcpyDeviceToHost));
    float gpu_result = cpu_sum(h_output, num_blocks);

    /* correctness check */
    float diff = fabs(gpu_result - cpu_result) / fabs(cpu_result);
    if (diff > 1e-3f) {
        printf("MISMATCH: cpu=%.6f gpu=%.6f diff=%.2e\n", cpu_result, gpu_result, diff);
    } else {
        printf("OK: cpu=%.6f gpu=%.6f\n", cpu_result, gpu_result);
    }

    /* benchmark output */
    size_t bytes_moved = (size_t)n * sizeof(float);  // read once
    size_t flops       = (size_t)n;                  // n-1 adds ~ n
    print_bench("01_naive", bytes_moved, flops, avg_ms);

    /* cleanup */
    free(h_input);
    free(h_output);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return 0;
}

