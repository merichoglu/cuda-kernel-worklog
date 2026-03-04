// kernels/softmax/01_naive.cu

#include <cuda_runtime.h>
#include <numeric>
#include <cmath>
#include "common.h"

#define BLOCK_SIZE 1024

float compute_denom(float *input, int cols) {
    float denom = 0;
    for (int j = 0; j < cols; j++) {
        denom += expf(input[j]);
    }
    return denom;
}

/* CPU reference */
void cpu_softmax(float *input, float *output, int rows, int cols) {
    float denom;
    for (int i = 0; i < rows; i++) {
        denom = compute_denom(input + i * cols, cols);
        for (int j = 0; j < cols; j++) {
            output[i * cols + j] = expf(input[i * cols + j]) / denom;
        }
    }
}

/* GPU kernel, one block per row */
__global__ void softmax_naive(float *input, float *output, int rows, int cols) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    float *row_in = input + row * cols;
    float *row_out = output + row * cols;

    __shared__ float shared_data[BLOCK_SIZE];
    shared_data[tid] = expf(row_in[tid]);
    __syncthreads();

    int stride = BLOCK_SIZE / 2;
    while (stride > 0) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
        stride /= 2;
    }

    row_out[tid] = expf(row_in[tid]) / shared_data[0];
}

int main(void) {
    int rows = 4096;
    int cols = 1024;
    size_t bytes = rows * cols * sizeof(float);

    /* set up the host side io */
    float *h_input = 0;
    h_input = (float*)malloc(bytes);
    if (h_input == 0) {
        printf("Memory allocation on host side failed.");
        return 1;
    }

    float *h_cpu_output = 0;
    h_cpu_output = (float*)malloc(bytes);
    if (h_cpu_output == 0) {
        printf("Memory allocation on host side failed.");
        return 1;
    }

    float *h_gpu_output = 0;
    h_gpu_output = (float*)malloc(bytes);
    if (h_gpu_output == 0) {
        printf("Memory allocation on host side failed.");
        return 1;
    }

    /* input fill */
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            h_input[i * cols + j] = (float)rand() / RAND_MAX;
        }
    }

    /* cpu calculation */
    cpu_softmax(h_input, h_cpu_output, rows, cols);

    /* handle device */
    float *d_input = 0;
    CUDA_CHECK(cudaMalloc((void**)&d_input, bytes));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

    float *d_output = 0;
    CUDA_CHECK(cudaMalloc((void**)&d_output, bytes));

    /* one block per row and 1024 threads per block (one per column) */
    dim3 grid(rows);
    dim3 block(BLOCK_SIZE);

    /* warmup */
    softmax_naive<<<grid, block>>>(d_input, d_output, rows, cols);
    CUDA_CHECK(cudaDeviceSynchronize());

    /* benchmark */
    GPU_TIMER timer;
    int iters = 100;
    start(timer);
    for (int i = 0; i < iters; i++) {
        softmax_naive<<<grid, block>>>(d_input, d_output, rows, cols);
    }
    float ms = stop(timer);
    float avg_ms = ms / iters;

    /* copy back */
    CUDA_CHECK(cudaMemcpy(h_gpu_output, d_output, bytes, cudaMemcpyDeviceToHost));

    /* correctness check: max absolute error across all elements */
    float max_err = 0.0f;
    for (int i = 0; i < rows * cols; i++) {
        float err = fabs(h_gpu_output[i] - h_cpu_output[i]);
        if (err > max_err) max_err = err;
    }
    if (max_err > 1e-4f) {
        printf("MISMATCH: max_err=%.2e\n", max_err);
    } else {
        printf("OK: max_err=%.2e\n", max_err);
    }

    /* benchmark output: kernel reads input twice, writes output once */
    size_t bytes_moved = 3 * bytes;
    size_t flops       = (size_t)rows * cols * 3;  // 2x expf + 1 divide per element
    print_bench("01_naive", bytes_moved, flops, avg_ms);

    /* cleanup */
    free(h_input);
    free(h_cpu_output);
    free(h_gpu_output);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return 0;
}