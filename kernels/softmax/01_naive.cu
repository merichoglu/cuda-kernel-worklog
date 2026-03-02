// kernels/softmax/01_naive.cu

#include <cuda_runtime.h>
#include <numeric>
#include <cmath>
#include "common.h"

#define BLOCK_SIZE 256

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

}

int main(void) {

}