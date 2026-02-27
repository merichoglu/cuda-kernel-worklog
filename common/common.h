// common/common.h
#pragma once

#include <stdio.h>

/*
    CUDA error-checking macro
*/
#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/* 
    CUDA event-based timer
*/
struct GPU_TIMER {
    cudaEvent_t start_event;
    cudaEvent_t stop_event;

    GPU_TIMER() {
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);
    }

    ~GPU_TIMER(){
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }
};

void start(GPU_TIMER &timer, cudaStream_t stream = 0) {
    cudaEventRecord(timer.start_event, stream);
}

float stop(GPU_TIMER &timer, cudaStream_t stream = 0) {
    float ms = 0;
    cudaEventRecord(timer.stop_event, stream);
    cudaEventSynchronize(timer.stop_event);
    cudaEventElapsedTime(&ms, timer.start_event, timer.stop_event);
    return ms;
}

/*
    CUDA BW / FLOP printer
*/
void print_bench(const char* name, size_t bytes_moved, size_t flops, float ms_per_iter) {
    double seconds = ms_per_iter / 1000.0;

    double gp_per_sec = (bytes_moved / 1e9) / seconds;
    double gflops = (flops / 1e9) / seconds;

    printf("  %-26s | %8.2f GB/s | %9.2f GFLOP/s\n", name, gp_per_sec, gflops);
}



