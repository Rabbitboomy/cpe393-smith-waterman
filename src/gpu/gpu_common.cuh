// gpu_common.cuh — host-side plumbing shared by every kernel version.
// Owner: Titiwut. Kernel files should not need to touch this.
#pragma once

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#include "sw_common.h"

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, \
                         __LINE__, cudaGetErrorString(err_));                     \
            std::exit(1);                                                         \
        }                                                                         \
    } while (0)

// Device copy of a PackedBatch. All pointers are device pointers.
struct DeviceBatch {
    uint8_t* a = nullptr;
    uint8_t* b = nullptr;
    int* a_off = nullptr;
    int* a_len = nullptr;
    int* b_off = nullptr;
    int* b_len = nullptr;
    int* scores = nullptr;  // one int per pair
    int n = 0, max_a = 0, max_b = 0;
};

// What every version returns. See sw_common.h for the exact definitions:
// kernel_ms excludes transfers, total_ms = H2D + kernel + D2H.
struct GpuResult {
    float kernel_ms = 0.f;
    float total_ms = 0.f;
};

DeviceBatch alloc_device(const sw::PackedBatch& hb);  // allocation only, no copies
void copy_to_device(const sw::PackedBatch& hb, DeviceBatch& d, cudaStream_t s = 0);
void free_device(DeviceBatch& d);

// Standard single-stream run: alloc (untimed) -> H2D -> launch -> D2H -> free.
// `launch(d)` must enqueue the kernel(s) on the default stream.
// v0..v3 use this; v4 (streams) writes its own pipeline instead.
template <class Launch>
GpuResult run_simple(const sw::PackedBatch& hb, int* scores_host, Launch launch) {
    DeviceBatch d = alloc_device(hb);
    cudaEvent_t e0, e1, e2, e3;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventCreate(&e2));
    CUDA_CHECK(cudaEventCreate(&e3));

    CUDA_CHECK(cudaEventRecord(e0));
    copy_to_device(hb, d);
    CUDA_CHECK(cudaEventRecord(e1));
    launch(d);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(e2));
    CUDA_CHECK(cudaMemcpy(scores_host, d.scores, sizeof(int) * d.n, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(e3));
    CUDA_CHECK(cudaEventSynchronize(e3));

    GpuResult r;
    CUDA_CHECK(cudaEventElapsedTime(&r.kernel_ms, e1, e2));
    CUDA_CHECK(cudaEventElapsedTime(&r.total_ms, e0, e3));
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    cudaEventDestroy(e2);
    cudaEventDestroy(e3);
    free_device(d);
    return r;
}
