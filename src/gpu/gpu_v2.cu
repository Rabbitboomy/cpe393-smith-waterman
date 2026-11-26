// gpu_v2.cu — owner: Titiwut.
// v1 + constant memory: the 5x5 scoring table (i==j ? MATCH : MISMATCH,
// see sw_common.h) goes in __constant__ memory, set with cudaMemcpyToSymbol.
#include "sw_gpu.h"

namespace {

__global__ void sw_kernel_v2(DeviceBatch d) {
    const int pair = blockIdx.x;
    if (pair >= d.n) return;
    // TODO(Titiwut): implement. Placeholder writes -1 so verify.py fails loudly.
    if (threadIdx.x == 0) d.scores[pair] = -1;
}

}  // namespace

GpuResult sw_gpu_v2(const sw::PackedBatch& hb, int* scores_host) {
    return run_simple(hb, scores_host, [](const DeviceBatch& d) {
        const int threads = 128;  // tune; one block per pair
        sw_kernel_v2<<<d.n, threads>>>(d);
    });
}
