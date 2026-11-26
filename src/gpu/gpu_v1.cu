// gpu_v1.cu — owner: Titiwut.
// v0 + shared memory: the three rotating diagonals live in shared memory
// instead of global. Copy of gpu_v0.cu as a starting point once v0 is correct.
#include "sw_gpu.h"

namespace {

__global__ void sw_kernel_v1(DeviceBatch d) {
    const int pair = blockIdx.x;
    if (pair >= d.n) return;
    // TODO(Titiwut): implement. Placeholder writes -1 so verify.py fails loudly.
    if (threadIdx.x == 0) d.scores[pair] = -1;
}

}  // namespace

GpuResult sw_gpu_v1(const sw::PackedBatch& hb, int* scores_host) {
    return run_simple(hb, scores_host, [](const DeviceBatch& d) {
        const int threads = 128;  // tune; one block per pair
        sw_kernel_v1<<<d.n, threads>>>(d);
    });
}
