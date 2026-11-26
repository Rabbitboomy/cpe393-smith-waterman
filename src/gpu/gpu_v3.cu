// gpu_v3.cu — owner: Puwanut.
// v2 + warp shuffle: neighbouring threads exchange H values with
// __shfl_up_sync instead of going through shared memory. Starts from gpu_v2.cu
// once v2 is merged; until then prototype on a copy of v0/v1.
#include "sw_gpu.h"

namespace {

__global__ void sw_kernel_v3(DeviceBatch d) {
    const int pair = blockIdx.x;
    if (pair >= d.n) return;
    // TODO(Puwanut): implement. Placeholder writes -1 so verify.py fails loudly.
    if (threadIdx.x == 0) d.scores[pair] = -1;
}

}  // namespace

GpuResult sw_gpu_v3(const sw::PackedBatch& hb, int* scores_host) {
    return run_simple(hb, scores_host, [](const DeviceBatch& d) {
        const int threads = 128;  // tune; one block per pair
        sw_kernel_v3<<<d.n, threads>>>(d);
    });
}
