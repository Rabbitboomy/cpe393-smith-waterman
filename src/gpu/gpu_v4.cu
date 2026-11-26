// gpu_v4.cu — owner: Puwanut.
// v3 (or best available kernel) + streams: split the batch into chunks and
// overlap chunk k+1's H2D copy with chunk k's kernel (pinned host memory,
// cudaMemcpyAsync, 2-4 streams). Write your own pipeline here instead of
// run_simple(). Timing: total_ms = first H2D -> last D2H; kernel_ms = sum of
// per-chunk kernel times (document this in the report). The host pipeline
// can be built now against the v0 kernel and swapped to v3 later.
#include "sw_gpu.h"

namespace {

__global__ void sw_kernel_v4(DeviceBatch d) {
    const int pair = blockIdx.x;
    if (pair >= d.n) return;
    // TODO(Puwanut): implement. Placeholder writes -1 so verify.py fails loudly.
    if (threadIdx.x == 0) d.scores[pair] = -1;
}

}  // namespace

GpuResult sw_gpu_v4(const sw::PackedBatch& hb, int* scores_host) {
    return run_simple(hb, scores_host, [](const DeviceBatch& d) {
        const int threads = 128;  // tune; one block per pair
        sw_kernel_v4<<<d.n, threads>>>(d);
    });
}
