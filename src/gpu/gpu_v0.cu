// gpu_v0.cu — owner: Titiwut.
// Naive anti-diagonal kernel, global memory only. This is the "before" in every
// before/after comparison: keep it simple and do not optimize it quietly.
// Plan: one block per pair; the block sweeps anti-diagonals d = 2 .. len_a+len_b,
// each thread computes cells (i, d-i) of the current diagonal, __syncthreads()
// between diagonals. Keep the last two diagonals in a global scratch buffer
// (3 rotating diagonals: d-2, d-1, d), never the full matrix. Block-wide max
// reduction at the end -> d.scores[blockIdx.x].
#include "sw_gpu.h"

namespace {

__global__ void sw_kernel_v0(DeviceBatch d) {
    const int pair = blockIdx.x;
    if (pair >= d.n) return;
    // TODO(Titiwut): implement. Placeholder writes -1 so verify.py fails loudly.
    if (threadIdx.x == 0) d.scores[pair] = -1;
}

}  // namespace

GpuResult sw_gpu_v0(const sw::PackedBatch& hb, int* scores_host) {
    return run_simple(hb, scores_host, [](const DeviceBatch& d) {
        const int threads = 128;  // tune; one block per pair
        sw_kernel_v0<<<d.n, threads>>>(d);
    });
}
