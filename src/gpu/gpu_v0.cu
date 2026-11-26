// gpu_v0.cu — owner: Titiwut.
// Naive anti-diagonal kernel, global memory only. This is the "before" in every
// before/after comparison: keep it simple and do not optimize it quietly.
//
// Mapping:
//   - one block per pair (blockIdx.x = pair index)
//   - the block sweeps the anti-diagonals k = i + j = 2 .. n + m in order;
//     cells on one anti-diagonal are independent, so the block's threads split
//     them (thread t takes rows i = i_lo + t, i_lo + t + blockDim.x, ...)
//   - __syncthreads() between diagonals: diagonal k needs k-1 and k-2 finished
//
// Memory: cell (i, j) of diagonal k only reads
//   H(i-1, j-1) on diagonal k-2, H(i-1, j) and H(i, j-1) on diagonal k-1,
// so three rotating diagonal buffers (indexed by row i) replace the full
// n x m matrix. In v0 they live in GLOBAL memory (v1 moves them to shared).
// Row 0 / column 0 of the matrix are all zeros; instead of storing them we
// test i > 1 / j > 1, so the buffers never need initialising.
#include "sw_gpu.h"

namespace {

constexpr int kThreads = 128;  // threads per block (one block per pair)

// Global scratch: 3 diagonals x (max_a + 1) ints per pair. Allocated once and
// reused, so cudaMalloc happens in the untimed warm-up run, not in timed runs.
int* g_scratch = nullptr;
size_t g_scratch_ints = 0;

int* scratch_for(const DeviceBatch& d) {
    const size_t need = size_t(d.n) * 3 * size_t(d.max_a + 1);
    if (need > g_scratch_ints) {
        if (g_scratch) CUDA_CHECK(cudaFree(g_scratch));
        CUDA_CHECK(cudaMalloc(&g_scratch, need * sizeof(int)));
        g_scratch_ints = need;
    }
    return g_scratch;
}

__global__ void sw_kernel_v0(DeviceBatch d, int* scratch) {
    const int pair = blockIdx.x;
    const int n = d.a_len[pair];  // rows    (sequence a)
    const int m = d.b_len[pair];  // columns (sequence b)
    const uint8_t* a = d.a + d.a_off[pair];
    const uint8_t* b = d.b + d.b_off[pair];

    // This pair's three diagonal buffers, each indexed by row i (1..n).
    const int stride = d.max_a + 1;
    int* base = scratch + size_t(pair) * 3 * stride;
    int* prev2 = base;               // diagonal k-2
    int* prev1 = base + stride;      // diagonal k-1
    int* cur = base + 2 * stride;    // diagonal k (being written)

    if (threadIdx.x == 0) d.scores[pair] = 0;
    __syncthreads();

    int best = 0;  // this thread's best cell so far
    for (int k = 2; k <= n + m; ++k) {
        // Rows that exist on diagonal k: 1 <= i <= n and 1 <= j = k - i <= m.
        const int i_lo = max(1, k - m);
        const int i_hi = min(n, k - 1);
        for (int i = i_lo + threadIdx.x; i <= i_hi; i += blockDim.x) {
            const int j = k - i;
            const int s = (a[i - 1] == b[j - 1]) ? sw::MATCH : sw::MISMATCH;
            const int h_diag = (i > 1 && j > 1) ? prev2[i - 1] : 0;  // H(i-1, j-1)
            const int h_up = (i > 1) ? prev1[i - 1] : 0;             // H(i-1, j)
            const int h_left = (j > 1) ? prev1[i] : 0;               // H(i, j-1)
            int h = max(0, h_diag + s);
            h = max(h, h_up + sw::GAP);
            h = max(h, h_left + sw::GAP);
            cur[i] = h;
            best = max(best, h);
        }
        __syncthreads();  // diagonal k complete before anyone starts k+1
        int* t = prev2;   // rotate: k-1 -> k-2, k -> k-1, old k-2 is reused
        prev2 = prev1;
        prev1 = cur;
        cur = t;
    }
    atomicMax(&d.scores[pair], best);  // naive block-wide max
}

}  // namespace

GpuResult sw_gpu_v0(const sw::PackedBatch& hb, int* scores_host) {
    return run_simple(hb, scores_host, [](const DeviceBatch& d) {
        int* scratch = scratch_for(d);
        sw_kernel_v0<<<d.n, kThreads>>>(d, scratch);
    });
}