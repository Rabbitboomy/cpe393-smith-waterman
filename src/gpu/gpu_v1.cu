// gpu_v1.cu — owner: Titiwut.
// v0 + SHARED MEMORY. The only change from v0: the three rotating anti-diagonal
// buffers live in the block's shared memory instead of a global scratch array.
// Everything else (mapping, loop, boundaries, atomicMax) is identical, so the
// v0 -> v1 difference measures the shared-memory optimization alone.
//
// Why it helps: every cell reads 3 neighbours and writes 1 value. In v0 those
// are global-memory accesses (hundreds of cycles if they miss in cache); in v1
// they are on-chip shared-memory accesses (~tens of cycles). Consecutive threads
// touch consecutive rows i, i.e. consecutive 4-byte words -> no bank conflicts.
//
// Cost / limit: 3 * (max_a + 1) * 4 bytes of shared memory per block.
//   max_a = 1000 -> ~12 KB, 2048 -> ~24 KB. Above 48 KB we opt in to the larger
//   limit (RTX 5060: 99 KB -> max_a ~ 8400). Longer sequences belong to stage two.
#include "sw_gpu.h"

namespace {

constexpr int kThreads = 128;  // keep equal to v0 for a fair comparison

__global__ void sw_kernel_v1(DeviceBatch d) {
    extern __shared__ int smem[];  // 3 * (max_a + 1) ints, sized at launch

    const int pair = blockIdx.x;
    const int n = d.a_len[pair];
    const int m = d.b_len[pair];
    const uint8_t* a = d.a + d.a_off[pair];
    const uint8_t* b = d.b + d.b_off[pair];

    const int stride = d.max_a + 1;
    int* prev2 = smem;               // diagonal k-2
    int* prev1 = smem + stride;      // diagonal k-1
    int* cur = smem + 2 * stride;    // diagonal k

    if (threadIdx.x == 0) d.scores[pair] = 0;
    __syncthreads();

    int best = 0;
    for (int k = 2; k <= n + m; ++k) {
        const int i_lo = max(1, k - m);
        const int i_hi = min(n, k - 1);
        for (int i = i_lo + threadIdx.x; i <= i_hi; i += blockDim.x) {
            const int j = k - i;
            const int s = (a[i - 1] == b[j - 1]) ? sw::MATCH : sw::MISMATCH;
            const int h_diag = (i > 1 && j > 1) ? prev2[i - 1] : 0;
            const int h_up = (i > 1) ? prev1[i - 1] : 0;
            const int h_left = (j > 1) ? prev1[i] : 0;
            int h = max(0, h_diag + s);
            h = max(h, h_up + sw::GAP);
            h = max(h, h_left + sw::GAP);
            cur[i] = h;
            best = max(best, h);
        }
        __syncthreads();
        int* t = prev2;
        prev2 = prev1;
        prev1 = cur;
        cur = t;
    }
    atomicMax(&d.scores[pair], best);
}

// Allow more than the default 48 KB of dynamic shared memory when needed.
// Done once (in the untimed warm-up run), not on every timed call.
void configure_smem(size_t bytes) {
    static size_t configured = 48 * 1024;
    static int optin_max = -1;
    if (bytes <= configured) return;
    if (optin_max < 0) {
        int dev = 0;
        CUDA_CHECK(cudaGetDevice(&dev));
        CUDA_CHECK(cudaDeviceGetAttribute(&optin_max, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    }
    if (bytes > size_t(optin_max))
        sw::die("v1: sequence too long for shared memory (" + std::to_string(bytes) + " B needed, " +
                std::to_string(optin_max) + " B available); use stage two for long pairs");
    CUDA_CHECK(cudaFuncSetAttribute(sw_kernel_v1, cudaFuncAttributeMaxDynamicSharedMemorySize, int(bytes)));
    configured = bytes;
}

}  // namespace

GpuResult sw_gpu_v1(const sw::PackedBatch& hb, int* scores_host) {
    return run_simple(hb, scores_host, [](const DeviceBatch& d) {
        const size_t smem_bytes = size_t(3) * (d.max_a + 1) * sizeof(int);
        configure_smem(smem_bytes);
        sw_kernel_v1<<<d.n, kThreads, smem_bytes>>>(d);
    });
}