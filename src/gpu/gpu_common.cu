// gpu_common.cu — owner: Titiwut.
#include "gpu_common.cuh"

DeviceBatch alloc_device(const sw::PackedBatch& hb) {
    DeviceBatch d;
    d.n = int(hb.n());
    d.max_a = hb.max_a;
    d.max_b = hb.max_b;
    CUDA_CHECK(cudaMalloc(&d.a, hb.a.size()));
    CUDA_CHECK(cudaMalloc(&d.b, hb.b.size()));
    CUDA_CHECK(cudaMalloc(&d.a_off, sizeof(int) * d.n));
    CUDA_CHECK(cudaMalloc(&d.a_len, sizeof(int) * d.n));
    CUDA_CHECK(cudaMalloc(&d.b_off, sizeof(int) * d.n));
    CUDA_CHECK(cudaMalloc(&d.b_len, sizeof(int) * d.n));
    CUDA_CHECK(cudaMalloc(&d.scores, sizeof(int) * d.n));
    return d;
}

void copy_to_device(const sw::PackedBatch& hb, DeviceBatch& d, cudaStream_t s) {
    const size_t ni = sizeof(int) * d.n;
    CUDA_CHECK(cudaMemcpyAsync(d.a, hb.a.data(), hb.a.size(), cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(d.b, hb.b.data(), hb.b.size(), cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(d.a_off, hb.a_off.data(), ni, cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(d.a_len, hb.a_len.data(), ni, cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(d.b_off, hb.b_off.data(), ni, cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(d.b_len, hb.b_len.data(), ni, cudaMemcpyHostToDevice, s));
}

void free_device(DeviceBatch& d) {
    cudaFree(d.a);
    cudaFree(d.b);
    cudaFree(d.a_off);
    cudaFree(d.a_len);
    cudaFree(d.b_off);
    cudaFree(d.b_len);
    cudaFree(d.scores);
    d = DeviceBatch{};
}
