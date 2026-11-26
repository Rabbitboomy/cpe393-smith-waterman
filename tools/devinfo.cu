// devinfo.cu — prints the GPU/driver/runtime facts the report must state.
//   devinfo          human-readable summary of every visible GPU
//   devinfo --name   just the name of GPU 0 (bench script uses this as a CSV column)
#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>

int main(int argc, char** argv) {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
        std::fprintf(stderr, "no CUDA device visible (driver missing or GPU not attached)\n");
        return 1;
    }
    cudaDeviceProp p;
    if (argc > 1 && std::strcmp(argv[1], "--name") == 0) {
        cudaGetDeviceProperties(&p, 0);
        std::printf("%s\n", p.name);
        return 0;
    }
    int drv = 0, rt = 0;
    cudaDriverGetVersion(&drv);
    cudaRuntimeGetVersion(&rt);
    std::printf("CUDA driver API : %d.%d\n", drv / 1000, (drv % 1000) / 10);
    std::printf("CUDA runtime    : %d.%d\n", rt / 1000, (rt % 1000) / 10);
    for (int i = 0; i < n; ++i) {
        cudaGetDeviceProperties(&p, i);
        std::printf("\nGPU %d           : %s\n", i, p.name);
        std::printf("compute cap.    : %d.%d  (build needs sm_%d%d)\n", p.major, p.minor, p.major, p.minor);
        std::printf("SMs             : %d\n", p.multiProcessorCount);
        std::printf("global memory   : %.2f GiB\n", p.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        std::printf("shared mem/block: %zu KiB (opt-in max %zu KiB)\n", p.sharedMemPerBlock / 1024,
                    p.sharedMemPerBlockOptin / 1024);
        std::printf("constant memory : %zu KiB\n", p.totalConstMem / 1024);
        std::printf("warp size       : %d\n", p.warpSize);
        std::printf("async engines   : %d (copy/compute overlap for v4 needs >= 1)\n", p.asyncEngineCount);
    }
    return 0;
}
