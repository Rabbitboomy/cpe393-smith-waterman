// main_gpu.cu — sw_gpu entry point: every kernel version runs from this one
// binary via --version, on identical inputs. Owner: Titiwut.
#include <map>
#include <string>
#include <vector>

#include "sw_common.h"
#include "sw_gpu.h"

using VersionFn = GpuResult (*)(const sw::PackedBatch&, int*);

static const std::map<std::string, VersionFn> kVersions = {
    {"v0", sw_gpu_v0}, {"v1", sw_gpu_v1}, {"v2", sw_gpu_v2},
    {"v3", sw_gpu_v3}, {"v4", sw_gpu_v4},
};

int main(int argc, char** argv) {
    const sw::Args args = sw::parse_args(argc, argv, /*gpu=*/true);
    auto it = kVersions.find(args.version);
    if (it == kVersions.end()) sw::die("unknown --version " + args.version + " (expected v0..v4)");

    const std::vector<sw::Pair> pairs = sw::read_pairs(args.in);
    const sw::PackedBatch packed = sw::pack(pairs);
    const int seq_len = sw::max_seq_len(pairs);
    std::vector<int> scores(pairs.size(), 0);

    CUDA_CHECK(cudaFree(nullptr));     // create the CUDA context outside any timing
    it->second(packed, scores.data()); // untimed warm-up run (JIT, caches, clocks)

    if (args.header) sw::timing_header();
    for (int r = 0; r < args.repeat; ++r) {
        const GpuResult res = it->second(packed, scores.data());
        sw::emit_timing(args.version, pairs.size(), seq_len, r, res.kernel_ms, res.total_ms);
    }
    sw::write_scores(args.out, pairs, scores);
    return 0;
}
