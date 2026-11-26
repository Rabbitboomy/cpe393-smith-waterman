// main_cpu.cpp — sw_cpu entry point. Owner: Hein.
#include <chrono>
#include <vector>

#include "cpu_sw.h"
#include "sw_common.h"

int main(int argc, char** argv) {
    const sw::Args args = sw::parse_args(argc, argv, /*gpu=*/false);
    const std::vector<sw::Pair> pairs = sw::read_pairs(args.in);
    const int seq_len = sw::max_seq_len(pairs);
    std::vector<int> scores(pairs.size());

    if (args.header) sw::timing_header();
    for (int r = 0; r < args.repeat; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        for (size_t i = 0; i < pairs.size(); ++i) scores[i] = sw_score_cpu(pairs[i].a, pairs[i].b);
        const auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        sw::emit_timing("cpu", pairs.size(), seq_len, r, ms, ms);
    }
    sw::write_scores(args.out, pairs, scores);
    return 0;
}
