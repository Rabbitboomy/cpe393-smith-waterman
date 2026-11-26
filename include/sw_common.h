// sw_common.h — the shared contract between sw_cpu, sw_gpu and the scripts.
// Owner: whole team. Change this file only after the team agrees: every
// binary, kernel and script depends on it.
//
// Header-only so it compiles under both g++/MSVC and nvcc (host code only).
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace sw {

// ---------------------------------------------------------------------------
// Scoring parameters — fixed and identical everywhere (linear gap penalty).
// Affine gaps, if ever added, go behind a separate flag; never change these.
// ---------------------------------------------------------------------------
constexpr int MATCH    =  2;
constexpr int MISMATCH = -1;
constexpr int GAP      = -2;

// Alphabet: input is DNA, A/C/G/T/N (lowercase accepted, converted to upper).
// Score rule on characters: same letter -> MATCH, different -> MISMATCH.
// N vs N counts as a match, so the rule is purely "a == b".
// GPU code works on 0..4 codes (A=0 C=1 G=2 T=3 N=4); the same rule applies
// (code_a == code_b), so a 5x5 table built as i==j ? MATCH : MISMATCH is exact.
constexpr int ALPHABET = 5;

inline int encode(char c) {
    switch (c) {
        case 'A': return 0; case 'C': return 1; case 'G': return 2;
        case 'T': return 3; case 'N': return 4;
        default:  return -1;
    }
}

inline int score(char a, char b) { return a == b ? MATCH : MISMATCH; }

// ---------------------------------------------------------------------------
// Input: one pair per line, tab-separated:  pair_id <TAB> seq_a <TAB> seq_b
// Lines that are empty or start with '#' are ignored.
// ---------------------------------------------------------------------------
struct Pair {
    std::string id, a, b;
};

[[noreturn]] inline void die(const std::string& msg) {
    std::fprintf(stderr, "error: %s\n", msg.c_str());
    std::exit(1);
}

inline void normalize_seq(std::string& s, const std::string& where) {
    for (char& c : s) {
        if (c >= 'a' && c <= 'z') c = char(c - 'a' + 'A');
        if (encode(c) < 0) die(where + ": invalid character '" + std::string(1, c) + "' (allowed: ACGTN)");
    }
    if (s.empty()) die(where + ": empty sequence");
}

inline std::vector<Pair> read_pairs(const std::string& path) {
    std::ifstream f(path);
    if (!f) die("cannot open input file " + path);
    std::vector<Pair> pairs;
    std::string line;
    size_t lineno = 0;
    while (std::getline(f, line)) {
        ++lineno;
        if (!line.empty() && line.back() == '\r') line.pop_back();  // Windows line endings
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        Pair p;
        if (!std::getline(ss, p.id, '\t') || !std::getline(ss, p.a, '\t') || !std::getline(ss, p.b, '\t'))
            die(path + ":" + std::to_string(lineno) + ": expected 3 tab-separated fields");
        const std::string where = path + ":" + std::to_string(lineno);
        normalize_seq(p.a, where);
        normalize_seq(p.b, where);
        pairs.push_back(std::move(p));
    }
    if (pairs.empty()) die("no pairs in " + path);
    return pairs;
}

// ---------------------------------------------------------------------------
// Output: one line per pair, input order:  pair_id <TAB> score
// Scores are integers, so CPU and GPU files must be byte-identical.
// ---------------------------------------------------------------------------
inline void write_scores(const std::string& path, const std::vector<Pair>& pairs,
                         const std::vector<int>& scores) {
    std::ofstream f(path, std::ios::binary);  // binary: '\n' endings on Windows too
    if (!f) die("cannot open output file " + path);
    for (size_t i = 0; i < pairs.size(); ++i) f << pairs[i].id << '\t' << scores[i] << '\n';
}

// ---------------------------------------------------------------------------
// Packed batch: every pair concatenated into two flat code arrays plus
// offset/length arrays. This is the layout copied to the GPU, shared by all
// kernel versions v0..v4.
// ---------------------------------------------------------------------------
struct PackedBatch {
    std::vector<uint8_t> a, b;              // codes 0..4, all pairs back to back
    std::vector<int> a_off, a_len, b_off, b_len;
    int max_a = 0, max_b = 0;
    size_t n() const { return a_off.size(); }
};

inline PackedBatch pack(const std::vector<Pair>& pairs) {
    PackedBatch pb;
    for (const Pair& p : pairs) {
        pb.a_off.push_back(int(pb.a.size()));
        pb.a_len.push_back(int(p.a.size()));
        for (char c : p.a) pb.a.push_back(uint8_t(encode(c)));
        pb.b_off.push_back(int(pb.b.size()));
        pb.b_len.push_back(int(p.b.size()));
        for (char c : p.b) pb.b.push_back(uint8_t(encode(c)));
        pb.max_a = std::max(pb.max_a, int(p.a.size()));
        pb.max_b = std::max(pb.max_b, int(p.b.size()));
    }
    return pb;
}

// seq_len column in the timing CSV = longest sequence in the batch.
inline int max_seq_len(const std::vector<Pair>& pairs) {
    size_t m = 0;
    for (const Pair& p : pairs) m = std::max({m, p.a.size(), p.b.size()});
    return int(m);
}

// ---------------------------------------------------------------------------
// Command line, identical for every binary:
//   sw_cpu --in pairs.txt --out scores.txt --repeat 5
//   sw_gpu --in pairs.txt --out scores.txt --repeat 5 --version v3
// ---------------------------------------------------------------------------
struct Args {
    std::string in, out;
    int repeat = 5;
    std::string version = "v0";  // sw_gpu only
    bool header = false;         // print the timing CSV header line first
};

inline void usage(const char* prog, bool gpu) {
    std::fprintf(stderr,
                 "usage: %s --in PAIRS.txt --out SCORES.txt [--repeat N]%s [--header]\n",
                 prog, gpu ? " [--version v0|v1|v2|v3|v4]" : "");
    std::exit(2);
}

inline Args parse_args(int argc, char** argv, bool gpu) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto val = [&]() -> std::string {
            if (i + 1 >= argc) usage(argv[0], gpu);
            return argv[++i];
        };
        if (k == "--in") a.in = val();
        else if (k == "--out") a.out = val();
        else if (k == "--repeat") a.repeat = std::atoi(val().c_str());
        else if (k == "--version" && gpu) a.version = val();
        else if (k == "--header") a.header = true;
        else usage(argv[0], gpu);
    }
    if (a.in.empty() || a.out.empty() || a.repeat < 1) usage(argv[0], gpu);
    return a;
}

// ---------------------------------------------------------------------------
// Timing: stderr, CSV, one line per run.
//   version,n_pairs,seq_len,repeat_idx,kernel_ms,total_ms
// kernel_ms excludes host<->device transfers, total_ms includes them
// (H2D + kernel + D2H; device allocation is NOT included in either).
// For sw_cpu both columns are the same compute time.
// ---------------------------------------------------------------------------
inline void timing_header() {
    std::fprintf(stderr, "version,n_pairs,seq_len,repeat_idx,kernel_ms,total_ms\n");
}

inline void emit_timing(const std::string& version, size_t n_pairs, int seq_len, int repeat_idx,
                        double kernel_ms, double total_ms) {
    std::fprintf(stderr, "%s,%zu,%d,%d,%.4f,%.4f\n", version.c_str(), n_pairs, seq_len,
                 repeat_idx, kernel_ms, total_ms);
}

}  // namespace sw
