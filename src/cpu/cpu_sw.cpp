// cpu_sw.cpp — single-core Smith-Waterman reference (score only).
// Owner: Hein. Every GPU result is checked against this, so it must be right
// before anything else matters.
//
// Recurrence (linear gap):
//   H[i][j] = max(0, H[i-1][j-1] + s(a_i, b_j), H[i-1][j] + GAP, H[i][j-1] + GAP)
//   score   = max over all H[i][j]
// Only the previous row is kept, so memory is O(len_b).

#include "cpu_sw.h"

#include <algorithm>
#include <vector>

int sw_score_cpu(const std::string& a, const std::string& b) {
    const size_t m = b.size();
    std::vector<int> prev(m + 1, 0), cur(m + 1, 0);
    int best = 0;
    for (size_t i = 1; i <= a.size(); ++i) {
        cur[0] = 0;
        const char ai = a[i - 1];
        for (size_t j = 1; j <= m; ++j) {
            int h = prev[j - 1] + sw::score(ai, b[j - 1]);  // diagonal
            h = std::max(h, prev[j] + sw::GAP);              // up
            h = std::max(h, cur[j - 1] + sw::GAP);           // left
            h = std::max(h, 0);
            cur[j] = h;
            best = std::max(best, h);
        }
        std::swap(prev, cur);
    }
    return best;
}

// TODO(Hein): CPU traceback from the best-scoring cell, kept in its own file
// (e.g. cpu_traceback.cpp) so the scoring path above stays untouched.
