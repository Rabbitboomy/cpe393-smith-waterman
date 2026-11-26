#pragma once
#include <string>

#include "sw_common.h"

// Best local-alignment score of a vs b (match +2, mismatch -1, gap -2).
int sw_score_cpu(const std::string& a, const std::string& b);
