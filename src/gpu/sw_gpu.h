// sw_gpu.h — one entry per kernel version. Each version lives in its own file
// (gpu_v0.cu ... gpu_v4.cu) so two people never edit the same file.
//
// Contract for every version:
//   - input  : host PackedBatch (codes 0..4, see sw_common.h)
//   - output : scores_host[i] = best SW score of pair i, input order
//   - return : GpuResult{kernel_ms, total_ms}
#pragma once

#include "gpu_common.cuh"

GpuResult sw_gpu_v0(const sw::PackedBatch& hb, int* scores_host);  // naive anti-diagonal, global memory  (Titiwut)
GpuResult sw_gpu_v1(const sw::PackedBatch& hb, int* scores_host);  // + shared memory                     (Titiwut)
GpuResult sw_gpu_v2(const sw::PackedBatch& hb, int* scores_host);  // + constant-memory scoring table     (Titiwut)
GpuResult sw_gpu_v3(const sw::PackedBatch& hb, int* scores_host);  // + warp shuffle                      (Puwanut)
GpuResult sw_gpu_v4(const sw::PackedBatch& hb, int* scores_host);  // + streams, batch pipelining         (Puwanut)
// Stage two (tiled single long pair) gets its own entry here when Hein starts it.
