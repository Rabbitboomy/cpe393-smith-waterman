# GPU Acceleration of Smith-Waterman Sequence Alignment

CPE 393 GPU Programming final project — team **WaWaWeeWa**

| Member | ID | Contribution |
|---|---|---|
| Hein Min Thu | 66070503460 | CPU baseline + traceback, input generator, golden outputs, stage two (tiled long pair) |
| Titiwut Phokaeo | 66070503428 | CUDA kernels v0 (naive), v1 (shared memory), v2 (constant memory); GPU driver; README merge |
| Puwanut Theeranuluk | 66070503477 | Kernels v3 (warp shuffle), v4 (streams); benchmark harness, plots, Nsight profiling, report assembly |

> **Team members: start with [TEAM.md](TEAM.md)** — setup, rules, task list and schedule.
>
> **Status: skeleton.** Kernels are placeholders that write `-1`; `verify.py` fails until they are implemented.
> Sections marked _TODO_ are filled in as the project progresses.

## Goal

Compute the best Smith-Waterman local-alignment score for DNA sequence pairs on the GPU, producing
exactly the same integer scores as a single-core CPU reference, and measure the speedup.
Scoring: match **+2**, mismatch **−1**, linear gap **−2**. Traceback stays on the CPU.

## Repository layout

```
include/sw_common.h   shared contract: scoring, I/O formats, CLI, timing CSV   (team)
src/cpu/              CPU reference sw_cpu                                     (Hein)
src/gpu/              GPU driver sw_gpu + one file per kernel version v0..v4   (Titiwut, Puwanut)
tools/devinfo.cu      prints GPU / driver / CUDA versions for the report
scripts/              gen_seqs.py (inputs), verify.py (correctness)
bench/                benchmark + plotting scripts                             (Puwanut)
data/                 smoke input + golden outputs (large inputs are regenerated, not committed)
results/              benchmark CSVs, plots, Nsight reports
report/               IEEE report (Overleaf mirror / PDF)
```

## Requirements

- NVIDIA GPU + driver, **CUDA Toolkit ≥ 12.8** (needed for RTX 50xx / sm_120; older toolkits build for T4/L4/RTX 30–40 only)
- CMake ≥ 3.24 and a C++17 compiler (Windows: Visual Studio 2022 "Desktop development with C++"; Linux: g++)
- Python ≥ 3.8 (standard library only for the core scripts)

The default build targets sm_75 (T4), sm_86 (RTX 30xx), sm_89 (L4 / RTX 40xx) and sm_120 (RTX 50xx), so one
binary runs on every GPU the team uses. Override with `-DSW_CUDA_ARCHS=native` (CMake) if needed.

## Build

**Windows** — open **x64 Native Tools Command Prompt for VS 2022** (from the Start menu; it puts the
64-bit MSVC compiler on PATH), then:
```bat
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
build\bin\devinfo.exe
```
Needs Visual Studio 2022 or VS Build Tools 2022 with "Desktop development with C++", CUDA ≥ 12.8, CMake and Ninja
(`winget install Kitware.CMake Ninja-build.Ninja`). Ninja avoids depending on CUDA's Visual Studio integration.

**Linux:**
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
# or simply: make
./build/bin/devinfo
```

**Google Colab** (Runtime → Change runtime type → T4/L4 GPU):
```bash
!nvidia-smi && nvcc --version
!git clone <REPO_URL> sw-cuda
%cd sw-cuda
!make && make smoke
```

## Run

Every binary takes the same arguments:
```bash
./build/bin/sw_cpu --in data/smoke.txt --out cpu_scores.txt --repeat 5 --header
./build/bin/sw_gpu --in data/smoke.txt --out gpu_scores.txt --repeat 5 --version v2 --header
```

- **Input** (`--in`): one pair per line, tab-separated `pair_id<TAB>seq_a<TAB>seq_b`; alphabet ACGTN; `#` lines ignored.
- **Output** (`--out`): one line per pair, input order, `pair_id<TAB>score`.
- **Timing** (stderr, CSV, one line per run): `version,n_pairs,seq_len,repeat_idx,kernel_ms,total_ms`.
  `kernel_ms` excludes host↔device transfers; `total_ms` = H2D + kernel + D2H (device allocation excluded).
  `sw_gpu` does one untimed warm-up run before the timed repeats.

## Verify correctness

Scores are integers, so GPU output must equal the CPU reference exactly (no tolerance).
```bash
python scripts/verify.py --in data/smoke.txt --golden data/smoke.golden.txt     # all versions
python scripts/verify.py --in data/generated/big.txt                             # runs sw_cpu as reference
```

## Generate inputs

```bash
python scripts/gen_seqs.py --pairs 64 --len 256 --seed 1 --out data/smoke.txt   # the committed smoke set
```
`data/smoke.golden.txt` was produced by `sw_cpu` from that file. _TODO (Hein): full input grid with seeds, NCBI instructions._

## Reproduce the benchmarks

_TODO (Puwanut): one command that regenerates inputs, runs the sweep, writes `results/*.csv` and the plots._

## Benchmark hardware

_TODO: fill from `devinfo` on the headline machine (RTX 5060 Laptop GPU), plus CPU model, OS, driver, CUDA version, compiler flags._

## Team workflow

- One branch per person (`hein`, `titiwut`, `puwanut`), merged into `main` by pull request.
- Before opening a PR: `make smoke` (or `verify.py`) passes for the versions you touched.
- Do not edit `include/sw_common.h` without telling the team: every binary depends on it.
- The submitted commit is tagged `v1.0`.

## Credits

_TODO: external code, libraries, datasets (e.g. NCBI sequences)._
