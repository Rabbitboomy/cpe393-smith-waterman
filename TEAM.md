# Team guide — WaWaWeeWa (read this first)

Due **Wednesday 2 December 2026**. 100 points: 60 repo, 20 IEEE report, 20 presentation + live demo.
Full rationale is in the "Work Split & Schedule" PDF; this page is the short version: how to start, what you own, when it is due.

---

## 1. Get set up (do this tonight, ~15 min)

1. Accept the GitHub invite (check your email), then:
   ```bash
   git clone https://github.com/Rabbitboomy/cpe393-smith-waterman.git
   cd cpe393-smith-waterman
   git checkout <your-branch>        # hein / puwanut
   ```
2. Build — see README → Build. Short version:
   - **Windows**: open *x64 Native Tools Command Prompt for VS 2022* (not plain PowerShell), then
     `cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release` and `cmake --build build`.
     Needs CUDA ≥ 12.8, VS 2022 (or Build Tools) with C++, CMake, Ninja (`winget install Kitware.CMake Ninja-build.Ninja`).
   - **Colab / Linux**: `make`
3. Check it works:
   ```bash
   build/bin/devinfo                     # post the output in the group chat
   python scripts/verify.py --in data/smoke.txt --golden data/smoke.golden.txt
   ```
   Expected right now: **every version FAILs with `got -1`**. That means the GPU runs; kernels are just placeholders.
   A `CUDA error ...` instead means your setup has a problem — post it in the chat.

## 2. Rules (so three people can work in parallel)

- Work only on **your branch**, only in **your files** (see the table below).
- Everything reaches `main` through a **pull request**. Before opening one, `verify.py` must PASS for the versions you touched.
- **Do not change `include/sw_common.h`** (scoring, file formats, CLI, timing CSV) without telling the team first — every binary depends on it.
- Pull others' work regularly: `git pull origin main` while on your branch.
- Every binary takes the same arguments; timing goes to stderr as
  `version,n_pairs,seq_len,repeat_idx,kernel_ms,total_ms` (`kernel_ms` excludes transfers, `total_ms` includes them).
- Scoring is fixed everywhere: match +2, mismatch −1, gap −2 (linear). Scores are integers → CPU and GPU must match **exactly**.

Daily loop:
```bash
# edit your files
cmake --build build                       # or: make
python scripts/verify.py --in data/smoke.txt --golden data/smoke.golden.txt --versions v0
git add <your files>
git commit -m "short description"
git push
# when it works: GitHub → "Compare & pull request" → base: main ← your branch → Merge
```

## 3. Who owns what

| | Hein | Titiwut | Puwanut |
|---|---|---|---|
| Files | `src/cpu/`, `scripts/gen_seqs.py`, `data/`, stage-two kernel file | `src/gpu/` (driver, `gpu_common.*`, `gpu_v0..v2.cu`) | `gpu_v3.cu`, `gpu_v4.cu`, `bench/`, `results/` |
| Report | II Baseline, stage-two part of IV | III CUDA design, IV Optimizations | I Intro, V Setup, VI Results, VII Conclusion + assembly |
| Slides | 1–2 slides on your own work | 1–2 slides on your own work | 1–2 slides on your own work |

Everyone writes the README part for their own code; Titiwut merges it. Nobody presents someone else's section (rubric: 4 pts participation + 4 pts answering about your own work).

### Hein Min Thu — baseline, data, correctness, stage two
- [ ] Review `src/cpu/cpu_sw.cpp` (already a working single-row reference, cross-checked against a full-matrix version) — you own and explain it.
- [ ] CPU traceback from the best-scoring cell, in its **own file** (e.g. `src/cpu/cpu_traceback.cpp`), not mixed into the scoring loop.
- [ ] Input grid: agree sizes with Puwanut, generate with fixed seeds, commit the exact `gen_seqs.py` commands to the README.
      Suggested start: seq length {128, 256, 512, 1024, 2048} × pairs {256, 1024, 4096}.
- [ ] Golden outputs (`sw_cpu`) for every grid input that is small enough to commit; big inputs go in `data/generated/` (ignored) and are regenerated from seeds.
- [ ] Instructions (or a script) for fetching real DNA from NCBI — cut first if short on time.
- [ ] **Stage two**: tiled kernel for one long pair, keeping only boundary row/column between tiles (never the full matrix — 1M×1M does not fit in 8 GB). Start as soon as the CPU items are done.
      **Go/no-go Sunday night**: if it does not give correct scores on a small case by then, it is cut and you move to report + slides.
- [ ] Second-GPU benchmark run on your GPU once Puwanut's `bench.py` exists (comparison column only).

### Titiwut Phokaeo — CUDA kernels v0–v2, GPU driver
- [ ] `gpu_v0.cu`: naive anti-diagonal kernel, global memory only. This is the "before" in every comparison — never skip or quietly optimize it.
- [ ] `gpu_v1.cu`: + shared memory for the active diagonals.
- [ ] `gpu_v2.cu`: + constant memory for the scoring table.
- [ ] Keep all versions runnable from the one `sw_gpu --version vN` binary.
- [ ] Final benchmark sweep + Nsight runs happen on this laptop (RTX 5060, headline numbers).

### Puwanut Theeranuluk — v3, v4, benchmarks, profiling, report assembly
- [ ] **Today**: set up the IEEE template on Overleaf and invite Hein + Titiwut.
- [ ] `bench/bench.py`: runs `sw_cpu` and every `sw_gpu --version` over the grid, `--repeat 5`, parses stderr, writes one CSV to `results/` with a `gpu` column (`build/bin/devinfo --name`) and a GCUPS column. Report the **median**. Can be built now against `sw_cpu` + the placeholder GPU versions.
- [ ] `bench/plots.py`: runtime vs size, speedup vs size, before/after bar chart for one optimization (units on both axes).
- [ ] `gpu_v4.cu` (streams): start **now** — it is host-side pipelining (pinned memory, chunks, 2–4 streams, `cudaMemcpyAsync`) and can wrap the v0 kernel until v3 exists. Note: the RTX 5060 Laptop has **1 copy engine**, so copy/compute overlap works but H2D and D2H cannot overlap each other — explain this in the report.
- [ ] `gpu_v3.cu` (warp shuffle): start from `gpu_v2.cu` once v2 is merged (target Saturday).
- [ ] Nsight Systems + Nsight Compute reports on the final versions, run on Titiwut's laptop, committed to `results/`.

## 4. Schedule

| Day | Hein | Titiwut | Puwanut | Gate |
|---|---|---|---|---|
| **Thu 26** | setup | setup ✅ | setup, Overleaf | everyone builds, verify FAILs with `-1` |
| **Fri 27** | traceback, input grid, golden files | v0 → PASS → PR | bench.py, v4 host pipeline | |
| **Sat 28** | start stage two | v1, v2 → PR | v4 PASS, plots.py, start v3 | **GPU output checked against golden files on all sizes** |
| **Sun 29** | stage two small case | fixes, README | v3 → PR | **Merge freeze** (no new stage-one kernels after today) · **stage-two go/no-go** |
| **Mon 30** | second-GPU sweep | headline sweep + Nsight on laptop | run sweep/Nsight, make plots | measurement only |
| **Tue 1** | report + slides | report + slides | assemble report | walk the submission checklist **in the evening** |
| **Wed 2** | | tag `v1.0`, submit | | due |

If we fall behind, cut in this order: stage two → second-GPU sweep → v3 → NCBI data.
Never cut: v0, README reproduction steps, repeated timings.
