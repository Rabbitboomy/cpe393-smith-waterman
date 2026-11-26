# bench/ — owner: Puwanut

To build here (nothing in this folder needs a GPU until the Monday sweep):

- `bench.py`: runs `sw_cpu` and every `sw_gpu --version vN` over the input-size grid,
  `--repeat 5` (or more), parses the stderr timing CSV, and writes one CSV to `results/`
  with an extra `gpu` column taken from `build/bin/devinfo --name`. Report the **median**.
  - Timing line format (stderr): `version,n_pairs,seq_len,repeat_idx,kernel_ms,total_ms`
  - Also add a `gcups` column computed from cells / time (cells = sum of len_a * len_b).
  - Inputs come from `scripts/gen_seqs.py` with fixed seeds; record the exact commands.
- `plots.py`: runtime vs input size, speedup vs input size, and a before/after bar chart
  for one optimization (log axes where sizes span decades; units on both axes).
- Nsight: `nsys profile -o results/nsys_v4 ...` and `ncu --set full -o results/ncu_v2 ...`
  on the final versions, run on the laptop (RTX 5060). Commit the `.nsys-rep` / `.ncu-rep` files.
