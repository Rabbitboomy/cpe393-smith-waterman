#!/usr/bin/env python3
"""Correctness check: every GPU version must produce exactly the CPU scores.

Scores are integers, so the files must match exactly (no tolerance).

  python scripts/verify.py --in data/smoke.txt --golden data/smoke.golden.txt
  python scripts/verify.py --in data/generated/x.txt              # runs sw_cpu for the reference
  python scripts/verify.py --in data/smoke.txt --versions v0,v1
Exit code 0 only if every requested version passes.
"""
import argparse
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def exe(bin_dir: str, name: str) -> str:
    for cand in (name + ".exe", name):
        p = os.path.join(bin_dir, cand)
        if os.path.isfile(p):
            return p
    sys.exit(f"cannot find {name} in {bin_dir} - build first (see README)")


def read_scores(path: str) -> list:
    with open(path) as f:
        return [line.rstrip("\r\n").split("\t") for line in f if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--golden", help="reference scores; if omitted, sw_cpu is run")
    ap.add_argument("--versions", default="v0,v1,v2,v3,v4")
    ap.add_argument("--bin-dir", default=os.path.join(ROOT, "build", "bin"))
    args = ap.parse_args()

    tmp = tempfile.mkdtemp(prefix="swverify_")
    golden = args.golden
    if not golden:
        golden = os.path.join(tmp, "cpu.txt")
        subprocess.run([exe(args.bin_dir, "sw_cpu"), "--in", args.inp, "--out", golden, "--repeat", "1"],
                       check=True, stderr=subprocess.DEVNULL)
    ref = read_scores(golden)

    failed = 0
    for v in args.versions.split(","):
        out = os.path.join(tmp, f"{v}.txt")
        r = subprocess.run([exe(args.bin_dir, "sw_gpu"), "--in", args.inp, "--out", out,
                            "--repeat", "1", "--version", v], stderr=subprocess.PIPE, text=True)
        if r.returncode != 0:
            print(f"{v}: FAIL (exit {r.returncode}) {r.stderr.strip().splitlines()[-1:]}")
            failed += 1
            continue
        got = read_scores(out)
        bad = [(e, g) for e, g in zip(ref, got) if e != g]
        if len(got) != len(ref):
            print(f"{v}: FAIL ({len(got)} scores, expected {len(ref)})")
            failed += 1
        elif bad:
            e, g = bad[0]
            print(f"{v}: FAIL {len(bad)}/{len(ref)} pairs differ; first: {e[0]} expected {e[1]} got {g[1]}")
            failed += 1
        else:
            print(f"{v}: PASS ({len(ref)} pairs)")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
