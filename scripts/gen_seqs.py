#!/usr/bin/env python3
"""Deterministic DNA pair generator. Owner: Hein.

Writes the team input format: pair_id <TAB> seq_a <TAB> seq_b, one pair per line.
The same --seed always produces byte-identical files, so a grader can regenerate
every input we benchmark.

seq_b is seq_a with random substitutions/insertions/deletions (rate --mutate),
so pairs actually align and scores are meaningful. --mutate 1.0 gives
independent random sequences.

Examples:
  python scripts/gen_seqs.py --pairs 64   --len 256   --seed 1 --out data/smoke.txt
  python scripts/gen_seqs.py --pairs 1024 --len 1000  --seed 42 --out data/generated/b1024_l1000.txt
"""
import argparse
import os
import random

ALPHABET = "ACGT"


def mutate(rng: random.Random, s: str, rate: float) -> str:
    out = []
    for c in s:
        r = rng.random()
        if r >= rate:
            out.append(c)                       # keep
            continue
        kind = rng.randrange(3)
        if kind == 0:
            out.append(rng.choice(ALPHABET))    # substitution
        elif kind == 1:
            out.append(c)                       # insertion after c
            out.append(rng.choice(ALPHABET))
        # kind == 2: deletion
    return "".join(out) or rng.choice(ALPHABET)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", type=int, required=True, help="number of pairs")
    ap.add_argument("--len", type=int, required=True, help="length of seq_a")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--mutate", type=float, default=0.15, help="per-base mutation rate for seq_b (default 0.15)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="\n") as f:
        f.write(f"# gen_seqs.py --pairs {args.pairs} --len {args.len} --seed {args.seed} --mutate {args.mutate}\n")
        for i in range(args.pairs):
            a = "".join(rng.choice(ALPHABET) for _ in range(args.len))
            b = mutate(rng, a, args.mutate) if args.mutate < 1.0 else \
                "".join(rng.choice(ALPHABET) for _ in range(args.len))
            f.write(f"p{i:06d}\t{a}\t{b}\n")


if __name__ == "__main__":
    main()
