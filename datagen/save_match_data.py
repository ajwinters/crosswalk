"""
Generate and PERSIST a synthetic A/B match dataset to CSV.

This is the "produce the fixed input" step of the persistent-CSV workflow: it
writes match_dfA.csv and match_dfB.csv once, and the matcher
(gpu/run_match_csv.py) loads them going forward. Unlike regenerating from the
seed each run, the files can be opened, inspected, and joined back to matcher
output for analysis.

The CSVs carry every column the generator produces, including true_entity_id
(ground truth) and the binary indicators -- so downstream analysis can measure
precision/recall.

  docker run --rm -v "<repo>:/workspace/crosswalk" crosswalk-gpu \
      python /workspace/crosswalk/datagen/save_match_data.py --n-records 10000
"""

import argparse
import os

from generate_data import generate_match_data

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
)


def main():
    ap = argparse.ArgumentParser(description="Generate + persist A/B match CSVs")
    ap.add_argument("--n-records", type=int, default=10000)
    ap.add_argument("--match-rate", type=float, default=0.85)
    ap.add_argument("--defeat-rate", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=DATA_DIR)
    args = ap.parse_args()

    dfA, dfB = generate_match_data(
        n_records=args.n_records, match_rate=args.match_rate,
        defeat_rate=args.defeat_rate, seed=args.seed,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    a_path = os.path.join(args.out_dir, "match_dfA.csv")
    b_path = os.path.join(args.out_dir, "match_dfB.csv")
    dfA.to_csv(a_path, index=False)
    dfB.to_csv(b_path, index=False)

    # ground-truth summary: how many B records have a true counterpart in A
    a_eids = set(dfA["true_entity_id"])
    true_pairs = int(dfB["true_entity_id"].isin(a_eids).sum())

    print(f"seed={args.seed} match_rate={args.match_rate} defeat_rate={args.defeat_rate}")
    print(f"dfA: {len(dfA):,} rows -> {a_path}")
    print(f"dfB: {len(dfB):,} rows -> {b_path}")
    print(f"true corresponding pairs (B rows whose entity is in A): {true_pairs:,}")
    print(f"  (of those, ~{args.defeat_rate:.0%} defeat blocking and are unrecoverable)")


if __name__ == "__main__":
    main()
