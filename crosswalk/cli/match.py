"""
Load persisted A/B CSVs, run the match (GPU when available, else CPU), and write
the matches (fs_score >= threshold) to an output CSV for analysis.

This is the "consume the fixed input" step of the persistent-CSV workflow. It
does NOT regenerate data -- it reads the files written by
crosswalk-generate, so the same inputs are used every run and the output
can be joined back to them.

The output CSV has one row per match, with both records' ids/names/ssn, both
true_entity_ids, an is_true_match flag, and the fs_score -- enough to measure
precision/recall and eyeball why a pair matched.

Run as `crosswalk-match --threshold 15` (after `pip install -e .`) or
`python -m crosswalk.cli.match --threshold 15`.
"""

import argparse
import os

import pandas as pd

import crosswalk.shared.preprocessing
from crosswalk.gpu.backend import link, resolve_backend
from crosswalk.gpu.config import FIELDS, INDEXER, TRANSPOSED, FEATURES

DATA_DIR = "data"   # default I/O dir, relative to the current working directory

# Force text columns to load as strings. county in particular is all-numeric
# text ('1'..'67') and pandas would otherwise read it as int, changing the
# schema the matcher expects (it factorizes county as an object column).
_STR_COLS = {"county": str, "referralid": str, "referraltype": str,
             "firstname": str, "lastname": str}


def load(path):
    return pd.read_csv(path, dtype=_STR_COLS)


def main():
    ap = argparse.ArgumentParser(description="GPU match over persisted CSVs")
    ap.add_argument("--a-file", default=os.path.join(DATA_DIR, "match_dfA.csv"))
    ap.add_argument("--b-file", default=os.path.join(DATA_DIR, "match_dfB.csv"))
    ap.add_argument("--threshold", type=float, default=15.0,
                    help="keep pairs with fs_score >= threshold (default 15)")
    ap.add_argument("--out", default=os.path.join(DATA_DIR, "matches.csv"))
    ap.add_argument("--backend", choices=["auto", "gpu", "cpu"], default="auto",
                    help="compute backend (default auto: GPU if available, else CPU)")
    args = ap.parse_args()

    dfA = load(args.a_file)
    dfB = load(args.b_file)
    print(f"loaded A={len(dfA):,} rows, B={len(dfB):,} rows")

    dfA = crosswalk.shared.preprocessing.standardize(FIELDS, dfA)
    dfB = crosswalk.shared.preprocessing.standardize(FIELDS, dfB)
    vA = dfA[dfA["valid"] == 1].copy()
    vB = dfB[dfB["valid"] == 1].copy()

    chosen = resolve_backend(args.backend)
    print(f"backend: {chosen.upper()}")
    links = link(vA, vB, INDEXER, TRANSPOSED, FEATURES,
                 threshold=args.threshold, backend=chosen)

    la = links.index.get_level_values(0)
    lb = links.index.get_level_values(1)
    out = pd.DataFrame({
        "a_id": vA.loc[la, "referralid"].to_numpy(),
        "b_id": vB.loc[lb, "referralid"].to_numpy(),
        "a_firstname": vA.loc[la, "firstname"].to_numpy(),
        "a_lastname": vA.loc[la, "lastname"].to_numpy(),
        "b_firstname": vB.loc[lb, "firstname"].to_numpy(),
        "b_lastname": vB.loc[lb, "lastname"].to_numpy(),
        "a_ssn": vA.loc[la, "ssn"].to_numpy(),
        "b_ssn": vB.loc[lb, "ssn"].to_numpy(),
        "a_entity": vA.loc[la, "true_entity_id"].to_numpy(),
        "b_entity": vB.loc[lb, "true_entity_id"].to_numpy(),
        "fs_score": links["fs_score"].to_numpy(),
    })
    out["is_true_match"] = out["a_entity"] == out["b_entity"]
    out = out.sort_values("fs_score", ascending=False)
    out.to_csv(args.out, index=False)

    # performance summary vs ground truth
    a_eids = set(vA["true_entity_id"])
    true_total = int(vB["true_entity_id"].isin(a_eids).sum())  # recoverable + defeated
    found = int(out["is_true_match"].sum())
    n = len(out)
    precision = found / n if n else 0.0
    recall = found / true_total if true_total else 0.0
    print(f"\nmatches (fs_score >= {args.threshold}): {n:,} -> {args.out}")
    print(f"  precision : {precision:.3f}  ({found:,} true / {n:,} kept)")
    print(f"  recall    : {recall:.3f}  ({found:,} true found / {true_total:,} true total)")
    print(f"  fs_score range in output: {out['fs_score'].min():.1f} .. {out['fs_score'].max():.1f}"
          if n else "  (no matches at this threshold)")


if __name__ == "__main__":
    main()
