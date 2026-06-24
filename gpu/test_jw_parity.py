"""
Correctness harness for the GPU Jaro-Winkler kernel.

De-risks the single custom primitive of the port BEFORE building the pipeline
around it: compares the Numba CUDA kernel against CPU `jellyfish` on many string
pairs deliberately spread across the 0.9 threshold (identical, 1-2 edit typos,
nickname-ish, and unrelated pairs), then reports raw-float agreement AND the
boolean (sim >= 0.9) agreement that recordlinkage actually uses.

Run inside the container with the GPU attached:
  docker run --rm --gpus all -v "<repo>:/workspace/crosswalk" crosswalk-gpu \
      python /workspace/crosswalk/gpu/test_jw_parity.py
"""

import numpy as np
import jellyfish

from jaro_winkler import jaro_winkler_pairwise

ALPHA = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

POOL = [
    "JAMES", "ROBERT", "JOHN", "MICHAEL", "WILLIAM", "DAVID", "RICHARD",
    "JOSEPH", "THOMAS", "CHRISTOPHER", "ELIZABETH", "JENNIFER", "MARGARET",
    "CATHERINE", "STEPHANIE", "MOHAMMED", "PRIYA", "OKONKWO", "GONZALEZ",
    "WASHINGTON", "CUNNINGHAM", "SMITH", "NGUYEN", "MARTINEZ", "ANDERSON",
    "LI", "WU", "MO",  # short names (min length after standardize)
]


def _edit(s, rng, n_edits=1):
    """Apply n single-character edits to a string."""
    for _ in range(n_edits):
        if len(s) < 2:
            return s
        action = rng.integers(0, 4)
        pos = int(rng.integers(0, len(s)))
        if action == 0 and len(s) > 2:
            s = s[:pos] + s[pos + 1:]                       # delete
        elif action == 1 and pos < len(s) - 1:
            s = s[:pos] + s[pos + 1] + s[pos] + s[pos + 2:]  # transpose
        elif action == 2:
            s = s[:pos] + ALPHA[int(rng.integers(0, 26))] + s[pos + 1:]  # sub
        else:
            s = s[:pos] + ALPHA[int(rng.integers(0, 26))] + s[pos:]      # insert
    return s


def build_pairs(rng, n_each=4000):
    """Build (a, b) pairs spread across the JW range."""
    a_list, b_list, kind = [], [], []

    def add(a, b, k):
        a_list.append(a); b_list.append(b); kind.append(k)

    for _ in range(n_each):
        base = POOL[int(rng.integers(0, len(POOL)))]
        add(base, base, "identical")                          # JW = 1.0
        add(base, _edit(base, rng, 1), "1-edit")              # near boundary
        add(base, _edit(base, rng, 2), "2-edit")              # below-ish
        other = POOL[int(rng.integers(0, len(POOL)))]
        add(base, other, "random")                            # mostly low
    return a_list, b_list, kind


def main():
    rng = np.random.default_rng(123)
    a_list, b_list, kind = build_pairs(rng)
    n = len(a_list)

    gpu = jaro_winkler_pairwise(a_list, b_list)
    cpu = np.array([jellyfish.jaro_winkler_similarity(a, b)
                    for a, b in zip(a_list, b_list)], dtype=np.float64)

    diff = np.abs(gpu - cpu)
    print(f"pairs compared           : {n:,}")
    print(f"max abs float diff       : {diff.max():.3e}")
    print(f"mean abs float diff      : {diff.mean():.3e}")
    print(f"pairs with diff > 1e-6   : {(diff > 1e-6).sum()}")

    # the decision recordlinkage actually uses
    gpu_b = gpu >= 0.9
    cpu_b = cpu >= 0.9
    disagree = gpu_b != cpu_b
    print(f"boolean(>=0.9) mismatches: {disagree.sum()} / {n}")

    if disagree.any():
        print("\nsample boolean mismatches (near-boundary):")
        idx = np.where(disagree)[0][:15]
        for i in idx:
            print(f"  {a_list[i]:>14} vs {b_list[i]:<14} "
                  f"gpu={gpu[i]:.6f} cpu={cpu[i]:.6f} [{kind[i]}]")
    if (diff > 1e-6).any():
        print("\nworst float diffs:")
        for i in np.argsort(-diff)[:15]:
            print(f"  {a_list[i]:>14} vs {b_list[i]:<14} "
                  f"gpu={gpu[i]:.6f} cpu={cpu[i]:.6f} diff={diff[i]:.2e} [{kind[i]}]")

    ok = diff.max() < 1e-6 and not disagree.any()
    print("\nRESULT:", "PASS — GPU JW matches CPU jellyfish" if ok
          else "FAIL — see mismatches above")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
