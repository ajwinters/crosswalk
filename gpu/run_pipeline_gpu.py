"""
End-to-end GPU match pipeline runner (mirror of datagen/run_pipeline.py).

Runs: generate -> standardize (CPU) -> indexing -> comparing -> classifier,
where indexing/comparing/classifier use the GPU ports. Reports per-stage wall
time. With --compare-cpu it also runs the CPU stages on the same data and prints
the fs_score parity (max abs diff) and per-stage speedups -- a one-command
CPU-vs-GPU harness.

standardize stays on CPU by design (O(N) string cleaning, never the bottleneck).
A small warm-up triggers the Numba JIT compile so it is not charged to timing.

  docker run --rm --gpus all -v "<repo>:/workspace/crosswalk" crosswalk-gpu \
      python /workspace/crosswalk/gpu/run_pipeline_gpu.py --n-records 10000 --compare-cpu
"""

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PSU = os.path.dirname(REPO)
sys.path.insert(0, PSU)
sys.path.insert(0, os.path.join(REPO, "datagen"))
sys.path.insert(0, HERE)

import crosswalk.preprocessing
import crosswalk.indexing
import crosswalk.comparing
import crosswalk.classifier
from generate_data import generate_match_data
from numba import cuda
import indexing as gpu_indexing
import comparing as gpu_comparing
import classifier as gpu_classifier

FIELDS = {
    "firstname": "firstname", "lastname": "lastname", "suffix": "suffix",
    "ssn": "ssn", "mciid": "mciid", "county": "county",
    "dobyy": "dobyy", "dobmm": "dobmm", "dobdd": "dobdd",
}
INDEXER = ["firstname", "lastname", "ssn"]
TRANSPOSED = [["firstname", "lastname"]]
FEATURES = [["ssn", "county", "bin1", "bin2"], [[]], ["firstname", "lastname"]]
COMPARE_FEATURES = [FEATURES[0], FEATURES[1]]


def run_gpu(valid_A, valid_B):
    timings = {}
    t = time.perf_counter()
    cpairs = gpu_indexing.match(INDEXER, TRANSPOSED, valid_A, valid_B)
    cuda.synchronize()
    timings["Indexing"] = time.perf_counter() - t

    t = time.perf_counter()
    bm = gpu_comparing.match(cpairs, COMPARE_FEATURES, valid_A, valid_B)
    cuda.synchronize()
    timings["Comparing"] = time.perf_counter() - t

    t = time.perf_counter()
    vec = gpu_classifier.match(bm, FEATURES, valid_A, valid_B)
    cuda.synchronize()
    timings["Classifier"] = time.perf_counter() - t
    return cpairs, vec, timings


def run_cpu(valid_A, valid_B):
    timings = {}
    t = time.perf_counter()
    cpairs = crosswalk.indexing.match(INDEXER, TRANSPOSED, valid_A, valid_B)
    timings["Indexing"] = time.perf_counter() - t

    t = time.perf_counter()
    bm = crosswalk.comparing.match(cpairs, COMPARE_FEATURES, valid_A, valid_B)
    timings["Comparing"] = time.perf_counter() - t

    t = time.perf_counter()
    vec = crosswalk.classifier.match(bm, FEATURES, valid_A, valid_B)
    timings["Classifier"] = time.perf_counter() - t
    return cpairs, vec, timings


def main():
    ap = argparse.ArgumentParser(description="GPU match pipeline runner")
    ap.add_argument("--n-records", type=int, default=10000)
    ap.add_argument("--match-rate", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--compare-cpu", action="store_true",
                    help="also run CPU stages, print fs_score parity + speedups")
    args = ap.parse_args()

    print(f"GPU match pipeline | n={args.n_records:,} seed={args.seed} "
          f"match_rate={args.match_rate}")

    t = time.perf_counter()
    dfA, dfB = generate_match_data(n_records=args.n_records,
                                   match_rate=args.match_rate, seed=args.seed)
    gen_t = time.perf_counter() - t

    t = time.perf_counter()
    dfA = crosswalk.preprocessing.standardize(FIELDS, dfA)
    dfB = crosswalk.preprocessing.standardize(FIELDS, dfB)
    valid_A = dfA[dfA["valid"] == 1].copy()
    valid_B = dfB[dfB["valid"] == 1].copy()
    std_t = time.perf_counter() - t
    print(f"valid: {len(valid_A)} x {len(valid_B)}")

    # warm up the JIT (not timed)
    warm_pairs = gpu_indexing.match(INDEXER, TRANSPOSED, valid_A, valid_B)[:256]
    _ = gpu_classifier.match(
        gpu_comparing.match(warm_pairs, COMPARE_FEATURES, valid_A, valid_B),
        FEATURES, valid_A, valid_B)

    cpairs, gpu_vec, gpu_t = run_gpu(valid_A, valid_B)
    print(f"candidate pairs: {len(cpairs):,}")

    print("\n--- GPU timing ---")
    print(f"  {'Generate (CPU)':<16}{gen_t:7.2f}s")
    print(f"  {'Standardize(CPU)':<16}{std_t:7.2f}s")
    for k in ("Indexing", "Comparing", "Classifier"):
        print(f"  {k:<16}{gpu_t[k]:7.2f}s")
    gpu_core = sum(gpu_t.values())
    print(f"  {'GPU core total':<16}{gpu_core:7.2f}s  (Indexing+Comparing+Classifier)")

    if args.compare_cpu:
        cpu_cpairs, cpu_vec, cpu_t = run_cpu(valid_A, valid_B)
        gpu_fs = gpu_vec["fs_score"].reindex(cpu_vec.index).to_numpy()
        cpu_fs = cpu_vec["fs_score"].to_numpy()
        diff = float(np.abs(gpu_fs - cpu_fs).max())

        print("\n--- CPU vs GPU ---")
        print(f"  {'stage':<14}{'CPU':>9}{'GPU':>9}{'speedup':>9}")
        for k in ("Indexing", "Comparing", "Classifier"):
            print(f"  {k:<14}{cpu_t[k]:8.2f}s{gpu_t[k]:8.2f}s"
                  f"{cpu_t[k]/gpu_t[k]:8.1f}x")
        cpu_core = sum(cpu_t.values())
        print(f"  {'CORE TOTAL':<14}{cpu_core:8.2f}s{gpu_core:8.2f}s"
              f"{cpu_core/gpu_core:8.1f}x")
        print(f"\n  fs_score max abs diff: {diff:.3e}  "
              f"({'PARITY OK' if diff < 1e-6 else 'MISMATCH'})")


if __name__ == "__main__":
    main()
