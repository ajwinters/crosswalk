"""
End-to-end parity + timing for the GPU comparing+classifier path.

Shared front-end (generate -> standardize -> CPU indexing.match), then the
P x 7 matrix and Fellegi-Sunter fs_score are computed BOTH ways and the fs_score
is compared per pair. Also reports wall time for the two GPU-targeted stages vs
the CPU baseline (Comparing + Classifier), the ~8.16s in-container target.

A small warm-up call first triggers the Numba JIT compile so it isn't charged to
the timed run.

Run inside the container with the GPU attached:
  docker run --rm --gpus all -v "<repo>:/workspace/crosswalk" crosswalk-gpu \
      python /workspace/crosswalk/gpu/test_pipeline_parity.py
"""

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


def main(n_records=10000, seed=42):
    dfA, dfB = generate_match_data(n_records=n_records, seed=seed)
    dfA = crosswalk.preprocessing.standardize(FIELDS, dfA)
    dfB = crosswalk.preprocessing.standardize(FIELDS, dfB)
    valid_A = dfA[dfA["valid"] == 1].copy()
    valid_B = dfB[dfB["valid"] == 1].copy()
    cpairs = crosswalk.indexing.match(INDEXER, TRANSPOSED, valid_A, valid_B)
    cf = [FEATURES[0], FEATURES[1]]
    print(f"records: {len(valid_A)} x {len(valid_B)} | candidate pairs: {len(cpairs):,}")

    # --- warm up the JIT on a tiny slice so compile time isn't timed ---
    warm = cpairs[:256]
    _ = gpu_classifier.match(gpu_comparing.match(warm, cf, valid_A, valid_B),
                             FEATURES, valid_A, valid_B)

    # --- CPU path ---
    t0 = time.perf_counter()
    cpu_bm = crosswalk.comparing.match(cpairs, cf, valid_A, valid_B)
    t1 = time.perf_counter()
    cpu_vec = crosswalk.classifier.match(cpu_bm, FEATURES, valid_A, valid_B)
    t2 = time.perf_counter()

    # --- GPU path ---
    t3 = time.perf_counter()
    gpu_bm = gpu_comparing.match(cpairs, cf, valid_A, valid_B)
    t4 = time.perf_counter()
    gpu_vec = gpu_classifier.match(gpu_bm, FEATURES, valid_A, valid_B)
    t5 = time.perf_counter()

    # --- fs_score parity ---
    gpu_fs = gpu_vec["fs_score"].reindex(cpu_vec.index).to_numpy()
    cpu_fs = cpu_vec["fs_score"].to_numpy()
    diff = np.abs(gpu_fs - cpu_fs)
    print(f"\nfs_score parity:")
    print(f"  max abs diff      : {diff.max():.3e}")
    print(f"  mean abs diff     : {diff.mean():.3e}")
    print(f"  pairs diff > 1e-6 : {(diff > 1e-6).sum():,} / {len(diff):,}")

    cpu_cmp, cpu_cls = t1 - t0, t2 - t1
    gpu_cmp, gpu_cls = t4 - t3, t5 - t4
    print(f"\ntiming (Comparing + Classifier):")
    print(f"  CPU : {cpu_cmp:6.2f}s + {cpu_cls:6.2f}s = {cpu_cmp + cpu_cls:6.2f}s")
    print(f"  GPU : {gpu_cmp:6.2f}s + {gpu_cls:6.2f}s = {gpu_cmp + gpu_cls:6.2f}s")
    speedup = (cpu_cmp + cpu_cls) / (gpu_cmp + gpu_cls)
    print(f"  speedup: {speedup:.1f}x")

    ok = diff.max() < 1e-6
    print("\nRESULT:", "PASS — GPU fs_score matches CPU" if ok else "FAIL — see diffs")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
