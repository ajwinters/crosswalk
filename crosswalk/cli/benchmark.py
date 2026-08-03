"""
CPU-vs-GPU scaling benchmark, up to 1M records.

Both paths SCORE every candidate pair; they differ only in whether the pair set
is materialized (CPU, recordlinkage) or streamed (GPU). A fixed fs_score
threshold is used so output handling stays cheap and the timing reflects scoring,
not host post-processing.

  both-run sizes : CPU full pipeline (indexing+comparing+classifier) vs GPU
                   streaming -- speedup + parity on the kept-link subset.
  gpu-only sizes : streaming only (CPU would OOM materializing the pair set).

The CPU cannot run 1M: indexing.match builds the entire ~20B-pair MultiIndex in
RAM (~320 GB). We extrapolate its 1M time from measured throughput instead.

Run as `crosswalk-benchmark` (after `pip install -e .`) or
`python -m crosswalk.cli.benchmark`.
"""

import time

import numpy as np
import cupy as cp

import crosswalk.shared.preprocessing
import crosswalk.cpu.indexing
import crosswalk.cpu.comparing
import crosswalk.cpu.classifier
from crosswalk.datagen.generate_data import generate_match_data
from crosswalk.gpu.streaming import StreamingMatcher
from crosswalk.gpu.config import FIELDS, INDEXER, TRANSPOSED, FEATURES
CF = [FEATURES[0], FEATURES[1]]
THRESHOLD = 15.0


def log(*a):
    print(*a, flush=True)


def prep(n, seed=42):
    t = time.perf_counter()
    dfA, dfB = generate_match_data(n_records=n, seed=seed)
    dfA = crosswalk.shared.preprocessing.standardize(FIELDS, dfA)
    dfB = crosswalk.shared.preprocessing.standardize(FIELDS, dfB)
    vA = dfA[dfA["valid"] == 1].copy()
    vB = dfB[dfB["valid"] == 1].copy()
    return vA, vB, time.perf_counter() - t


def candidate_count(vA, vB):
    total = 0
    for key in INDEXER:
        gA = vA.groupby(key).size().rename("na")
        gB = vB.groupby(key).size().rename("nb")
        m = gA.to_frame().join(gB.to_frame(), how="inner")
        total += int((m["na"] * m["nb"]).sum())
    gA = vA.groupby(["firstname", "lastname"]).size().reset_index(name="na")
    gB = vB.groupby(["firstname", "lastname"]).size().reset_index(name="nb")
    m = gA.merge(gB, left_on=["firstname", "lastname"], right_on=["lastname", "firstname"])
    return total + int((m["na"] * m["nb"]).sum())


def gpu_run(vA, vB, threshold):
    matcher = StreamingMatcher(vA, vB, FEATURES)
    t = time.perf_counter()
    links = matcher.run(INDEXER, TRANSPOSED, score_threshold=threshold)
    return links, time.perf_counter() - t


def cpu_run(vA, vB):
    t = time.perf_counter()
    cpairs = crosswalk.cpu.indexing.match(INDEXER, TRANSPOSED, vA, vB)
    bm = crosswalk.cpu.comparing.match(cpairs, CF, vA, vB)
    vec = crosswalk.cpu.classifier.match(bm, FEATURES, vA, vB)
    return vec["fs_score"], time.perf_counter() - t


def main():
    both = [10000, 25000]   # CPU capped low: recordlinkage at 50k+ OOMs the container
    gpu_only = [100000, 500000, 1000000]
    rows = []
    cpu_pps = None  # CPU pairs/sec, for extrapolation

    log("=" * 72)
    log("CPU vs GPU -- both run (score every candidate pair)")
    log("=" * 72)
    for n in both:
        vA, vB, prep_t = prep(n)
        cand = candidate_count(vA, vB)
        # warm GPU JIT once
        if n == both[0]:
            _ = gpu_run(vA, vB, THRESHOLD)
        cpu_fs, cpu_t = cpu_run(vA, vB)
        links, gpu_t = gpu_run(vA, vB, THRESHOLD)
        cpu_pps = cand / cpu_t

        # parity on the kept-link subset
        cpu_keep = cpu_fs[cpu_fs >= THRESHOLD]
        gpu_aligned = links["fs_score"].reindex(cpu_keep.index)
        same_set = set(map(tuple, np.asarray(cpu_keep.index.to_list()))) == \
            set(map(tuple, np.asarray(links.index.to_list())))
        d = np.abs(gpu_aligned.to_numpy() - cpu_keep.to_numpy())
        d = d[~np.isnan(d)]
        parity = (same_set and d.size and d.max() < 1e-6)

        log(f"n={n:>8,} pairs={cand:>14,}  CPU={cpu_t:7.1f}s  GPU={gpu_t:6.1f}s  "
            f"speedup={cpu_t/gpu_t:5.1f}x  links={len(links):>8,}  "
            f"parity={'OK' if parity else 'FAIL'}")
        rows.append((n, cand, cpu_t, gpu_t))

    log("\n" + "=" * 72)
    log("GPU only -- CPU would OOM materializing the pair set")
    log("=" * 72)
    for n in gpu_only:
        vA, vB, prep_t = prep(n)
        cand = candidate_count(vA, vB)
        links, gpu_t = gpu_run(vA, vB, THRESHOLD)
        free_b, total_b = cp.cuda.runtime.memGetInfo()
        cpu_proj = cand / cpu_pps if cpu_pps else float("nan")
        log(f"n={n:>8,} pairs={cand:>14,}  GPU={gpu_t:7.1f}s "
            f"({cand/gpu_t/1e6:5.0f}M pairs/s)  links={len(links):>9,}  "
            f"VRAM={(total_b-free_b)/1e9:4.1f}GB  "
            f"[CPU would take ~{cpu_proj/60:5.1f} min, and OOM]")

    log("\nNote: CPU 1M projection is extrapolated from measured CPU throughput "
        f"(~{cpu_pps/1e6:.1f}M pairs/s); the CPU cannot actually run it (pair "
        "index alone ~320 GB).")


if __name__ == "__main__":
    main()
