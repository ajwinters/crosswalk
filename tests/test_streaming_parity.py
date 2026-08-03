"""
Parity harness for the streaming GPU matcher.

With score_threshold = -inf the streaming matcher keeps every candidate pair, so
it must reproduce the FULL CPU pipeline output exactly: same set of pairs AND
identical fs_score per pair. This validates that output-bounded chunking +
on-device scoring + cross-key dedup is faithful to the reference.

  docker run --rm --gpus all -v "<repo>:/workspace/crosswalk" crosswalk-gpu \
      python /workspace/crosswalk/tests/test_streaming_parity.py
"""

import numpy as np

import crosswalk.shared.preprocessing
import crosswalk.cpu.indexing
import crosswalk.cpu.comparing
import crosswalk.cpu.classifier
from crosswalk.datagen.generate_data import generate_match_data
from crosswalk.gpu.streaming import StreamingMatcher
from crosswalk.gpu.config import FIELDS, INDEXER, TRANSPOSED, FEATURES


def main(n_records=5000, seed=42, target_chunk_pairs=2_000_000):
    dfA, dfB = generate_match_data(n_records=n_records, seed=seed)
    dfA = crosswalk.shared.preprocessing.standardize(FIELDS, dfA)
    dfB = crosswalk.shared.preprocessing.standardize(FIELDS, dfB)
    vA = dfA[dfA["valid"] == 1].copy()
    vB = dfB[dfB["valid"] == 1].copy()

    # CPU reference: full pipeline
    cpairs = crosswalk.cpu.indexing.match(INDEXER, TRANSPOSED, vA, vB)
    cpu_bm = crosswalk.cpu.comparing.match(cpairs, [FEATURES[0], FEATURES[1]], vA, vB)
    cpu_vec = crosswalk.cpu.classifier.match(cpu_bm, FEATURES, vA, vB)
    cpu_fs = cpu_vec["fs_score"]
    print(f"CPU candidate pairs: {len(cpu_fs):,}")

    # GPU streaming, keep everything (small chunk size to force multi-chunk path)
    matcher = StreamingMatcher(vA, vB, FEATURES)
    links = matcher.run(INDEXER, TRANSPOSED, score_threshold=-np.inf,
                        target_chunk_pairs=target_chunk_pairs)
    print(f"GPU streaming pairs: {len(links):,}")

    cpu_set = set(map(tuple, np.asarray(cpu_fs.index.to_list())))
    gpu_set = set(map(tuple, np.asarray(links.index.to_list())))
    print(f"pairs only in CPU: {len(cpu_set - gpu_set):,}")
    print(f"pairs only in GPU: {len(gpu_set - cpu_set):,}")

    aligned = links["fs_score"].reindex(cpu_fs.index)
    n_missing = int(aligned.isna().sum())
    diff = np.abs(aligned.to_numpy() - cpu_fs.to_numpy())
    diff = diff[~np.isnan(diff)]
    print(f"fs_score max abs diff: {diff.max() if diff.size else float('nan'):.3e}")
    print(f"aligned pairs missing in GPU: {n_missing}")

    ok = (cpu_set == gpu_set) and n_missing == 0 and diff.size and diff.max() < 1e-6
    print("\nRESULT:", "PASS — streaming matches full CPU output" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
