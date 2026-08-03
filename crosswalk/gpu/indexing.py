"""
GPU port of crosswalk.cpu.indexing.match -> candidate-pair MultiIndex via cuDF.

Blocking is just an exact-equality inner join: for each blocking key, every dfA
record is paired with every dfB record sharing that key value. cuDF does these
hash joins on the GPU. We union the per-key results (plus the transposed
firstname/lastname block) and drop duplicates, returning the same pandas
MultiIndex of (labelA, labelB) the CPU function produces.

SCALE NOTE: this materializes the FULL candidate-pair set, which is fine at the
current test sizes (~2M pairs at 10k records) but grows ~N^2 / n_distinct within
blocks. At ~1M records that is billions of pairs and will not fit in memory --
that regime needs block-by-block streaming, a deliberate redesign tracked
separately. This module is the correctness-equivalent of the CPU stage.

Missing keys are dropped (NaN never blocks), matching the exact-equality
semantics; standardized blocking keys in valid rows have no missing values in
practice, so this is parity-safe here.
"""

import numpy as np
import pandas as pd
import cudf


def _block_single(dfA, dfB, key, lblA, lblB):
    a = cudf.DataFrame({"k": dfA[key].to_numpy(), "a": lblA}).dropna(subset=["k"])
    b = cudf.DataFrame({"k": dfB[key].to_numpy(), "b": lblB}).dropna(subset=["k"])
    m = a.merge(b, on="k")
    return m["a"].to_numpy(), m["b"].to_numpy()


def _block_transposed(dfA, dfB, j0, j1, lblA, lblB):
    # pair A where (A[j0], A[j1]) == (B[j1], B[j0])
    a = cudf.DataFrame({"k0": dfA[j0].to_numpy(), "k1": dfA[j1].to_numpy(),
                        "a": lblA}).dropna(subset=["k0", "k1"])
    b = cudf.DataFrame({"k0": dfB[j1].to_numpy(), "k1": dfB[j0].to_numpy(),
                        "b": lblB}).dropna(subset=["k0", "k1"])
    m = a.merge(b, on=["k0", "k1"])
    return m["a"].to_numpy(), m["b"].to_numpy()


def match(indexer, transposed, dfA, dfB):
    """GPU equivalent of crosswalk.cpu.indexing.match. Returns a deduplicated
    pandas MultiIndex of (dfA label, dfB label) candidate pairs.
    """
    lblA = dfA.index.to_numpy()
    lblB = dfB.index.to_numpy()

    a_parts, b_parts = [], []
    for key in indexer:
        a, b = _block_single(dfA, dfB, key, lblA, lblB)
        a_parts.append(a)
        b_parts.append(b)
    for j in transposed:
        a, b = _block_transposed(dfA, dfB, j[0], j[1], lblA, lblB)
        a_parts.append(a)
        b_parts.append(b)

    allA = np.concatenate(a_parts) if a_parts else np.array([], dtype=lblA.dtype)
    allB = np.concatenate(b_parts) if b_parts else np.array([], dtype=lblB.dtype)
    return pd.MultiIndex.from_arrays([allA, allB]).drop_duplicates()
