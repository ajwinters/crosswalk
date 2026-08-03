"""
GPU port of crosswalk.cpu.comparing.match -> the P x 7 boolean comparison matrix.

Mirrors the CPU function exactly so the rest of the pipeline (and the parity
test) is unaffected:

  columns produced (after the cross-field fixup), matching the CPU output:
    0: firstname   1: lastname   2: suffix   3..: each field in features[0]
  with run_pipeline's features[0] = ['ssn','county','bin1','bin2'] that is
    [0]firstname [1]lastname [2]suffix [3]ssn [4]county [5]bin1 [6]bin2

The name columns use the custom Jaro-Winkler kernel (the one thing RAPIDS lacks);
the exact columns are plain elementwise equality (cuPy for numeric fields,
host-side factorize for string/object fields like county). Everything is keyed
by candidate pair, so a single Numba kernel handles all four JW comparisons per
pair at once.

INDEX HANDLING (parity-critical): candidate_pairs is a 2-level MultiIndex whose
values are dfA / dfB *index labels*. After the validity filter those labels are
non-contiguous, so we translate them to row POSITIONS for on-device gathering,
then rebuild the identical unnamed MultiIndex on the result so that the CPU
classifier's `reset_index()` still yields 'level_0' / 'level_1'.
"""

import numpy as np
import pandas as pd
import cupy as cp
from numba import cuda

from .jaro_winkler import encode_names, jaro_winkler_device


# ===========================================================================
# Kernel: one thread per candidate pair. Computes all four JW comparisons
# (direct firstname/lastname plus the two cross-field ones used to catch
# transposed names) and applies the cross-field rule inline:
#   firstname_match = (fn~fn) OR (fn~ln AND ln~fn)
#   lastname_match  = (ln~ln) OR (fn~ln AND ln~fn)
# This is exactly the CPU fixup in comparing.match, just done per-thread.
# ===========================================================================
@cuda.jit
def _name_compare_kernel(fnA, lfnA, lnA, llnA, fnB, lfnB, lnB, llnB,
                         posA, posB, thr, out_fn, out_ln):
    p = cuda.grid(1)
    if p < out_fn.size:
        i = posA[p]
        j = posB[p]
        fn_fn = jaro_winkler_device(fnA[i], lfnA[i], fnB[j], lfnB[j]) >= thr
        ln_ln = jaro_winkler_device(lnA[i], llnA[i], lnB[j], llnB[j]) >= thr
        fn_ln = jaro_winkler_device(fnA[i], lfnA[i], lnB[j], llnB[j]) >= thr
        ln_fn = jaro_winkler_device(lnA[i], llnA[i], fnB[j], lfnB[j]) >= thr
        cross = fn_ln and ln_fn
        out_fn[p] = 1 if (fn_fn or cross) else 0
        out_ln[p] = 1 if (ln_ln or cross) else 0


def _exact_compare(a_series, b_series, posA, posB):
    """Elementwise exact-match column for one field over the candidate pairs.

    Returns an int array (1 where the gathered values are equal and both
    present, else 0) -- matching recordlinkage.exact, where any missing value
    or inequality yields 0 (NaN == NaN is False).
    """
    a = a_series.to_numpy()
    b = b_series.to_numpy()

    if a.dtype.kind in "OUS" or b.dtype.kind in "OUS":
        # string/object (e.g. county): factorize jointly so equal strings share
        # a code; -1 marks missing and must never count as a match.
        codes, _ = pd.factorize(np.concatenate([a, b]), use_na_sentinel=True)
        na = len(a)
        ca = codes[:na][posA]
        cb = codes[na:][posB]
        eq = (ca == cb) & (ca != -1) & (cb != -1)
        return eq.astype(np.int64)

    # numeric (ssn, suffix, bin*): gather + compare on the GPU. NaN != NaN, so
    # missing values correctly yield 0.
    ag = cp.asarray(a)[cp.asarray(posA)]
    bg = cp.asarray(b)[cp.asarray(posB)]
    return cp.asnumpy(ag == bg).astype(np.int64)


def match(candidate_pairs, features, dfA, dfB, threshold=0.9, threads_per_block=128):
    """GPU equivalent of crosswalk.cpu.comparing.match.

    Parameters mirror the CPU function: `features[0]` is the list of exact-match
    fields (after suffix). Returns a pandas DataFrame with integer columns
    0..N indexed by the original candidate-pair MultiIndex.
    """
    level0 = candidate_pairs.get_level_values(0)
    level1 = candidate_pairs.get_level_values(1)

    # label -> row position for on-device gathering
    posA = dfA.index.get_indexer(level0).astype(np.int64)
    posB = dfB.index.get_indexer(level1).astype(np.int64)
    P = len(posA)

    # --- name columns via the JW kernel ---
    fnA_buf, fnA_len = encode_names(dfA["firstname"].to_numpy())
    lnA_buf, lnA_len = encode_names(dfA["lastname"].to_numpy())
    fnB_buf, fnB_len = encode_names(dfB["firstname"].to_numpy())
    lnB_buf, lnB_len = encode_names(dfB["lastname"].to_numpy())

    d_fnA, d_lfnA = cuda.to_device(fnA_buf), cuda.to_device(fnA_len)
    d_lnA, d_llnA = cuda.to_device(lnA_buf), cuda.to_device(lnA_len)
    d_fnB, d_lfnB = cuda.to_device(fnB_buf), cuda.to_device(fnB_len)
    d_lnB, d_llnB = cuda.to_device(lnB_buf), cuda.to_device(lnB_len)
    d_posA, d_posB = cuda.to_device(posA), cuda.to_device(posB)
    out_fn = cuda.device_array(P, dtype=np.uint8)
    out_ln = cuda.device_array(P, dtype=np.uint8)

    blocks = (P + threads_per_block - 1) // threads_per_block
    _name_compare_kernel[blocks, threads_per_block](
        d_fnA, d_lfnA, d_lnA, d_llnA, d_fnB, d_lfnB, d_lnB, d_llnB,
        d_posA, d_posB, threshold, out_fn, out_ln,
    )
    cuda.synchronize()

    cols = {
        0: out_fn.copy_to_host().astype(np.int64),
        1: out_ln.copy_to_host().astype(np.int64),
    }

    # --- exact columns: suffix, then each field in features[0] ---
    exact_fields = ["suffix"] + list(features[0])
    for offset, field in enumerate(exact_fields):
        cols[2 + offset] = _exact_compare(dfA[field], dfB[field], posA, posB)

    bmatrix = pd.DataFrame(cols)
    bmatrix.index = pd.MultiIndex.from_arrays([level0, level1])
    return bmatrix
