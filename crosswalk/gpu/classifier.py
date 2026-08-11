"""
GPU port of crosswalk.cpu.classifier.match -> Fellegi-Sunter fs_score per pair.

The match path of the CPU classifier (classifier.match) deliberately SKIPS the
Method-I frequency weighting that the dedupe path uses, so every per-feature
weight is a pair-independent SCALAR:

    m   = (1-e)^2 + e^2 - epsilon                 (constant for all features)
    u   = 1 / nunique(dfA[feature]) + epsilon     (per feature)
    w_a = log2(m / u)        # weight when the feature AGREES (bmatrix col == 1)
    w_d = log2((1-m)/(1-u))  # weight when it DISAGREES

The only per-pair work is: pick w_a or w_d by the boolean comparison column,
zero it out when either underlying value is missing, and sum across features.
That elementwise selection + reduction is what runs on the GPU (cuPy).

PARITY NOTES
  * nunique is computed with pandas on valid_A (a cheap scalar) so u -- and hence
    w_a / w_d -- are bit-identical to the CPU values.
  * weights are summed in feat_list order, matching pandas' left-to-right
    DataFrame.sum(axis=1), so fs_score matches to floating-point exactness.
  * missingness uses pd.notna on the gathered values (works for numeric and the
    object 'region' column alike), mirroring the CPU notna_cond.
"""

import sys

import numpy as np
import pandas as pd
import cupy as cp


def match(bmatrix, features, dfA, dfB, e=0.05):
    """GPU equivalent of crosswalk.cpu.classifier.match. Returns a DataFrame with the
    per-feature w_<feature> columns plus fs_score, indexed by the pair MultiIndex.
    """
    epsilon = sys.float_info.epsilon
    feat_list = ["firstname", "lastname", "suffix"] + list(features[0])

    level0 = bmatrix.index.get_level_values(0)
    level1 = bmatrix.index.get_level_values(1)
    posA = dfA.index.get_indexer(level0)
    posB = dfB.index.get_indexer(level1)
    P = len(posA)

    m = (1 - e) * (1 - e) + e * e - epsilon  # constant across features

    out = {}
    fs = cp.zeros(P, dtype=np.float64)
    for var_order, var in enumerate(feat_list):
        # scalar agreement/disagreement weights (identical to CPU)
        u = (1.0 / dfA[var].nunique()) + epsilon
        w_a = float(np.log2(m / u))
        w_d = float(np.log2((1 - m) / (1 - u)))

        # gather underlying values only to reproduce the missing-value zeroing
        fa = dfA[var].to_numpy()[posA]
        gb = dfB[var].to_numpy()[posB]
        notna = cp.asarray(pd.notna(fa) & pd.notna(gb))

        agree = cp.asarray(bmatrix[var_order].to_numpy()) == 1
        weight = cp.where(notna, cp.where(agree, w_a, w_d), 0.0)

        out["w_" + var] = cp.asnumpy(weight)
        fs = fs + weight  # accumulate in feat_list order to match pandas row-sum

    out["fs_score"] = cp.asnumpy(fs)
    vector = pd.DataFrame(out)
    vector.index = pd.MultiIndex.from_arrays([level0, level1])
    return vector
