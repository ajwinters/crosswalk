"""
Streaming / batched GPU matcher -- the scalable path.

The non-streaming gpu modules materialize the entire candidate-pair set, which is
fine to ~100k records but impossible at the 1M target (~20 billion pairs, ~140 GB
just for the boolean matrix). This module never holds the full set: it uploads the
record data to the GPU ONCE, then for each blocking key streams the candidate
pairs through in output-bounded CHUNKS, scores each chunk entirely on-device, and
keeps only the links whose fs_score clears a threshold.

KEY IDEAS
  * Upload once, reuse: encoded name buffers + per-field value/notna arrays live
    on the GPU for the whole run; nothing is re-transferred per chunk.
  * Output-bounded chunking: within a blocking key, each A record will produce
    nB[key_value] pairs; we cumulative-sum that and slice A so every chunk's pair
    count stays under a VRAM budget -- so a common name ("JAMES") can't blow up a
    chunk.
  * On-device scoring: each chunk's pairs come straight off the cuDF join as cuPy
    index arrays, feed the JW kernel + cuPy exact + Fellegi-Sunter math, and only
    the surviving links (a small set) ever come back to the host.
  * Dedup at the end: a pair found via two keys is scored twice with identical
    fs_score; we drop duplicate (labelA, labelB) once, cheaply.

fs_score is accumulated in feat_list order (firstname, lastname, suffix, then
features[0]) to match the CPU classifier's row-sum exactly, so with a -inf
threshold this reproduces the full CPU output bit-for-bit (see test).
"""

import sys

import numpy as np
import pandas as pd
import cupy as cp
import cudf
from numba import cuda

from jaro_winkler import encode_names
from comparing import _name_compare_kernel

_TPB = 128


class StreamingMatcher:
    def __init__(self, valid_A, valid_B, features, e=0.05, jw_threshold=0.9):
        self.features = features
        self.feat_list = ["firstname", "lastname", "suffix"] + list(features[0])
        self.exact_fields = ["suffix"] + list(features[0])
        self.jw_threshold = float(jw_threshold)
        self.nA, self.nB = len(valid_A), len(valid_B)
        self.labelA = valid_A.index.to_numpy()
        self.labelB = valid_B.index.to_numpy()

        # --- name buffers, uploaded once ---
        fnA_buf, fnA_len = encode_names(valid_A["firstname"].to_numpy())
        lnA_buf, lnA_len = encode_names(valid_A["lastname"].to_numpy())
        fnB_buf, fnB_len = encode_names(valid_B["firstname"].to_numpy())
        lnB_buf, lnB_len = encode_names(valid_B["lastname"].to_numpy())
        self.d_fnA, self.d_lfnA = cuda.to_device(fnA_buf), cuda.to_device(fnA_len)
        self.d_lnA, self.d_llnA = cuda.to_device(lnA_buf), cuda.to_device(lnA_len)
        self.d_fnB, self.d_lfnB = cuda.to_device(fnB_buf), cuda.to_device(fnB_len)
        self.d_lnB, self.d_llnB = cuda.to_device(lnB_buf), cuda.to_device(lnB_len)

        # --- exact-field value + notna arrays, uploaded once ---
        # numeric: keep values (NaN-aware); object (county): factorize to int codes
        self.exact = {}
        for f in self.exact_fields:
            a = valid_A[f].to_numpy()
            b = valid_B[f].to_numpy()
            if a.dtype.kind in "OUS" or b.dtype.kind in "OUS":
                codes, _ = pd.factorize(np.concatenate([a, b]), use_na_sentinel=True)
                ca, cb = codes[: self.nA], codes[self.nA:]
                self.exact[f] = (cp.asarray(ca), cp.asarray(ca != -1),
                                 cp.asarray(cb), cp.asarray(cb != -1))
            else:
                self.exact[f] = (cp.asarray(a), cp.asarray(~pd.isna(a)),
                                 cp.asarray(b), cp.asarray(~pd.isna(b)))

        # --- Fellegi-Sunter scalar weights per feature (match path, no Method I) ---
        epsilon = sys.float_info.epsilon
        m = (1 - e) * (1 - e) + e * e - epsilon
        self.w = {}
        for var in self.feat_list:
            u = (1.0 / valid_A[var].nunique()) + epsilon
            self.w[var] = (float(np.log2(m / u)), float(np.log2((1 - m) / (1 - u))))

        # --- blocking inputs ---
        self._valid_A = valid_A
        self._valid_B = valid_B

    # -----------------------------------------------------------------
    def _score(self, posA, posB):
        """fs_score for a chunk of pairs (cuPy index arrays), computed on-device."""
        C = int(posA.size)
        out_fn = cuda.device_array(C, np.uint8)
        out_ln = cuda.device_array(C, np.uint8)
        blocks = (C + _TPB - 1) // _TPB
        _name_compare_kernel[blocks, _TPB](
            self.d_fnA, self.d_lfnA, self.d_lnA, self.d_llnA,
            self.d_fnB, self.d_lfnB, self.d_lnB, self.d_llnB,
            posA, posB, self.jw_threshold, out_fn, out_ln,
        )
        cuda.synchronize()

        fs = cp.zeros(C, dtype=np.float64)
        # name features: valid rows always present, so no missing-value zeroing
        wa, wd = self.w["firstname"]
        fs += cp.where(cp.asarray(out_fn) != 0, wa, wd)
        wa, wd = self.w["lastname"]
        fs += cp.where(cp.asarray(out_ln) != 0, wa, wd)
        # exact features (in feat_list order)
        for f in self.exact_fields:
            valA, notnaA, valB, notnaB = self.exact[f]
            na = notnaA[posA] & notnaB[posB]
            eq = (valA[posA] == valB[posB]) & na
            wa, wd = self.w[f]
            fs += cp.where(na, cp.where(eq, wa, wd), 0.0)
        return fs

    def _stream(self, a_keycols, B_frame, nb_for_a, score_threshold,
                target, sink):
        """Stream one blocking key: chunk A by predicted output, score, collect."""
        cs = np.cumsum(nb_for_a)
        if len(cs) == 0 or cs[-1] == 0:
            return
        grp = np.floor(cs / max(target, 1)).astype(np.int64)
        bnd = np.flatnonzero(np.diff(grp)) + 1
        starts = np.concatenate([[0], bnd])
        ends = np.concatenate([bnd, [len(grp)]])
        ncol = len(a_keycols)

        for s, e in zip(starts, ends):
            pos = np.arange(s, e, dtype=np.int64)
            if ncol == 1:
                ga = cudf.DataFrame({"k0": a_keycols[0][s:e], "a": pos}).dropna(subset=["k0"])
                m = ga.merge(B_frame, on="k0")
            else:
                ga = cudf.DataFrame({"k0": a_keycols[0][s:e], "k1": a_keycols[1][s:e],
                                     "a": pos}).dropna(subset=["k0", "k1"])
                m = ga.merge(B_frame, on=["k0", "k1"])
            if len(m) == 0:
                continue
            posA = m["a"].values
            posB = m["b"].values
            fs = self._score(posA, posB)
            keep = fs >= score_threshold
            if bool(keep.any()):
                sink[0].append(cp.asnumpy(posA[keep]))
                sink[1].append(cp.asnumpy(posB[keep]))
                sink[2].append(cp.asnumpy(fs[keep]))

    # -----------------------------------------------------------------
    def run(self, indexer, transposed, score_threshold=-np.inf,
            target_chunk_pairs=20_000_000):
        """Stream all blocking keys and return links (level_0, level_1, fs_score).

        score_threshold defaults to -inf (keep everything -> reproduces the full
        CPU output, for parity). Real large-scale runs pass a real threshold so
        only matches are retained.
        """
        A, B = self._valid_A, self._valid_B
        sink = ([], [], [])

        for key in indexer:
            keyA = A[key].to_numpy()
            nb = B.groupby(key).size()
            nb_for_a = pd.Series(keyA).map(nb).fillna(0).to_numpy()
            B_frame = cudf.DataFrame({"k0": B[key].to_numpy(),
                                      "b": np.arange(self.nB, dtype=np.int64)}
                                     ).dropna(subset=["k0"])
            self._stream([keyA], B_frame, nb_for_a, score_threshold,
                         target_chunk_pairs, sink)

        for j in transposed:
            f0, f1 = j[0], j[1]
            kA0 = A[f0].to_numpy()
            kA1 = A[f1].to_numpy()
            # A(f0,f1) matches B(f1,f0): predict via B grouped on (f1, f0)
            nb = B.groupby([f1, f0]).size()
            nb_for_a = nb.reindex(pd.MultiIndex.from_arrays([kA0, kA1])).fillna(0).to_numpy()
            B_frame = cudf.DataFrame({"k0": B[f1].to_numpy(), "k1": B[f0].to_numpy(),
                                      "b": np.arange(self.nB, dtype=np.int64)}
                                     ).dropna(subset=["k0", "k1"])
            self._stream([kA0, kA1], B_frame, nb_for_a, score_threshold,
                         target_chunk_pairs, sink)

        if not sink[0]:
            return pd.DataFrame({"fs_score": []},
                                index=pd.MultiIndex.from_arrays([[], []],
                                                                names=["level_0", "level_1"]))
        a_pos = np.concatenate(sink[0])
        b_pos = np.concatenate(sink[1])
        fs = np.concatenate(sink[2])
        out = pd.DataFrame({"level_0": self.labelA[a_pos],
                            "level_1": self.labelB[b_pos],
                            "fs_score": fs})
        out = out.drop_duplicates(subset=["level_0", "level_1"])
        return out.set_index(["level_0", "level_1"])
