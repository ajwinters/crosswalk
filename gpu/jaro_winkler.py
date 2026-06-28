"""
Jaro-Winkler similarity as a Numba CUDA kernel.
================================================

WHY THIS FILE EXISTS
--------------------
The GPU port maps almost entirely onto RAPIDS: cuDF does the blocking joins,
cuPy does the Fellegi-Sunter arithmetic. The ONE thing neither library provides
is a string-similarity function, and Jaro-Winkler is the heart of the comparison
step. So this is the single piece of hand-written GPU code in the whole port.

We write it with Numba, which lets us author a real CUDA kernel in Python: Numba
compiles the functions below to PTX (GPU assembly) and runs them on the device.
No C++ or nvcc needed, but underneath it is genuinely CUDA.

MATCHING THE CPU EXACTLY
------------------------
recordlinkage uses `jellyfish.jaro_winkler_similarity` on the CPU. For the port
to be trustworthy, this kernel must return the same numbers, so it mirrors
jellyfish precisely:
  * match window  = max(len_a, len_b) // 2 - 1
  * transpositions are floored via INTEGER division (jellyfish quirk; see below)
  * Winkler prefix bonus: capped at 4 chars, scaling factor p = 0.1
  * the prefix bonus is only applied when the base Jaro score exceeds 0.7

recordlinkage's `comparer.string(..., threshold=0.9)` ultimately keeps only the
BOOLEAN (sim >= 0.9), so exact float parity strictly matters only near 0.9 (where
Jaro is well above 0.7 and the boost always applies). We match the raw float
anyway, so the kernel is reusable and easy to validate (see test_jw_parity.py).

THE STRING-ON-GPU PROBLEM
-------------------------
A GPU kernel cannot work with Python `str` objects (they're heap objects with no
fixed layout). So before launching, every name is ENCODED into a fixed-width grid
of byte values: an (n, MAXLEN) uint8 array where row i holds the ASCII codes of
name i, zero-padded, plus a separate length array so the kernel knows where each
name really ends. Names are short uppercase ASCII after standardization, so this
is cheap and wastes little memory.
"""

import numpy as np
import pandas as pd
import numba
from numba import cuda

# Maximum supported name length (in characters).
#
# This MUST be a compile-time constant because each GPU thread allocates small
# local scratch arrays of this exact size (see `cuda.local.array` below) -- the
# compiler needs the size baked in. Standardized names are short (< ~15 chars),
# so 32 is comfortable headroom. Anything longer is simply truncated at encode
# time, which is fine for real names.
MAXLEN = 32


# ===========================================================================
# DEVICE FUNCTION: runs ON the GPU, called BY a kernel (one call per thread).
#
# `@cuda.jit(device=True)` marks this as a *device function* -- GPU code that is
# not launched directly from the host but is inlined into a kernel. Think of it
# as a helper that executes inside a single GPU thread. It can return a value
# (unlike a kernel, which cannot).
#
# Arguments are the ENCODED form of two strings:
#   a, b   : 1D uint8 arrays (rows of the encoded buffer) holding ASCII codes
#   la, lb : the true lengths of those two strings (ignore the zero padding)
# ===========================================================================
@cuda.jit(device=True)
def jaro_winkler_device(a, la, b, lb):
    """Jaro-Winkler similarity between two encoded strings (runs in one thread)."""

    # --- Edge cases, matching jellyfish's behavior ---
    if la == 0 and lb == 0:
        return 1.0          # two empty strings are considered identical
    if la == 0 or lb == 0:
        return 0.0          # one empty, one not -> nothing in common

    # --- Step 1: the "match window" ---
    # Two characters can only be considered a match if they sit within this many
    # positions of each other. This is the classic Jaro definition.
    max_len = la if la > lb else lb
    match_distance = max_len // 2 - 1
    if match_distance < 0:          # very short strings can give -1; clamp to 0
        match_distance = 0

    # --- Per-thread scratch space ---
    # `cuda.local.array(size, dtype)` gives each thread its own private array,
    # living in fast per-thread memory. We use two flag arrays to mark which
    # characters in `a` and in `b` have been matched. Size must be the constant
    # MAXLEN (hence the constant requirement noted above).
    a_matches = cuda.local.array(MAXLEN, numba.uint8)
    b_matches = cuda.local.array(MAXLEN, numba.uint8)
    for i in range(la):
        a_matches[i] = 0
    for j in range(lb):
        b_matches[j] = 0

    # --- Step 2: count matching characters ---
    # Walk through `a`. For each character, scan only the window of `b` around the
    # same position, and claim the FIRST not-yet-used equal character. "Greedy,
    # each b char used once" -- the b_matches flag prevents reusing a b position.
    matches = 0
    for i in range(la):
        start = i - match_distance
        if start < 0:
            start = 0
        end = i + match_distance + 1
        if end > lb:
            end = lb
        for j in range(start, end):
            if b_matches[j] == 0 and a[i] == b[j]:
                a_matches[i] = 1
                b_matches[j] = 1
                matches += 1
                break               # this `a[i]` is matched; move to next i

    if matches == 0:
        return 0.0                  # no characters in common -> similarity 0

    # --- Step 3: count transpositions ---
    # Among the matched characters, line up `a`'s matched chars (in a-order)
    # against `b`'s matched chars (in b-order) and count positions where they
    # differ. `k` walks b's matched positions in lockstep.
    #
    # IMPORTANT jellyfish quirk: the number of "half transpositions" is floored
    # with INTEGER division (t_count // 2), NOT a real divide by 2.0. Using float
    # division makes odd mismatch counts disagree by exactly 1/18 in the final
    # score -- which is precisely the bug the parity test caught the first time.
    t_count = 0
    k = 0
    for i in range(la):
        if a_matches[i] == 1:
            while b_matches[k] == 0:     # advance k to b's next matched char
                k += 1
            if a[i] != b[k]:
                t_count += 1
            k += 1
    t = float(t_count // 2)

    # --- Step 4: the Jaro score ---
    # Average of three ratios: fraction of `a` matched, fraction of `b` matched,
    # and fraction of matches that are in the right order.
    m = float(matches)
    jaro = (m / la + m / lb + (m - t) / m) / 3.0

    # --- Step 5: the Winkler boost ---
    # Reward strings that share a common prefix (people mistype the tail of a
    # name more than the head). The boost is only applied when the base Jaro is
    # above 0.7 -- jellyfish's behavior -- which is why low-similarity pairs are
    # left untouched. `prefix` is the shared leading-character count, capped at 4.
    if jaro > 0.7:
        lim = la if la < lb else lb
        if lim > 4:
            lim = 4
        prefix = 0
        for i in range(lim):
            if a[i] == b[i]:
                prefix += 1
            else:
                break
        return jaro + prefix * 0.1 * (1.0 - jaro)
    return jaro


# ===========================================================================
# KERNEL: the function launched from the host. `@cuda.jit` (no device=True).
#
# A kernel runs once per GPU thread, in parallel across thousands of threads.
# `cuda.grid(1)` returns THIS thread's unique index in a 1D launch -- so thread
# k handles element k. The bounds check `k < out.size` guards the last block,
# which may have more threads than there are elements.
#
# Here the layout is "pairwise": row k of bufA is compared with row k of bufB.
# (When this is wired into the real comparison step, we'll instead pass candidate
# pair indices so thread k compares dfA[pair_i[k]] with dfB[pair_j[k]].)
# ===========================================================================
@cuda.jit
def _jw_pairwise_kernel(bufA, lenA, bufB, lenB, out):
    """out[k] = JW(A[k], B[k]) for every k, all threads in parallel."""
    k = cuda.grid(1)
    if k < out.size:
        out[k] = jaro_winkler_device(bufA[k], lenA[k], bufB[k], lenB[k])


# ===========================================================================
# HOST HELPERS: ordinary CPU Python that prepares data and launches the kernel.
# ===========================================================================

def encode_names(strings, maxlen=MAXLEN):
    """Pack a list of strings into the GPU-friendly form.

    Returns:
      buf  : (n, maxlen) uint8 array -- row i is name i as ASCII codes, 0-padded
      lens : (n,) int32 array -- the real length of each name

    This is the bridge from "Python strings" to "numbers a kernel can read."

    Vectorized: cast to fixed-width null-padded bytes ('S{maxlen}') and view as
    uint8. For standardized names (short, uppercase ASCII) this is byte-identical
    to the per-character loop it replaces, but runs in C instead of Python -- the
    difference between negligible and a real bottleneck at 1M+ records.
    """
    s = pd.Series(strings).fillna("").astype(str).str.slice(0, maxlen)
    lens = s.str.len().to_numpy().astype(np.int32)
    if len(s) == 0:
        return np.zeros((0, maxlen), dtype=np.uint8), lens
    buf = s.to_numpy().astype("S" + str(maxlen)).view(np.uint8).reshape(-1, maxlen)
    return np.ascontiguousarray(buf), lens


def jaro_winkler_pairwise(list_a, list_b, maxlen=MAXLEN, threads_per_block=128):
    """Compute JW similarity elementwise between two equal-length string lists.

    This is the host-side entry point used by the parity test. It encodes the
    inputs, ships them to the GPU, launches the kernel, and brings the results
    back. Returns a host float64 array of length len(list_a).
    """
    if len(list_a) != len(list_b):
        raise ValueError("list_a and list_b must be the same length")
    n = len(list_a)
    out = np.zeros(n, dtype=np.float64)
    if n == 0:
        return out

    # 1) Encode both string lists into byte buffers on the CPU.
    bufA, lenA = encode_names(list_a, maxlen)
    bufB, lenB = encode_names(list_b, maxlen)

    # 2) Copy the inputs (and the output buffer) from host RAM to GPU VRAM.
    #    `cuda.to_device` returns a handle to the device-side array.
    dA = cuda.to_device(bufA)
    dlA = cuda.to_device(lenA)
    dB = cuda.to_device(bufB)
    dlB = cuda.to_device(lenB)
    dout = cuda.to_device(out)

    # 3) Choose the launch configuration. GPU threads are grouped into blocks;
    #    we need enough blocks of `threads_per_block` to cover all n elements.
    #    The ceiling division ensures the last (possibly partial) block exists.
    blocks = (n + threads_per_block - 1) // threads_per_block

    # 4) Launch: kernel[blocks, threads_per_block](args). This returns
    #    immediately (the GPU runs asynchronously), so we synchronize before
    #    reading the result back to the host.
    _jw_pairwise_kernel[blocks, threads_per_block](dA, dlA, dB, dlB, dout)
    cuda.synchronize()
    return dout.copy_to_host()
