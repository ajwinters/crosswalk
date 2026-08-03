"""
Backend selection — run the match on the GPU when available, fall back to the CPU.

The GPU path (StreamingMatcher) needs an NVIDIA GPU plus RAPIDS (cuDF/cuPy) and
Numba. The CPU path needs only pandas + recordlinkage + jellyfish. This module
detects what's actually available and dispatches, so callers get the *same*
links DataFrame either way -- and the repo runs on a plain laptop with no GPU.

`link()` returns one row per candidate pair scoring at or above `threshold`,
indexed by (level_0, level_1) with an `fs_score` column, identical in shape for
both backends. The CPU backend is the same code the GPU port is validated
against, so results agree bit-for-bit.
"""


def gpu_available():
    """True iff the GPU stack imports AND a CUDA device is actually present.

    Guards both failure modes: RAPIDS/Numba not installed (import raises) and
    installed-but-no-device (`cuda.is_available()` is False).
    """
    try:
        from numba import cuda
        import cupy   # noqa: F401  (import is the availability check)
        import cudf   # noqa: F401
        return bool(cuda.is_available())
    except Exception:
        return False


def resolve_backend(backend="auto"):
    """Resolve 'auto'/'gpu'/'cpu' to a concrete 'gpu' or 'cpu'.

    'gpu' errors if no GPU is available; 'auto' picks GPU when present, else CPU.
    """
    if backend == "cpu":
        return "cpu"
    if backend == "gpu":
        if not gpu_available():
            raise RuntimeError(
                "backend='gpu' requested but no CUDA GPU / RAPIDS is available")
        return "gpu"
    if backend == "auto":
        return "gpu" if gpu_available() else "cpu"
    raise ValueError(f"backend must be 'auto', 'gpu', or 'cpu' (got {backend!r})")


def _link_gpu(vA, vB, indexer, transposed, features, threshold):
    from .streaming import StreamingMatcher   # lazy: only imported on the GPU path
    matcher = StreamingMatcher(vA, vB, features)
    links = matcher.run(indexer, transposed, score_threshold=threshold)
    return links[["fs_score"]]


def _link_cpu(vA, vB, indexer, transposed, features, threshold):
    import crosswalk.cpu.indexing as cpu_indexing
    import crosswalk.cpu.comparing as cpu_comparing
    import crosswalk.cpu.classifier as cpu_classifier
    cpairs = cpu_indexing.match(indexer, transposed, vA, vB)
    bm = cpu_comparing.match(cpairs, [features[0], features[1]], vA, vB)
    vec = cpu_classifier.match(bm, features, vA, vB)
    links = vec[vec["fs_score"] >= threshold]
    return links[["fs_score"]]


def link(vA, vB, indexer, transposed, features, threshold=15.0, backend="auto"):
    """Match two standardized frames; return links (level_0, level_1, fs_score).

    Parameters mirror the pipeline config: `indexer` / `transposed` are the
    blocking keys, `features` is the [exact_fields, interaction_terms,
    fs_frequency_fields] triple. `backend` is 'auto' (GPU if present, else CPU),
    'gpu' (force; errors if absent), or 'cpu' (force).
    """
    chosen = resolve_backend(backend)
    fn = _link_gpu if chosen == "gpu" else _link_cpu
    return fn(vA, vB, indexer, transposed, features, threshold)
