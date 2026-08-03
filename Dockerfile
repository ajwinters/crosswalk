# GPU environment for crosswalk — RAPIDS record linkage.
#
# Runs BOTH the CPU and GPU backends in one pinned environment, so CPU-vs-GPU
# comparisons are controlled to identical library versions on identical hardware.
# Built on RAPIDS pip wheels (cudf-cu12, cupy-cuda12x) + Numba for the custom
# Jaro-Winkler kernel, rather than the full official RAPIDS image, to stay lean.
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip git && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# GPU stack (RAPIDS wheels live on NVIDIA's index) + the CPU pipeline deps,
# installed together so pip resolves one consistent version set.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --extra-index-url=https://pypi.nvidia.com \
        "cudf-cu12" \
        "cupy-cuda12x" \
        "numba" \
        "pandas" \
        "numpy" \
        "recordlinkage" \
        "jellyfish" \
        "networkx" \
        "scipy" \
        "seaborn"

# Register the `crosswalk` package + console entry points (crosswalk-match, etc.).
# The repo is COPYied in only to bootstrap the EDITABLE install; deps are already
# present above so --no-deps skips re-resolution. At runtime you MOUNT the repo
# over this same path, and the editable install picks up your live code — so the
# entry points and `import crosswalk` always reflect the mounted source.
COPY . /workspace/crosswalk
RUN pip install --no-cache-dir --no-deps -e /workspace/crosswalk

WORKDIR /workspace/crosswalk
CMD ["/bin/bash"]
