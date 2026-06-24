# GPU environment for the crosswalk record-linkage port (gpu branch).
#
# Deliberately runs BOTH the CPU baseline and the GPU version in one pinned
# environment, so CPU-vs-GPU performance comparisons are controlled to identical
# library versions on identical hardware.
#
# Built on RAPIDS pip wheels (cudf-cu12, cupy-cuda12x) + Numba for the custom
# Jaro-Winkler kernel, rather than the full official RAPIDS image, to keep it
# lean and self-documenting. The repo is mounted at runtime (see README), not
# COPYied in, so edits on the host are live inside the container.
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip git && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# GPU stack (RAPIDS wheels live on NVIDIA's index) + the existing CPU pipeline
# dependencies, installed together so pip resolves one consistent version set.
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

WORKDIR /workspace
CMD ["/bin/bash"]
