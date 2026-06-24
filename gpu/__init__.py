"""GPU port of the crosswalk comparison pipeline (gpu branch).

Mirrors the CPU library's math (indexing -> comparing -> classifier) on the GPU
via RAPIDS (cuDF/cuPy) plus a custom Numba CUDA Jaro-Winkler kernel. The CPU
modules are kept intact as the parity reference.
"""
