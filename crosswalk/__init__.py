"""crosswalk -- GPU-accelerated probabilistic record linkage.

Originally authored by Hyun Woo Kim, The Pennsylvania State University, 2018-2019.

Subpackages (import explicitly; nothing is loaded eagerly here, so a CPU-only
install never touches the GPU or plotting dependencies):
  crosswalk.shared    standardize -- the shared front-end
  crosswalk.cpu       CPU scoring backend (recordlinkage / jellyfish)
  crosswalk.gpu       GPU port (RAPIDS cuDF/cuPy + Numba) with CPU failover
  crosswalk.datagen   synthetic-data generator
  crosswalk.cli       console entry points (crosswalk-match / -generate / -benchmark)
"""

__version__ = "0.1.0"
