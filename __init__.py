"""hyun woo kim, the pennsylvania state university, 2018-2019"""

import sys
import pandas
import numpy
import jellyfish
import recordlinkage
import os
import scipy

import crosswalk.preprocessing
import crosswalk.indexing
import crosswalk.comparing
import crosswalk.classifier
import crosswalk.evaluation

   
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
del get_versions, versions

