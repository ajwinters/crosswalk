"""
Shared pipeline configuration for the demo / runner scripts.

This is the **sample schema** (the Pennsylvania administrative-data layout the
library was built on). It lives in one place so the runners, benchmark, and
parity tests don't each copy it. To run your own data, edit these four values —
see the "Using your own data" section of the README.

  FIELDS      column map handed to standardize (what to clean)
  INDEXER     blocking keys — a pair is a candidate if it matches exactly on any
  TRANSPOSED  extra blocking on swapped name pairs (SMITH JAMES == JAMES SMITH)
  FEATURES    [exact_match_fields, interaction_terms, frequency_weighted_fields]

`firstname`, `lastname`, and `suffix` are fixed names required by the comparison
stages; everything else is your choice.
"""

FIELDS = {
    "firstname": "firstname", "lastname": "lastname", "suffix": "suffix",
    "pid1": "pid1", "pid2": "pid2", "region": "region",
    "dobyy": "dobyy", "dobmm": "dobmm", "dobdd": "dobdd",
}
INDEXER = ["firstname", "lastname", "pid1"]
TRANSPOSED = [["firstname", "lastname"]]
FEATURES = [["pid1", "region", "bin1", "bin2"], [[]], ["firstname", "lastname"]]
