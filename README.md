# Crosswalk — GPU-Accelerated Probabilistic Record Linkage

Crosswalk performs **probabilistic (Fellegi–Sunter) record linkage** — deciding
which records across two datasets refer to the same real-world person, even when
names are misspelled, transposed, or nicknamed and identifiers are missing. It
was built for linkages across Pennsylvania administrative data but works on **any
two datasets** that share names and other identifiers.

This fork **moves the comparison engine to the GPU.** The linkage math is
unchanged — the GPU version produces results **bit-identical** to the original
CPU library — but it runs several times faster on modest data and, thanks to a
streaming design, scales to datasets the CPU cannot process at all.

> Originally authored by **Hyun Woo Kim**, The Pennsylvania State University,
> 2018–2019. Code review by **Alex Winters** (2020–2021); GPU port by Alex
> Winters.

---

## How record linkage works here

Given two record sets *A* and *B*, comparing every record in *A* against every
record in *B* is *N × M* — a trillion comparisons at a million records each. The
pipeline avoids that with four stages:

1. **Standardize** (`shared/preprocessing.py`) — clean and normalize names and
   fields (uppercase, strip junk, coerce SSN/DOB), and flag unusable records.
   *O(N)* per dataset; the shared front-end for both backends.
2. **Index / blocking** (`indexing`) — only *pair up* records that agree exactly
   on at least one blocking key (e.g. first name, last name, or SSN, plus
   transposed names). This is what makes the problem tractable — it produces
   **candidate pairs** instead of the full cross-product.
3. **Compare** (`comparing`) — for each candidate pair, measure field agreement:
   **Jaro–Winkler** similarity (thresholded at 0.9) on names, exact match on the
   rest. Produces a boolean *agreement pattern* per pair.
4. **Classify** (`classifier`) — turn each agreement pattern into a
   **Fellegi–Sunter score** (`fs_score`): a sum of per-field log-likelihood
   weights (agreement is evidence *for* a match, disagreement *against*, weighted
   by how informative each field is). Threshold the score to declare **links**.

The heavy work is the last two stages, which run once **per candidate pair** —
and candidate pairs grow roughly with *N²*. That's exactly the load the GPU port
targets.

---

## Setup

RAPIDS is Linux-only, so everything runs in a container (the `Dockerfile`
pip-installs cuDF, cuPy, Numba, and the CPU dependencies). On Windows this uses
Docker Desktop's WSL2 backend with GPU passthrough. You need an **NVIDIA GPU**.

```bash
# build the image once
docker build -t crosswalk-gpu .

# run anything inside it, mounting the repo and attaching the GPU
docker run --rm --gpus all -v "<repo-path>:/workspace/crosswalk" crosswalk-gpu \
    python /workspace/crosswalk/<script>
```

---

## Using your own data

Nothing in the pipeline is tied to the sample data — point it at any two
DataFrames or CSVs.

### The data contract

| Column | Required | Role |
|---|---|---|
| `firstname` | **yes**, this exact name | fuzzy-compared (Jaro–Winkler ≥ 0.9) |
| `lastname` | **yes**, this exact name | fuzzy-compared (Jaro–Winkler ≥ 0.9) |
| `suffix` | **yes** (may be all-`NaN`) | exact-compared |
| anything else | your choice | blocking keys and/or exact-match features |

The two fuzzy-matched name fields and `suffix` are referenced by name inside the
comparison stages, so **rename your columns to `firstname` / `lastname` /
`suffix`** before running. If your data has no suffix concept, add
`df["suffix"] = np.nan` — a missing field simply contributes zero weight.

**Everything else is configuration, not code.** Your blocking keys and exact-match
comparison fields are just column names you pass in — SSN, date of birth, ZIP,
county, member ID, gender, race flags, anything. Binary / one-hot indicators need
no special handling: drop them into `FEATURES[0]` and they flow through the
exact-match + Fellegi–Sunter path automatically.

### Minimal example

```python
import pandas as pd
import crosswalk.shared.preprocessing as preprocessing
from streaming import StreamingMatcher          # gpu/streaming.py

# 1. load your two datasets (rename name columns to the contract above)
dfA = pd.read_csv("my_data_A.csv")
dfB = pd.read_csv("my_data_B.csv")

# 2. describe your schema
FIELDS = {                                       # what standardize should clean
    "firstname": "firstname", "lastname": "lastname", "suffix": "suffix",
    "ssn": "ssn", "county": "county",            # optional; omit what you lack
}
INDEXER    = ["firstname", "lastname", "ssn"]    # blocking keys — YOUR columns
TRANSPOSED = [["firstname", "lastname"]]         # also catch swapped names
FEATURES   = [
    ["ssn", "county", "gender"],                 # exact-match fields — YOUR columns
    [[]],                                        # interaction terms (none)
    ["firstname", "lastname"],                   # frequency-weighted (dedupe only)
]

# 3. standardize and drop unusable rows
dfA = preprocessing.standardize(FIELDS, dfA)
dfB = preprocessing.standardize(FIELDS, dfB)
vA = dfA[dfA["valid"] == 1].copy()
vB = dfB[dfB["valid"] == 1].copy()

# 4. match on the GPU — streams, so record count is not memory-bound
matcher = StreamingMatcher(vA, vB, FEATURES)
links = matcher.run(INDEXER, TRANSPOSED, score_threshold=25)
# links: one row per match, indexed by (level_0, level_1) with an fs_score column
```

`gpu/run_match_csv.py` is a working reference implementation of exactly this
(CSV in → scored matches out). The simplest way to run your own data is to copy
it and edit the config block at the top.

### Two things to tune

- **Blocking keys** decide what's even *comparable*. A true match whose blocking
  keys are all corrupted is never compared — an invisible false negative. More
  keys = better recall but more candidate pairs (more work).
- **`score_threshold`** is data-dependent and **scales with dataset size**: false
  positives grow ~*N²* while true matches grow ~*N*, so a threshold tuned on 10k
  records will be far too permissive at 1M. Tune it against a labeled sample.

---

## The GPU port

**What moves to the GPU:** blocking (`indexing`), comparison (`comparing`), and
scoring (`classifier`). Standardization stays on the CPU by design — it's *O(N)*
string cleaning and never the bottleneck.

**How it's built:**
- **RAPIDS cuDF** does the blocking joins; **cuPy** does the Fellegi–Sunter
  arithmetic — the parts that map onto existing GPU primitives.
- A **custom Numba CUDA kernel** (`gpu/jaro_winkler.py`) implements Jaro–Winkler,
  the one primitive RAPIDS doesn't provide. It mirrors the CPU (`jellyfish`)
  implementation exactly, so the thresholded results match bit-for-bit.
- A **streaming matcher** (`gpu/streaming.py`) is the scalable path: it uploads
  the record data once, then streams candidate pairs through in output-bounded
  chunks, scoring each chunk fully on-device and keeping only links above a
  threshold. **Peak memory is bounded by chunk size, not pair count** — so it
  processes billions of pairs in a couple of gigabytes of VRAM.

**Results** (RTX 4070 Ti SUPER, 16 GB; both paths in one container, same data,
`fs_score` bit-identical throughout):

| Records | Candidate pairs | CPU | GPU | Speedup |
|--------:|----------------:|----:|----:|--------:|
| 10,000  | 1.98 M          | 6.9 s  | 1.2 s  | **5.7×** (19× on the Jaro–Winkler stage) |
| 25,000  | 12.5 M          | 35.5 s | 0.5 s  | **79×** |
| 1,000,000 | ~20 B         | ~16 h *(extrapolated; OOMs)* | **~4 min @ 1.4 GB VRAM** | — |

The CPU cannot run the 1M locally — it would need to materialize a ~320 GB
candidate-pair index. The original CPU port was run on a distributed platform
with some additional blocking tricks and 16hr estimate matches closely with those
earlier runs. The GPU streams it in minutes within a couple GB of VRAM.
`gpu/plot_benchmark.py` renders this comparison as `benchmark_cpu_vs_gpu.png`.

---

## Repository layout

```
crosswalk/
├── shared/preprocessing.py     # standardize — shared front-end (CPU, O(N))
├── cpu/                        # original CPU backend (recordlinkage / jellyfish)
│   └── indexing.py  comparing.py  classifier.py  evaluation.py
├── gpu/                        # GPU port (RAPIDS cuDF/cuPy + Numba)
│   ├── jaro_winkler.py         # custom CUDA Jaro–Winkler kernel
│   ├── indexing.py comparing.py classifier.py   # per-stage GPU ports
│   ├── streaming.py            # StreamingMatcher — the scalable path
│   ├── run_match_csv.py        # CSV in -> scored matches out (start here)
│   ├── run_pipeline_gpu.py     # one-command CPU-vs-GPU runner
│   ├── benchmark_1m.py  plot_benchmark.py
│   └── test_streaming_parity.py  test_pipeline_parity.py
├── datagen/                    # synthetic-data generator + CPU harness
└── Dockerfile                  # builds the `crosswalk-gpu` image
```

The CPU backend is kept intact — it's the correctness oracle the GPU port is
validated against.

---

## Demo & benchmarks (synthetic data)

`datagen/` generates realistic synthetic linkage data — Zipf-distributed names,
typos, transpositions, nicknames, phonetic and OCR variants, missing fields — with
a `true_entity_id` ground-truth column, so you can measure precision and recall
without real data.

| Task | Script |
|------|--------|
| CPU-vs-GPU on generated data | `gpu/run_pipeline_gpu.py --n-records 10000 --compare-cpu` |
| Persist a fixed synthetic dataset | `datagen/save_match_data.py --n-records 100000` |
| Match those CSVs → scored output | `gpu/run_match_csv.py --threshold 25` |
| Scaling benchmark up to 1M | `gpu/benchmark_1m.py` |
| Parity tests (GPU == CPU) | `gpu/test_streaming_parity.py`, `gpu/test_pipeline_parity.py` |
| Original CPU pipeline (sanity check) | `datagen/run_pipeline.py --small` |

`run_match_csv.py` writes `data/matches.csv` — one row per match with both
records' fields, both `true_entity_id`s, an `is_true_match` flag, and the
`fs_score` — so precision/recall against ground truth is immediate.

---

## Authors

- **Hyun Woo Kim** — original library, The Pennsylvania State University, 2018–2019.
- **Alex Winters** — code review and the GPU port.
