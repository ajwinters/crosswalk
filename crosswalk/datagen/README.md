# datagen — synthetic data for crosswalk testing

Self-contained synthetic-data generator and end-to-end test harness, kept
separate from the matching library (`crosswalk/*.py`). Nothing in the matching
pipeline imports from here.

## Files
- `generate_data.py` — parameterized generator. Emits both dedupe and match
  datasets with a `true_entity_id` ground-truth column, generic binary indicator
  fields, and a calibrated mix of realistic variations.
- `name_variants.py` — curated nickname / phonetic / OCR variant tables, sampled
  deterministically (no LLM or external service at runtime).
- `run_pipeline.py` — runs generated data through the full pipeline
  (standardize → index → compare → classify → tiebreak → components) with
  optional profiling.

## Usage (run from the repo root)
```
python scripts/run_pipeline.py                 # 10k records, both pipelines
python scripts/run_pipeline.py --small         # hardcoded sanity-check data
python scripts/run_pipeline.py --n-records 50000 --profile
python scripts/run_pipeline.py --save-data     # writes CSVs to crosswalk/data/
```

Key knobs: `--n-records`, `--match-rate`, `--dedupe-rate`, `--defeat-rate`
(fraction of true matches that defeat blocking → invisible false negatives),
`--seed`.

## Ground truth
`true_entity_id` is stable across a record and its variants; siblings and
unique-to-B records get fresh ids. Join on it to score precision/recall, and to
assert the future GPU port produces identical output on the same seeded input.
