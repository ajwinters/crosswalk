"""
End-to-end test of the crosswalk record linkage pipeline with synthetic data.
Tests both dedupe and match pipelines through:
  standardize -> indexing -> comparing -> classifier -> tiebreak -> components

Usage:
  python run_pipeline.py                          # default 10K records with profiling
  python run_pipeline.py --small                  # original 27-record sanity check
  python run_pipeline.py --n-records 1000 --profile
  python run_pipeline.py --n-records 50000 --match-rate 0.90 --dedupe-rate 0.25 --profile
"""

import sys
import os
import argparse
import time
import cProfile
import pstats
import tracemalloc
import io
import pandas as pd
import numpy as np
from datetime import datetime

import crosswalk.shared.preprocessing
import crosswalk.cpu.indexing
import crosswalk.cpu.comparing
import crosswalk.cpu.classifier
from crosswalk.datagen.generate_data import (
    generate_dedupe_data, generate_match_data, DEFAULT_BINARY_FIELDS,
)

# Generated CSVs go to crosswalk/data/, anchored to the repo root so the output
# location is stable no matter what directory the script is launched from.
DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
)


# ============================================================================
# Profiling helpers
# ============================================================================

class StepProfiler:
    """Tracks wall time and peak memory for each pipeline step."""

    def __init__(self, enabled=False):
        self.enabled = enabled
        self.steps = []
        self._step_name = None
        self._start_time = None
        self._start_mem = None

    def start(self, step_name):
        self._step_name = step_name
        if self.enabled:
            tracemalloc.start()
            self._start_time = time.perf_counter()

    def stop(self, extra_info=""):
        if not self.enabled:
            return
        elapsed = time.perf_counter() - self._start_time
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.steps.append({
            "step": self._step_name,
            "wall_time": elapsed,
            "peak_memory_mb": peak / (1024 * 1024),
            "info": extra_info,
        })

    def summary(self):
        if not self.enabled or not self.steps:
            return
        print("\n" + "=" * 78)
        print("PROFILING SUMMARY")
        print("=" * 78)
        print(f"{'Step':<25} {'Wall Time':>12} {'Peak Memory':>14} {'Info':>20}")
        print("-" * 78)
        total_time = 0
        max_mem = 0
        for s in self.steps:
            total_time += s["wall_time"]
            max_mem = max(max_mem, s["peak_memory_mb"])
            t = f"{s['wall_time']:.2f}s"
            m = f"{s['peak_memory_mb']:.1f} MB"
            print(f"{s['step']:<25} {t:>12} {m:>14} {s['info']:>20}")
        print("-" * 78)
        print(f"{'Total':<25} {f'{total_time:.2f}s':>12} {f'{max_mem:.1f} MB (peak)':>14}")
        print("=" * 78)


def dump_cprofile(func, args, prof_filename):
    """Run func(*args) under cProfile and dump stats to file."""
    profiler = cProfile.Profile()
    profiler.enable()
    result = func(*args)
    profiler.disable()
    profiler.dump_stats(prof_filename)
    # Print top 15 cumulative time entries
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats("cumulative")
    stats.print_stats(15)
    print(f"\ncProfile stats saved to {prof_filename}")
    print(stream.getvalue())
    return result


# ============================================================================
# Original small-scale data (27 dedupe, 20+20 match) for --small mode
# ============================================================================

def _augment_small(df, seed=0):
    """Add binary indicator + true_entity_id columns to hardcoded small data.

    Small mode is a smoke test (no precision/recall scoring), so true_entity_id is
    just a unique id per row; binary fields are seeded for reproducibility.
    """
    rng = np.random.default_rng(seed)
    if 'true_entity_id' not in df.columns:
        df['true_entity_id'] = np.arange(len(df))
    for f in DEFAULT_BINARY_FIELDS:
        df[f] = (rng.random(len(df)) < 0.5).astype(int)
    return df


def generate_small_dedupe_data():
    """Original ~27 synthetic records for deduplication testing."""
    records = [
        {"firstname": "MICHAEL", "lastname": "JOHNSON", "pid1": 123456789, "pid2": 1001,
         "region": "1", "dobyy": 1985, "dobmm": 3, "dobdd": 15, "suffix": np.nan,
         "recordid": "R001", "recordtype": "TYPE2",
         "dob": datetime(1985, 3, 15), "dt_record": datetime(2020, 1, 10)},
        {"firstname": "MICHEAL", "lastname": "JOHNSON", "pid1": 123456789, "pid2": 1001,
         "region": "1", "dobyy": 1985, "dobmm": 3, "dobdd": 15, "suffix": np.nan,
         "recordid": "R002", "recordtype": "TYPE2",
         "dob": datetime(1985, 3, 15), "dt_record": datetime(2020, 6, 20)},
        {"firstname": "MICHAEL", "lastname": "JOHNSEN", "pid1": 123456789, "pid2": np.nan,
         "region": "2", "dobyy": 1985, "dobmm": 3, "dobdd": 15, "suffix": np.nan,
         "recordid": "R003", "recordtype": "TYPE1",
         "dob": datetime(1985, 3, 15), "dt_record": datetime(2021, 2, 14)},
        {"firstname": "MICHAEL", "lastname": "JOHNSON", "pid1": 123456789, "pid2": 1001,
         "region": "2", "dobyy": 1985, "dobmm": 3, "dobdd": 15, "suffix": np.nan,
         "recordid": "R004", "recordtype": "TYPE2",
         "dob": datetime(1985, 3, 15), "dt_record": datetime(2021, 9, 1)},
        {"firstname": "SARAH", "lastname": "WILLIAMS", "pid1": 234567890, "pid2": 2001,
         "region": "5", "dobyy": 1990, "dobmm": 7, "dobdd": 22, "suffix": np.nan,
         "recordid": "R005", "recordtype": "TYPE1",
         "dob": datetime(1990, 7, 22), "dt_record": datetime(2019, 11, 5)},
        {"firstname": "SARA", "lastname": "WILLIAMS", "pid1": 234567890, "pid2": 2001,
         "region": "5", "dobyy": 1990, "dobmm": 7, "dobdd": 22, "suffix": np.nan,
         "recordid": "R006", "recordtype": "TYPE1",
         "dob": datetime(1990, 7, 22), "dt_record": datetime(2020, 3, 18)},
        {"firstname": "WILLIAMS", "lastname": "SARAH", "pid1": np.nan, "pid2": 2001,
         "region": "5", "dobyy": 1990, "dobmm": 7, "dobdd": 22, "suffix": np.nan,
         "recordid": "R007", "recordtype": "TYPE2",
         "dob": datetime(1990, 7, 22), "dt_record": datetime(2020, 8, 30)},
        {"firstname": "JAMES", "lastname": "SMITH", "pid1": 345678901, "pid2": 3001,
         "region": "10", "dobyy": 1988, "dobmm": 11, "dobdd": 5, "suffix": np.nan,
         "recordid": "R008", "recordtype": "TYPE1",
         "dob": datetime(1988, 11, 5), "dt_record": datetime(2020, 5, 1)},
        {"firstname": "JOHN", "lastname": "SMITH", "pid1": 345678902, "pid2": 3002,
         "region": "10", "dobyy": 1990, "dobmm": 2, "dobdd": 14, "suffix": np.nan,
         "recordid": "R008", "recordtype": "TYPE1",
         "dob": datetime(1990, 2, 14), "dt_record": datetime(2020, 5, 1)},
        {"firstname": "JAMES", "lastname": "SMITH", "pid1": 345678901, "pid2": 3001,
         "region": "10", "dobyy": 1988, "dobmm": 11, "dobdd": 5, "suffix": np.nan,
         "recordid": "R009", "recordtype": "TYPE2",
         "dob": datetime(1988, 11, 5), "dt_record": datetime(2021, 1, 15)},
        {"firstname": "EMMA", "lastname": "DAVIS", "pid1": 456789012, "pid2": 4001,
         "region": "7", "dobyy": 1995, "dobmm": 6, "dobdd": 10, "suffix": np.nan,
         "recordid": "R010", "recordtype": "TYPE2",
         "dob": datetime(1995, 6, 10), "dt_record": datetime(2020, 4, 20)},
        {"firstname": "OLIVIA", "lastname": "DAVIS", "pid1": 456789013, "pid2": 4002,
         "region": "7", "dobyy": 1997, "dobmm": 8, "dobdd": 25, "suffix": np.nan,
         "recordid": "R011", "recordtype": "TYPE2",
         "dob": datetime(1997, 8, 25), "dt_record": datetime(2020, 4, 20)},
        {"firstname": "EMMA", "lastname": "DAVIS", "pid1": 456789012, "pid2": 4001,
         "region": "7", "dobyy": 1995, "dobmm": 6, "dobdd": 10, "suffix": np.nan,
         "recordid": "R012", "recordtype": "TYPE2",
         "dob": datetime(1995, 6, 10), "dt_record": datetime(2021, 3, 5)},
        {"firstname": "ROBERT", "lastname": "BROWN", "pid1": 567890123, "pid2": 5001,
         "region": "3", "dobyy": 1975, "dobmm": 1, "dobdd": 8, "suffix": 1,
         "recordid": "R013", "recordtype": "TYPE2",
         "dob": datetime(1975, 1, 8), "dt_record": datetime(2019, 7, 12)},
        {"firstname": "JENNIFER", "lastname": "GARCIA", "pid1": 678901234, "pid2": 6001,
         "region": "12", "dobyy": 1982, "dobmm": 9, "dobdd": 30, "suffix": np.nan,
         "recordid": "R014", "recordtype": "TYPE1",
         "dob": datetime(1982, 9, 30), "dt_record": datetime(2020, 2, 28)},
        {"firstname": "WILLIAM", "lastname": "MARTINEZ", "pid1": 789012345, "pid2": 7001,
         "region": "8", "dobyy": 2000, "dobmm": 12, "dobdd": 1, "suffix": 2,
         "recordid": "R015", "recordtype": "TYPE2",
         "dob": datetime(2000, 12, 1), "dt_record": datetime(2021, 5, 10)},
        {"firstname": "DAVID", "lastname": "ANDERSON", "pid1": 890123456, "pid2": 8001,
         "region": "15", "dobyy": 1993, "dobmm": 4, "dobdd": 17, "suffix": np.nan,
         "recordid": "R016", "recordtype": "TYPE1",
         "dob": datetime(1993, 4, 17), "dt_record": datetime(2020, 9, 14)},
        {"firstname": "MARIA", "lastname": "HERNANDEZ", "pid1": 901234567, "pid2": 9001,
         "region": "20", "dobyy": 1978, "dobmm": 5, "dobdd": 23, "suffix": np.nan,
         "recordid": "R017", "recordtype": "TYPE2",
         "dob": datetime(1978, 5, 23), "dt_record": datetime(2021, 7, 22)},
        {"firstname": "ROBERT", "lastname": "BROWN", "pid1": 567890123, "pid2": 5001,
         "region": "3", "dobyy": 1975, "dobmm": 1, "dobdd": 8, "suffix": 1,
         "recordid": "R018", "recordtype": "TYPE1",
         "dob": datetime(1975, 1, 8), "dt_record": datetime(2021, 1, 3)},
        {"firstname": "LISA", "lastname": "TAYLOR", "pid1": 112233445, "pid2": 10001,
         "region": "1", "dobyy": 1987, "dobmm": 3, "dobdd": 20, "suffix": np.nan,
         "recordid": "R019", "recordtype": "TYPE2",
         "dob": datetime(1987, 3, 20), "dt_record": datetime(2020, 11, 11)},
        {"firstname": "LISA", "lastname": "TAYLOR", "pid1": 112233445, "pid2": 10001,
         "region": "1", "dobyy": 1987, "dobmm": 3, "dobdd": 20, "suffix": np.nan,
         "recordid": "R020", "recordtype": "TYPE1",
         "dob": datetime(1987, 3, 20), "dt_record": datetime(2021, 4, 25)},
        {"firstname": "DANIEL", "lastname": "THOMAS", "pid1": 223344556, "pid2": 11001,
         "region": "9", "dobyy": 1992, "dobmm": 10, "dobdd": 12, "suffix": np.nan,
         "recordid": "R021", "recordtype": "TYPE2",
         "dob": datetime(1992, 10, 12), "dt_record": datetime(2020, 6, 30)},
        {"firstname": "DANEIL", "lastname": "THOMAS", "pid1": 223344556, "pid2": np.nan,
         "region": "9", "dobyy": 1992, "dobmm": 10, "dobdd": 12, "suffix": np.nan,
         "recordid": "R022", "recordtype": "TYPE1",
         "dob": datetime(1992, 10, 12), "dt_record": datetime(2021, 2, 18)},
        {"firstname": "PATRICIA", "lastname": "WILSON", "pid1": 334455667, "pid2": 12001,
         "region": "4", "dobyy": 1980, "dobmm": 8, "dobdd": 7, "suffix": np.nan,
         "recordid": "R023", "recordtype": "TYPE2",
         "dob": datetime(1980, 8, 7), "dt_record": datetime(2019, 12, 20)},
        {"firstname": "CHRISTOPHER", "lastname": "LEE", "pid1": 445566778, "pid2": 13001,
         "region": "6", "dobyy": 1998, "dobmm": 2, "dobdd": 28, "suffix": np.nan,
         "recordid": "R024", "recordtype": "TYPE1",
         "dob": datetime(1998, 2, 28), "dt_record": datetime(2020, 7, 7)},
        {"firstname": "JESSICA", "lastname": "MOORE", "pid1": 556677889, "pid2": 14001,
         "region": "11", "dobyy": 1983, "dobmm": 11, "dobdd": 14, "suffix": np.nan,
         "recordid": "R025", "recordtype": "TYPE2",
         "dob": datetime(1983, 11, 14), "dt_record": datetime(2021, 6, 1)},
        {"firstname": "THOMAS", "lastname": "DANIEL", "pid1": np.nan, "pid2": 11001,
         "region": "9", "dobyy": 1992, "dobmm": 10, "dobdd": 12, "suffix": np.nan,
         "recordid": "R026", "recordtype": "TYPE2",
         "dob": datetime(1992, 10, 12), "dt_record": datetime(2021, 8, 15)},
    ]
    df = pd.DataFrame(records)
    df['dob'] = pd.to_datetime(df['dob'])
    df['dt_record'] = pd.to_datetime(df['dt_record'])
    for col in ['dobyy', 'dobmm', 'dobdd']:
        df[col] = df[col].astype(float)
    return _augment_small(df, seed=1)


def generate_small_match_data():
    """Original 20+20 synthetic records for matching."""
    dfA_records = [
        {"firstname": "MICHAEL", "lastname": "JOHNSON", "pid1": 123456789, "pid2": 1001,
         "region": "1", "dobyy": 1985, "dobmm": 3, "dobdd": 15, "suffix": np.nan,
         "recordid": "RA01", "recordtype": "TYPE2",
         "dob": datetime(1985, 3, 15), "dt_record": datetime(2020, 1, 10)},
        {"firstname": "SARAH", "lastname": "WILLIAMS", "pid1": 234567890, "pid2": 2001,
         "region": "5", "dobyy": 1990, "dobmm": 7, "dobdd": 22, "suffix": np.nan,
         "recordid": "RA02", "recordtype": "TYPE1",
         "dob": datetime(1990, 7, 22), "dt_record": datetime(2019, 11, 5)},
        {"firstname": "JAMES", "lastname": "SMITH", "pid1": 345678901, "pid2": 3001,
         "region": "10", "dobyy": 1988, "dobmm": 11, "dobdd": 5, "suffix": np.nan,
         "recordid": "RA03", "recordtype": "TYPE1",
         "dob": datetime(1988, 11, 5), "dt_record": datetime(2020, 5, 1)},
        {"firstname": "EMMA", "lastname": "DAVIS", "pid1": 456789012, "pid2": 4001,
         "region": "7", "dobyy": 1995, "dobmm": 6, "dobdd": 10, "suffix": np.nan,
         "recordid": "RA04", "recordtype": "TYPE2",
         "dob": datetime(1995, 6, 10), "dt_record": datetime(2020, 4, 20)},
        {"firstname": "ROBERT", "lastname": "BROWN", "pid1": 567890123, "pid2": 5001,
         "region": "3", "dobyy": 1975, "dobmm": 1, "dobdd": 8, "suffix": 1,
         "recordid": "RA05", "recordtype": "TYPE2",
         "dob": datetime(1975, 1, 8), "dt_record": datetime(2019, 7, 12)},
        {"firstname": "JENNIFER", "lastname": "GARCIA", "pid1": 678901234, "pid2": 6001,
         "region": "12", "dobyy": 1982, "dobmm": 9, "dobdd": 30, "suffix": np.nan,
         "recordid": "RA06", "recordtype": "TYPE1",
         "dob": datetime(1982, 9, 30), "dt_record": datetime(2020, 2, 28)},
        {"firstname": "WILLIAM", "lastname": "MARTINEZ", "pid1": 789012345, "pid2": 7001,
         "region": "8", "dobyy": 2000, "dobmm": 12, "dobdd": 1, "suffix": 2,
         "recordid": "RA07", "recordtype": "TYPE2",
         "dob": datetime(2000, 12, 1), "dt_record": datetime(2021, 5, 10)},
        {"firstname": "DAVID", "lastname": "ANDERSON", "pid1": 890123456, "pid2": 8001,
         "region": "15", "dobyy": 1993, "dobmm": 4, "dobdd": 17, "suffix": np.nan,
         "recordid": "RA08", "recordtype": "TYPE1",
         "dob": datetime(1993, 4, 17), "dt_record": datetime(2020, 9, 14)},
        {"firstname": "MARIA", "lastname": "HERNANDEZ", "pid1": 901234567, "pid2": 9001,
         "region": "20", "dobyy": 1978, "dobmm": 5, "dobdd": 23, "suffix": np.nan,
         "recordid": "RA09", "recordtype": "TYPE2",
         "dob": datetime(1978, 5, 23), "dt_record": datetime(2021, 7, 22)},
        {"firstname": "LISA", "lastname": "TAYLOR", "pid1": 112233445, "pid2": 10001,
         "region": "1", "dobyy": 1987, "dobmm": 3, "dobdd": 20, "suffix": np.nan,
         "recordid": "RA10", "recordtype": "TYPE2",
         "dob": datetime(1987, 3, 20), "dt_record": datetime(2020, 11, 11)},
        {"firstname": "DANIEL", "lastname": "THOMAS", "pid1": 223344556, "pid2": 11001,
         "region": "9", "dobyy": 1992, "dobmm": 10, "dobdd": 12, "suffix": np.nan,
         "recordid": "RA11", "recordtype": "TYPE2",
         "dob": datetime(1992, 10, 12), "dt_record": datetime(2020, 6, 30)},
        {"firstname": "PATRICIA", "lastname": "WILSON", "pid1": 334455667, "pid2": 12001,
         "region": "4", "dobyy": 1980, "dobmm": 8, "dobdd": 7, "suffix": np.nan,
         "recordid": "RA12", "recordtype": "TYPE2",
         "dob": datetime(1980, 8, 7), "dt_record": datetime(2019, 12, 20)},
        {"firstname": "CHRISTOPHER", "lastname": "LEE", "pid1": 445566778, "pid2": 13001,
         "region": "6", "dobyy": 1998, "dobmm": 2, "dobdd": 28, "suffix": np.nan,
         "recordid": "RA13", "recordtype": "TYPE1",
         "dob": datetime(1998, 2, 28), "dt_record": datetime(2020, 7, 7)},
        {"firstname": "JESSICA", "lastname": "MOORE", "pid1": 556677889, "pid2": 14001,
         "region": "11", "dobyy": 1983, "dobmm": 11, "dobdd": 14, "suffix": np.nan,
         "recordid": "RA14", "recordtype": "TYPE2",
         "dob": datetime(1983, 11, 14), "dt_record": datetime(2021, 6, 1)},
        {"firstname": "KEVIN", "lastname": "CLARK", "pid1": 667788990, "pid2": 15001,
         "region": "2", "dobyy": 1991, "dobmm": 1, "dobdd": 25, "suffix": np.nan,
         "recordid": "RA15", "recordtype": "TYPE1",
         "dob": datetime(1991, 1, 25), "dt_record": datetime(2020, 8, 15)},
        {"firstname": "NANCY", "lastname": "LEWIS", "pid1": 778899001, "pid2": 16001,
         "region": "14", "dobyy": 1986, "dobmm": 6, "dobdd": 3, "suffix": np.nan,
         "recordid": "RA16", "recordtype": "TYPE2",
         "dob": datetime(1986, 6, 3), "dt_record": datetime(2021, 2, 10)},
        {"firstname": "MARK", "lastname": "ROBINSON", "pid1": 889900112, "pid2": 17001,
         "region": "18", "dobyy": 1979, "dobmm": 9, "dobdd": 19, "suffix": np.nan,
         "recordid": "RA17", "recordtype": "TYPE1",
         "dob": datetime(1979, 9, 19), "dt_record": datetime(2020, 4, 1)},
        {"firstname": "SUSAN", "lastname": "WALKER", "pid1": 990011223, "pid2": 18001,
         "region": "13", "dobyy": 1994, "dobmm": 12, "dobdd": 8, "suffix": np.nan,
         "recordid": "RA18", "recordtype": "TYPE2",
         "dob": datetime(1994, 12, 8), "dt_record": datetime(2021, 9, 20)},
        {"firstname": "THOMAS", "lastname": "HALL", "pid1": 101112131, "pid2": 19001,
         "region": "16", "dobyy": 1977, "dobmm": 7, "dobdd": 11, "suffix": np.nan,
         "recordid": "RA19", "recordtype": "TYPE1",
         "dob": datetime(1977, 7, 11), "dt_record": datetime(2020, 12, 5)},
        {"firstname": "KAREN", "lastname": "ALLEN", "pid1": 121314151, "pid2": 20001,
         "region": "19", "dobyy": 1989, "dobmm": 4, "dobdd": 29, "suffix": np.nan,
         "recordid": "RA20", "recordtype": "TYPE2",
         "dob": datetime(1989, 4, 29), "dt_record": datetime(2021, 4, 14)},
    ]
    dfB_records = [
        {"firstname": "MICHEAL", "lastname": "JOHNSON", "pid1": 123456789, "pid2": 1001,
         "region": "1", "dobyy": 1985, "dobmm": 3, "dobdd": 15, "suffix": np.nan,
         "recordid": "RB01", "recordtype": "TYPE2",
         "dob": datetime(1985, 3, 15), "dt_record": datetime(2021, 3, 22)},
        {"firstname": "SARA", "lastname": "WILLIAMS", "pid1": 234567890, "pid2": 2001,
         "region": "5", "dobyy": 1990, "dobmm": 7, "dobdd": 22, "suffix": np.nan,
         "recordid": "RB02", "recordtype": "TYPE1",
         "dob": datetime(1990, 7, 22), "dt_record": datetime(2021, 1, 15)},
        {"firstname": "SMITH", "lastname": "JAMES", "pid1": 345678901, "pid2": 3001,
         "region": "10", "dobyy": 1988, "dobmm": 11, "dobdd": 5, "suffix": np.nan,
         "recordid": "RB03", "recordtype": "TYPE1",
         "dob": datetime(1988, 11, 5), "dt_record": datetime(2021, 6, 10)},
        {"firstname": "EMMA", "lastname": "DAVIS", "pid1": 456789012, "pid2": 4001,
         "region": "7", "dobyy": 1995, "dobmm": 6, "dobdd": 10, "suffix": np.nan,
         "recordid": "RB04", "recordtype": "TYPE2",
         "dob": datetime(1995, 6, 10), "dt_record": datetime(2021, 7, 18)},
        {"firstname": "ROBERT", "lastname": "BROWN", "pid1": 567890123, "pid2": 5001,
         "region": "3", "dobyy": 1975, "dobmm": 1, "dobdd": 8, "suffix": 1,
         "recordid": "RB05", "recordtype": "TYPE2",
         "dob": datetime(1975, 1, 8), "dt_record": datetime(2021, 4, 2)},
        {"firstname": "JENNIFER", "lastname": "GRACIA", "pid1": 678901234, "pid2": 6001,
         "region": "12", "dobyy": 1982, "dobmm": 9, "dobdd": 30, "suffix": np.nan,
         "recordid": "RB06", "recordtype": "TYPE1",
         "dob": datetime(1982, 9, 30), "dt_record": datetime(2021, 5, 20)},
        {"firstname": "WILLIAM", "lastname": "MARTINEZ", "pid1": 789012345, "pid2": 7001,
         "region": "8", "dobyy": 2000, "dobmm": 12, "dobdd": 1, "suffix": 2,
         "recordid": "RB07", "recordtype": "TYPE2",
         "dob": datetime(2000, 12, 1), "dt_record": datetime(2022, 1, 8)},
        {"firstname": "DAVID", "lastname": "ANDERSON", "pid1": np.nan, "pid2": 8001,
         "region": "15", "dobyy": 1993, "dobmm": 4, "dobdd": 17, "suffix": np.nan,
         "recordid": "RB08", "recordtype": "TYPE1",
         "dob": datetime(1993, 4, 17), "dt_record": datetime(2021, 11, 30)},
        {"firstname": "MARIA", "lastname": "HERNANDEZ", "pid1": 901234567, "pid2": np.nan,
         "region": "20", "dobyy": 1978, "dobmm": 5, "dobdd": 23, "suffix": np.nan,
         "recordid": "RB09", "recordtype": "TYPE2",
         "dob": datetime(1978, 5, 23), "dt_record": datetime(2022, 2, 14)},
        {"firstname": "LISA", "lastname": "TAYLOR", "pid1": 112233445, "pid2": 10001,
         "region": "1", "dobyy": 1987, "dobmm": 3, "dobdd": 20, "suffix": np.nan,
         "recordid": "RB10", "recordtype": "TYPE2",
         "dob": datetime(1987, 3, 20), "dt_record": datetime(2022, 3, 1)},
        {"firstname": "ANTHONY", "lastname": "YOUNG", "pid1": 161718192, "pid2": 21001,
         "region": "3", "dobyy": 1996, "dobmm": 3, "dobdd": 5, "suffix": np.nan,
         "recordid": "RB11", "recordtype": "TYPE1",
         "dob": datetime(1996, 3, 5), "dt_record": datetime(2021, 8, 25)},
        {"firstname": "BETTY", "lastname": "KING", "pid1": 171819202, "pid2": 22001,
         "region": "7", "dobyy": 1984, "dobmm": 10, "dobdd": 18, "suffix": np.nan,
         "recordid": "RB12", "recordtype": "TYPE2",
         "dob": datetime(1984, 10, 18), "dt_record": datetime(2021, 9, 30)},
        {"firstname": "CHARLES", "lastname": "WRIGHT", "pid1": 181920213, "pid2": 23001,
         "region": "11", "dobyy": 2001, "dobmm": 5, "dobdd": 7, "suffix": np.nan,
         "recordid": "RB13", "recordtype": "TYPE1",
         "dob": datetime(2001, 5, 7), "dt_record": datetime(2021, 10, 14)},
        {"firstname": "DOROTHY", "lastname": "LOPEZ", "pid1": 192021224, "pid2": 24001,
         "region": "15", "dobyy": 1976, "dobmm": 8, "dobdd": 22, "suffix": np.nan,
         "recordid": "RB14", "recordtype": "TYPE2",
         "dob": datetime(1976, 8, 22), "dt_record": datetime(2022, 1, 20)},
        {"firstname": "EDWARD", "lastname": "HILL", "pid1": 202122235, "pid2": 25001,
         "region": "2", "dobyy": 1999, "dobmm": 1, "dobdd": 30, "suffix": np.nan,
         "recordid": "RB15", "recordtype": "TYPE1",
         "dob": datetime(1999, 1, 30), "dt_record": datetime(2021, 12, 10)},
        {"firstname": "FRANCES", "lastname": "SCOTT", "pid1": 212223246, "pid2": 26001,
         "region": "9", "dobyy": 1981, "dobmm": 11, "dobdd": 3, "suffix": np.nan,
         "recordid": "RB16", "recordtype": "TYPE2",
         "dob": datetime(1981, 11, 3), "dt_record": datetime(2022, 2, 28)},
        {"firstname": "GEORGE", "lastname": "GREEN", "pid1": 222324257, "pid2": 27001,
         "region": "5", "dobyy": 1997, "dobmm": 7, "dobdd": 15, "suffix": np.nan,
         "recordid": "RB17", "recordtype": "TYPE1",
         "dob": datetime(1997, 7, 15), "dt_record": datetime(2022, 3, 15)},
        {"firstname": "HELEN", "lastname": "BAKER", "pid1": 232425268, "pid2": 28001,
         "region": "18", "dobyy": 1988, "dobmm": 2, "dobdd": 10, "suffix": np.nan,
         "recordid": "RB18", "recordtype": "TYPE2",
         "dob": datetime(1988, 2, 10), "dt_record": datetime(2022, 4, 1)},
        {"firstname": "IVAN", "lastname": "ADAMS", "pid1": 242526279, "pid2": 29001,
         "region": "13", "dobyy": 1990, "dobmm": 9, "dobdd": 28, "suffix": np.nan,
         "recordid": "RB19", "recordtype": "TYPE1",
         "dob": datetime(1990, 9, 28), "dt_record": datetime(2022, 4, 20)},
        {"firstname": "JULIA", "lastname": "NELSON", "pid1": 252627280, "pid2": 30001,
         "region": "16", "dobyy": 1985, "dobmm": 12, "dobdd": 14, "suffix": np.nan,
         "recordid": "RB20", "recordtype": "TYPE2",
         "dob": datetime(1985, 12, 14), "dt_record": datetime(2022, 5, 5)},
    ]
    dfA = pd.DataFrame(dfA_records)
    dfB = pd.DataFrame(dfB_records)
    for df in [dfA, dfB]:
        df['dob'] = pd.to_datetime(df['dob'])
        df['dt_record'] = pd.to_datetime(df['dt_record'])
        for col in ['dobyy', 'dobmm', 'dobdd']:
            df[col] = df[col].astype(float)
    return _augment_small(dfA, seed=2), _augment_small(dfB, seed=3)


# ============================================================================
# Pipeline Configuration
# ============================================================================

fields = {
    'firstname': 'firstname',
    'lastname': 'lastname',
    'suffix': 'suffix',
    'pid1': 'pid1',
    'pid2': 'pid2',
    'region': 'region',
    'dobyy': 'dobyy',
    'dobmm': 'dobmm',
    'dobdd': 'dobdd',
}

indexer = ['firstname', 'lastname', 'pid1']
transposed = [['firstname', 'lastname']]
# Binary indicators slot into features[0] -- the existing exact-match,
# Fellegi-Sunter-weighted feature path -- so they need no new comparison code.
features = [
    ['pid1', 'region'] + DEFAULT_BINARY_FIELDS,
    [[]],
    ['firstname', 'lastname'],
]


# ============================================================================
# Dedupe Pipeline
# ============================================================================

def run_dedupe_pipeline(args, profiler):
    print("=" * 70)
    print("DEDUPLICATION PIPELINE")
    print("=" * 70)
    verbose = args.n_records <= 100

    # Step 1: Generate data
    profiler.start("Generate (dedupe)")
    print("\n--- Step 1: Generate synthetic data ---")
    if args.small:
        df = generate_small_dedupe_data()
    else:
        df = generate_dedupe_data(
            n_records=args.n_records,
            dedupe_rate=args.dedupe_rate,
            defeat_rate=args.defeat_rate,
            seed=args.seed,
        )
    print(f"Generated {len(df)} records")
    if verbose:
        print(df[['firstname', 'lastname', 'pid1', 'suffix', 'region']].to_string())
    else:
        print(df[['firstname', 'lastname', 'pid1', 'suffix', 'region']].head(10).to_string())
        print(f"  ... ({len(df)} total rows)")
    if args.save_data:
        os.makedirs(DATA_DIR, exist_ok=True)
        out = os.path.join(DATA_DIR, "dedupe_data.csv")
        df.to_csv(out, index=False)
        print(f"Saved to {out}")
    profiler.stop(f"{len(df)} records")

    # Step 2: Standardize
    profiler.start("Standardize (dedupe)")
    print("\n--- Step 2: Standardize ---")
    df = crosswalk.shared.preprocessing.standardize(fields, df)
    valid_df = df[df['valid'] == 1].copy()
    print(f"Valid records: {len(valid_df)} / {len(df)}")
    invalid = df[df['valid'] == 0]
    if len(invalid) > 0:
        print(f"Invalid records removed: {len(invalid)}")
        if verbose:
            print(invalid[['firstname', 'lastname']].to_string())
    profiler.stop(f"{len(valid_df)} valid")

    # Step 3: Indexing (blocking)
    profiler.start("Indexing (dedupe)")
    print("\n--- Step 3: Indexing (blocking) ---")
    cpairs = crosswalk.cpu.indexing.dedupe(indexer, transposed, valid_df)
    print(f"Candidate pairs: {len(cpairs):,}")
    profiler.stop(f"{len(cpairs):,} pairs")

    if len(cpairs) == 0:
        print("No candidate pairs found. Check blocking fields.")
        return None, None, None

    # Step 4: Comparing
    profiler.start("Comparing (dedupe)")
    print("\n--- Step 4: Comparing (Jaro-Winkler + exact) ---")
    compare_features = [features[0], features[1]]
    bmatrix = crosswalk.cpu.comparing.dedupe(cpairs, compare_features, valid_df)
    print(f"Boolean matrix shape: {bmatrix.shape}")
    if verbose:
        print(bmatrix.head(10).to_string())
    profiler.stop(f"{bmatrix.shape[0]:,} x {bmatrix.shape[1]}")

    # Step 5: Classifier (Fellegi-Sunter)
    profiler.start("Classifier (dedupe)")
    print("\n--- Step 5: Classifier (Fellegi-Sunter weights) ---")
    vector = crosswalk.cpu.classifier.dedupe(bmatrix, features, valid_df)
    weight_cols = [c for c in vector.columns if isinstance(c, str) and c.startswith('w_')]
    print(f"Weight columns: {weight_cols}")
    print(f"\nfs_score distribution:")
    print(vector['fs_score'].describe().to_string())
    if verbose:
        print(f"\nTop 15 pairs by fs_score:")
        top = vector.nlargest(15, 'fs_score')
        display_cols = weight_cols + ['fs_score']
        print(top[display_cols].to_string())
    profiler.stop(f"{len(vector):,} scored")

    # Step 6: Tiebreak
    profiler.start("Tiebreak (dedupe)")
    print("\n--- Step 6: Tiebreak ---")
    result, broken = crosswalk.cpu.classifier.tiebreak(vector, valid_df)
    print(f"Pairs after tiebreak: {len(result)}")
    print(f"Ties broken (removed): {len(broken)}")
    if verbose and len(broken) > 0:
        print("Broken ties:")
        print(broken[['firstname_x', 'lastname_x', 'firstname_y', 'lastname_y']].to_string())
    profiler.stop(f"{len(broken)} removed")

    # Step 7: Components (entity assignment)
    profiler.start("Components (dedupe)")
    print("\n--- Step 7: Components (entity ID assignment) ---")
    threshold = 0
    links = result[result['fs_score'] >= threshold]
    print(f"Links with fs_score >= {threshold}: {len(links)}")

    famid = None
    if len(links) > 0:
        famid = crosswalk.cpu.classifier.components(
            valid_df, links['fs_score'], 'entity_id', 'index', 'region'
        )
        print(f"\nUnique entities identified: {famid['entity_id'].nunique()}")
        if verbose:
            print(famid[['firstname', 'lastname', 'pid1', 'entity_id']].to_string())
        else:
            print(famid[['firstname', 'lastname', 'pid1', 'entity_id']].head(20).to_string())
            print(f"  ... ({len(famid)} total rows)")
    else:
        print("No links above threshold.")
    profiler.stop(f"{famid['entity_id'].nunique() if famid is not None else 0} entities")

    return vector, result, broken


# ============================================================================
# Match Pipeline
# ============================================================================

def run_match_pipeline(args, profiler):
    print("\n" + "=" * 70)
    print("MATCHING PIPELINE")
    print("=" * 70)
    verbose = args.n_records <= 100

    # Step 1: Generate data
    profiler.start("Generate (match)")
    print("\n--- Step 1: Generate synthetic data ---")
    if args.small:
        dfA, dfB = generate_small_match_data()
    else:
        dfA, dfB = generate_match_data(
            n_records=args.n_records,
            match_rate=args.match_rate,
            defeat_rate=args.defeat_rate,
            seed=args.seed,
        )
    print(f"dfA: {len(dfA)} records, dfB: {len(dfB)} records")
    if args.save_data:
        os.makedirs(DATA_DIR, exist_ok=True)
        out_a = os.path.join(DATA_DIR, "match_dfA.csv")
        out_b = os.path.join(DATA_DIR, "match_dfB.csv")
        dfA.to_csv(out_a, index=False)
        dfB.to_csv(out_b, index=False)
        print(f"Saved to {out_a} and {out_b}")
    profiler.stop(f"{len(dfA)}+{len(dfB)} records")

    # Step 2: Standardize both
    profiler.start("Standardize (match)")
    print("\n--- Step 2: Standardize ---")
    dfA = crosswalk.shared.preprocessing.standardize(fields, dfA)
    dfB = crosswalk.shared.preprocessing.standardize(fields, dfB)
    valid_A = dfA[dfA['valid'] == 1].copy()
    valid_B = dfB[dfB['valid'] == 1].copy()
    print(f"Valid: dfA={len(valid_A)}, dfB={len(valid_B)}")
    profiler.stop(f"{len(valid_A)}+{len(valid_B)} valid")

    # Step 3: Indexing
    profiler.start("Indexing (match)")
    print("\n--- Step 3: Indexing (blocking) ---")
    cpairs = crosswalk.cpu.indexing.match(indexer, transposed, valid_A, valid_B)
    print(f"Candidate pairs: {len(cpairs):,}")
    profiler.stop(f"{len(cpairs):,} pairs")

    if len(cpairs) == 0:
        print("No candidate pairs found. Check blocking fields.")
        return None

    # Step 4: Comparing
    profiler.start("Comparing (match)")
    print("\n--- Step 4: Comparing ---")
    compare_features = [features[0], features[1]]
    bmatrix = crosswalk.cpu.comparing.match(cpairs, compare_features, valid_A, valid_B)
    print(f"Boolean matrix shape: {bmatrix.shape}")
    if verbose:
        print(bmatrix.head(10).to_string())
    profiler.stop(f"{bmatrix.shape[0]:,} x {bmatrix.shape[1]}")

    # Step 5: Classifier
    profiler.start("Classifier (match)")
    print("\n--- Step 5: Classifier (Fellegi-Sunter weights) ---")
    vector = crosswalk.cpu.classifier.match(bmatrix, features, valid_A, valid_B)
    weight_cols = [c for c in vector.columns if isinstance(c, str) and c.startswith('w_')]
    print(f"\nfs_score distribution:")
    print(vector['fs_score'].describe().to_string())
    if verbose:
        print(f"\nTop 15 pairs by fs_score:")
        top = vector.nlargest(15, 'fs_score')
        display_cols = weight_cols + ['fs_score']
        print(top[display_cols].to_string())
    profiler.stop(f"{len(vector):,} scored")

    return vector


# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Crosswalk Record Linkage Pipeline - End-to-End Test"
    )
    parser.add_argument(
        "--n-records", type=int, default=10000,
        help="Number of records per dataset (default: 10000)"
    )
    parser.add_argument(
        "--match-rate", type=float, default=0.85,
        help="Fraction of dfB that matches dfA (default: 0.85)"
    )
    parser.add_argument(
        "--dedupe-rate", type=float, default=0.30,
        help="Fraction of dedupe records that are duplicates (default: 0.30)"
    )
    parser.add_argument(
        "--defeat-rate", type=float, default=0.05,
        help="Fraction of true matches that defeat blocking (invisible false "
             "negatives), for measuring true recall (default: 0.05)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--profile", action="store_true",
        help="Enable CPU/memory profiling (cProfile + tracemalloc)"
    )
    parser.add_argument(
        "--small", action="store_true",
        help="Use original 27-record hardcoded data for quick sanity check"
    )
    parser.add_argument(
        "--save-data", action="store_true",
        help="Save generated datasets to CSV files in data/ directory"
    )
    parser.add_argument(
        "--dedupe-only", action="store_true",
        help="Run only the deduplication pipeline"
    )
    parser.add_argument(
        "--match-only", action="store_true",
        help="Run only the matching pipeline"
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.small:
        args.n_records = 27  # for display/verbosity logic
        mode_str = "small (hardcoded 27-record data)"
    else:
        mode_str = f"{args.n_records:,} records, match_rate={args.match_rate}, dedupe_rate={args.dedupe_rate}"

    print("Crosswalk Record Linkage - End-to-End Pipeline Test")
    print(f"Mode: {mode_str}")
    print(f"Seed: {args.seed}")
    print(f"Profiling: {'ON' if args.profile else 'OFF'}")
    print()

    profiler = StepProfiler(enabled=args.profile)
    run_dedupe = not args.match_only
    run_match = not args.dedupe_only

    if args.profile and run_dedupe:
        print("Running dedupe pipeline under cProfile...")
        dedupe_vector, dedupe_result, broken = dump_cprofile(
            run_dedupe_pipeline, (args, profiler), "profile_dedupe.prof"
        )
    elif run_dedupe:
        dedupe_vector, dedupe_result, broken = run_dedupe_pipeline(args, profiler)

    if args.profile and run_match:
        print("\nRunning match pipeline under cProfile...")
        match_vector = dump_cprofile(
            run_match_pipeline, (args, profiler), "profile_match.prof"
        )
    elif run_match:
        match_vector = run_match_pipeline(args, profiler)

    profiler.summary()

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
