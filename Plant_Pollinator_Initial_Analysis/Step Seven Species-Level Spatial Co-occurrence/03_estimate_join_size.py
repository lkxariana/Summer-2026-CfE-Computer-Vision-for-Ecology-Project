"""
03_estimate_join_size.py -- Step 7, Part 3

Before computing co-occurrence across every possible plant x pollinator
combination, estimate the size of the full cross join (restricted to shared
bins) to decide whether a direct merge is memory-safe.

Result: 2,431 distinct plant bins, all of which are also pollinator bins
(every place with plant data also has pollinator data -- full containment,
no spatial coverage gap). Estimated join size: ~17.4 million rows. Given the
21.9M-row datasets already in memory, this was judged too large to
materialize directly -- see 04_jaccard_overlap.py for the memory-safe
alternative actually used (a per-bin streaming approach with a bounded
min-heap), which was itself superseded by the Jaccard method in this same
script once the streaming min-count approach was found to produce
misleading results (see README).
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path("/scratch/ariana.l")


def estimate_join_size(plant_counts, pollinator_counts):
    plant_bins = set(zip(plant_counts["lat_bin"], plant_counts["lon_bin"]))
    pollinator_bins = set(zip(pollinator_counts["lat_bin"], pollinator_counts["lon_bin"]))
    shared_bins = plant_bins & pollinator_bins

    print(f"Distinct plant bins: {len(plant_bins):,}")
    print(f"Distinct pollinator bins: {len(pollinator_bins):,}")
    print(f"Shared bins: {len(shared_bins):,}")

    plant_per_bin = plant_counts.groupby(["lat_bin", "lon_bin"]).size()
    pollinator_per_bin = pollinator_counts.groupby(["lat_bin", "lon_bin"]).size()
    shared_idx = plant_per_bin.index.intersection(pollinator_per_bin.index)
    estimated_rows = (plant_per_bin.loc[shared_idx] * pollinator_per_bin.loc[shared_idx]).sum()
    print(f"Estimated join row count: {estimated_rows:,}")

    return shared_bins, estimated_rows


if __name__ == "__main__":
    plant_counts = pd.read_parquet(DATA_DIR / "cache_plant_counts.parquet")
    pollinator_counts = pd.read_parquet(DATA_DIR / "cache_pollinator_counts.parquet")
    estimate_join_size(plant_counts, pollinator_counts)
