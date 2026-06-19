"""
bin_and_check_sufficiency.py -- Step 8, Part 2

Spatially bin the filtered plant/pollinator observations at 1.5 degrees
(matching Step 5's original bin size) and check how many (species, bin)
cells would be dropped at various MIN_OBS thresholds before committing to
one. This step exists because percentile statistics alone (e.g. "the 25th
percentile is 17") do not tell you what fraction of cells fall below a given
absolute threshold -- the underlying distribution here is heavily
right-skewed, so threshold-counting and percentile reasoning diverge.

Result of this check (see Step 8 README for the decision):
  MIN_OBS=10 (Step 5's original) drops ~38% of plant cells, ~36% of
    pollinator cells -- too aggressive given the smaller, pre-filtered
    species set used here.
  MIN_OBS=5 drops ~24-25% on both sides -- the threshold used going forward.
"""

import pandas as pd
import numpy as np
from pathlib import Path

BIN_SIZE_TEMPORAL = 1.5


def add_bins(df, bin_size=BIN_SIZE_TEMPORAL):
    df = df.copy()
    df["lat_bin"] = np.floor(df["lat"] / bin_size) * bin_size
    df["lon_bin"] = np.floor(df["lon"] / bin_size) * bin_size
    return df


def bin_and_count(plants, pollinators):
    plants = add_bins(plants)
    pollinators = add_bins(pollinators)

    plant_bin_counts = (
        plants.groupby(["species", "lat_bin", "lon_bin"]).size().reset_index(name="n_obs")
    )
    pollinator_bin_counts = (
        pollinators.groupby(["pollinator_species", "lat_bin", "lon_bin"])
        .size()
        .reset_index(name="n_obs")
    )

    print("Plant (species, bin) observation count distribution:")
    print(plant_bin_counts["n_obs"].describe())
    print()
    print("Pollinator (species, bin) observation count distribution:")
    print(pollinator_bin_counts["n_obs"].describe())

    return plants, pollinators, plant_bin_counts, pollinator_bin_counts


def check_threshold(plant_bin_counts, pollinator_bin_counts, threshold):
    for label, df in [("Plant", plant_bin_counts), ("Pollinator", pollinator_bin_counts)]:
        total = len(df)
        below = (df["n_obs"] < threshold).sum()
        print(
            f"{label}: {below:,} / {total:,} (species, bin) cells below {threshold} obs "
            f"({below / total * 100:.1f}%)"
        )


if __name__ == "__main__":
    # Expects `plants` and `pollinators` from 01_load_and_filter.py
    from importlib import import_module

    mod = import_module("01_load_and_filter")
    jaccard_pairs, target_plants, target_pollinators = mod.load_jaccard_pairs()
    plants = mod.load_plants(target_plants)
    pollinators = mod.load_pollinators_v2(target_pollinators)

    plants, pollinators, plant_bin_counts, pollinator_bin_counts = bin_and_count(
        plants, pollinators
    )

    print("\n--- Checking MIN_OBS=10 ---")
    check_threshold(plant_bin_counts, pollinator_bin_counts, 10)
    print("\n--- Checking MIN_OBS=5 ---")
    check_threshold(plant_bin_counts, pollinator_bin_counts, 5)
