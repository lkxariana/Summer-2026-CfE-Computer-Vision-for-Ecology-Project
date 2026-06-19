"""
02_bin_and_filter.py -- Step 7, Part 2

Bin both datasets at 0.5deg (matching Step 4's original resolution), drop
globally rare species (<10 total observations across all of CONUS,
consistent with Step 4's >=10 observation EDA threshold -- this is a TOTAL
count filter, not a per-bin filter, so it only removes species before the
expensive computation rather than removing sparse bins), then collapse to
per-species, per-bin observation counts and cache the result.

Result of the rare-species filter: plants kept all 50 species (no records
lost). Pollinators dropped from 25,484 to 13,635 species, but lost only
~37K of 21.9M records (0.17%) -- confirming the long tail of rare species
contributed negligible data.
"""

import numpy as np
import pandas as pd
from pathlib import Path

BIN_SIZE = 0.5
MIN_TOTAL_OBS = 10
DATA_DIR = Path("/scratch/ariana.l")


def add_bins(df, bin_size=BIN_SIZE):
    df = df.copy()
    df["lat_bin"] = np.floor(df["lat"] / bin_size) * bin_size
    df["lon_bin"] = np.floor(df["lon"] / bin_size) * bin_size
    return df


def filter_rare_species(plants, pollinators, min_total_obs=MIN_TOTAL_OBS):
    plants = add_bins(plants)
    pollinators = add_bins(pollinators)

    plant_totals = plants["species"].value_counts()
    keep_plants = plant_totals[plant_totals >= min_total_obs].index
    plants = plants[plants["species"].isin(keep_plants)]

    pollinator_totals = pollinators["pollinator_species"].value_counts()
    keep_pollinators = pollinator_totals[pollinator_totals >= min_total_obs].index
    pollinators = pollinators[pollinators["pollinator_species"].isin(keep_pollinators)]

    print(f"Plants kept: {plants['species'].nunique()} species ({len(plants):,} records)")
    print(
        f"Pollinators kept: {pollinators['pollinator_species'].nunique()} species "
        f"({len(pollinators):,} records)"
    )
    return plants, pollinators


def build_and_cache_counts(plants, pollinators):
    plant_counts = (
        plants.groupby(["species", "lat_bin", "lon_bin"]).size().reset_index(name="plant_count")
    )
    pollinator_counts = (
        pollinators.groupby(["pollinator_species", "lat_bin", "lon_bin"])
        .size()
        .reset_index(name="pollinator_count")
    )

    print(f"Plant (species, bin) cells: {len(plant_counts):,}")
    print(f"Pollinator (species, bin) cells: {len(pollinator_counts):,}")

    plant_counts.to_parquet(DATA_DIR / "cache_plant_counts.parquet")
    pollinator_counts.to_parquet(DATA_DIR / "cache_pollinator_counts.parquet")
    print("Cached plant_counts and pollinator_counts.")

    return plant_counts, pollinator_counts


if __name__ == "__main__":
    from importlib import import_module

    mod = import_module("01_load_data")
    plants = mod.load_plants()
    pollinators = mod.load_pollinators_and_cache()

    plants, pollinators = filter_rare_species(plants, pollinators)
    plant_counts, pollinator_counts = build_and_cache_counts(plants, pollinators)
