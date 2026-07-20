"""
filter_and_build_densities.py -- Step 8, Part 3

Apply MIN_OBS=5 to drop sparse (species, bin) cells, then build a normalized
52-week density histogram for every surviving (species, bin) cell, for both
plants and pollinators.
"""

import pandas as pd
import numpy as np

MIN_OBS = 5
N_WEEKS = 52


def apply_min_obs_filter(plants, pollinators, plant_bin_counts, pollinator_bin_counts, min_obs=MIN_OBS):
    plant_bin_counts_filtered = plant_bin_counts[plant_bin_counts["n_obs"] >= min_obs]
    pollinator_bin_counts_filtered = pollinator_bin_counts[
        pollinator_bin_counts["n_obs"] >= min_obs
    ]

    print(
        f"Plant (species, bin) cells retained: {len(plant_bin_counts_filtered):,} "
        f"/ {len(plant_bin_counts):,}"
    )
    print(
        f"Pollinator (species, bin) cells retained: {len(pollinator_bin_counts_filtered):,} "
        f"/ {len(pollinator_bin_counts):,}"
    )

    plant_keep = set(
        zip(
            plant_bin_counts_filtered["species"],
            plant_bin_counts_filtered["lat_bin"],
            plant_bin_counts_filtered["lon_bin"],
        )
    )
    pollinator_keep = set(
        zip(
            pollinator_bin_counts_filtered["pollinator_species"],
            pollinator_bin_counts_filtered["lat_bin"],
            pollinator_bin_counts_filtered["lon_bin"],
        )
    )

    plants = plants.copy()
    pollinators = pollinators.copy()
    plants["_key"] = list(zip(plants["species"], plants["lat_bin"], plants["lon_bin"]))
    pollinators["_key"] = list(
        zip(pollinators["pollinator_species"], pollinators["lat_bin"], pollinators["lon_bin"])
    )

    plants_filtered = plants[plants["_key"].isin(plant_keep)].drop(columns="_key")
    pollinators_filtered = pollinators[pollinators["_key"].isin(pollinator_keep)].drop(
        columns="_key"
    )

    print(f"Plant observation rows retained: {len(plants_filtered):,} / {len(plants):,}")
    print(
        f"Pollinator observation rows retained: {len(pollinators_filtered):,} "
        f"/ {len(pollinators):,}"
    )

    return plants_filtered, pollinators_filtered, plant_bin_counts_filtered, pollinator_bin_counts_filtered


def weekly_density(doy_values, n_weeks=N_WEEKS):
    """Build a normalized 52-week histogram from a series of day-of-year values."""
    weeks = np.clip((doy_values - 1) // 7, 0, n_weeks - 1).astype(int)
    counts = np.bincount(weeks, minlength=n_weeks)
    total = counts.sum()
    return counts / total if total > 0 else counts


def build_density_dicts(plants_filtered, pollinators_filtered):
    plant_density = {
        key: weekly_density(group["doy"].values)
        for key, group in plants_filtered.groupby(["species", "lat_bin", "lon_bin"])
    }
    pollinator_density = {
        key: weekly_density(group["doy"].values)
        for key, group in pollinators_filtered.groupby(["pollinator_species", "lat_bin", "lon_bin"])
    }
    print(f"Plant density entries: {len(plant_density):,}")
    print(f"Pollinator density entries: {len(pollinator_density):,}")
    return plant_density, pollinator_density
