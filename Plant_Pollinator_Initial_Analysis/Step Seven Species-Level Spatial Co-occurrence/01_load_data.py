"""
01_load_data.py -- Step 7, Part 1

Load the raw plant flowering and pollinator observation data for the full
species-level co-occurrence analysis. This is the foundation for the Jaccard
range-overlap method developed in this step, which replaces Step 4's
aggregate-density overlap score (plant_count_in_bin / max) x
(pollinator_count_in_bin / max) with a species-pair-specific measure.

Motivation: Step 4's score has no notion of which plant species and which
pollinator species are actually present in a bin -- a bin can score high
purely from heavy total observation traffic (e.g. a city park with dense
iNaturalist/GBIF coverage) regardless of whether any real plant-pollinator
pair is densely co-located there. This step instead asks, for every possible
(plant species, pollinator species) combination, how much of each species'
range is shared with the other -- a measure of spatial co-occurrence at the
species-pair level.
"""

import pandas as pd
from pathlib import Path

BIN_SIZE = 0.5
DATA_DIR = Path("/scratch/ariana.l")
PLANT_PATH = DATA_DIR / "plant_flowering_events.parquet"
POLLINATOR_PATH = DATA_DIR / "pollinator_observations.csv"


def load_plants():
    plants = pd.read_parquet(PLANT_PATH, columns=["species", "lat", "lon"])
    print(f"Loading plant data from {PLANT_PATH} ...")
    print(f"  {len(plants):,} plant records, {plants['species'].nunique()} species")
    return plants


def load_pollinators_and_cache():
    """Loads the full 21.9M-row pollinator CSV (slow, ~30s-2min) and caches it
    to parquet so it never needs to be re-parsed from CSV again."""
    print(f"Loading pollinator data from {POLLINATOR_PATH} ...")
    pollinators = pd.read_csv(POLLINATOR_PATH, usecols=["pollinator_species", "lat", "lon"])
    print(
        f"  {len(pollinators):,} pollinator records, "
        f"{pollinators['pollinator_species'].nunique()} species"
    )

    pollinators.to_parquet(DATA_DIR / "cache_pollinators_raw.parquet")
    print("Cached.")
    return pollinators


if __name__ == "__main__":
    plants = load_plants()
    pollinators = load_pollinators_and_cache()
