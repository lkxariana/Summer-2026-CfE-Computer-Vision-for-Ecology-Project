"""
01_load_and_filter.py -- Step 8, Part 1

Load the top 100 Jaccard pairs (from Step 7, "Jaccard Range Overlap") and
the raw plant/pollinator observation data, then filter both down to only
the species actually involved in those 100 pairs. This is a large reduction
relative to Step 7's full sweep (50 plant species / 13,635 pollinator
species), since only 17 plant species and 63 pollinator species are needed.

Note on file choice: pollinator_observations.csv (used in Step 7) does not
contain a `doy` column, which is required here for weekly density histograms.
This script instead uses pollinator_observations_v2.csv, which does. The two
files have nearly identical row counts after filtering to the same species
(605,772 vs 604,940), so they are not meaningfully different datasets, but
they are not byte-identical either -- worth keeping in mind if comparing
Step 7 and Step 8 results directly.
"""

import pandas as pd
import numpy as np
from pathlib import Path

BIN_SIZE_TEMPORAL = 1.5   # spatial bin size for temporal/weekly density aggregation
N_WEEKS = 52
DATA_DIR = Path("/scratch/ariana.l")

PLANT_PATH = DATA_DIR / "plant_flowering_events.parquet"
JACCARD_PATH = DATA_DIR / "top100_jaccard_pairs.csv"
POLLINATOR_V2_PATH = DATA_DIR / "pollinator_observations_v2.csv"


def load_jaccard_pairs():
    jaccard_pairs = pd.read_csv(JACCARD_PATH)
    target_plants = set(jaccard_pairs["species"].unique())
    target_pollinators = set(jaccard_pairs["pollinator_species"].unique())
    print(f"Loaded {len(jaccard_pairs)} Jaccard pairs")
    print(f"Distinct plant species involved: {len(target_plants)}")
    print(f"Distinct pollinator species involved: {len(target_pollinators)}")
    return jaccard_pairs, target_plants, target_pollinators


def load_plants(target_plants):
    plants = pd.read_parquet(PLANT_PATH, columns=["species", "lat", "lon", "doy"])
    plants = plants[plants["species"].isin(target_plants)]
    print(f"Plant records (filtered): {len(plants):,}")
    return plants


def load_pollinators_v2(target_pollinators, chunksize=2_000_000):
    """Chunked filtered read of the v2 pollinator file (has doy), caches result."""
    cols = pd.read_csv(POLLINATOR_V2_PATH, nrows=0).columns.tolist()
    print(f"Columns in pollinator_observations_v2.csv: {cols}")

    chunks = []
    for chunk in pd.read_csv(
        POLLINATOR_V2_PATH,
        usecols=["pollinator_species", "lat", "lon", "doy"],
        chunksize=chunksize,
    ):
        filtered = chunk[chunk["pollinator_species"].isin(target_pollinators)]
        if len(filtered) > 0:
            chunks.append(filtered)

    pollinators = pd.concat(chunks, ignore_index=True)
    print(f"Pollinator records (filtered, with doy): {len(pollinators):,}")

    pollinators.to_parquet(DATA_DIR / "cache_step8_pollinators_filtered.parquet")
    print("Cached filtered pollinator data for Step 8.")
    return pollinators


if __name__ == "__main__":
    jaccard_pairs, target_plants, target_pollinators = load_jaccard_pairs()
    plants = load_plants(target_plants)
    pollinators = load_pollinators_v2(target_pollinators)
