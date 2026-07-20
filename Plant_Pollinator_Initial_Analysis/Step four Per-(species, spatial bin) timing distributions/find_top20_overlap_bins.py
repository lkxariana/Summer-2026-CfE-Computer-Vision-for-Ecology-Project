"""Identify the top 20 plant-pollinator overlap bins and reverse-geocode to US states.

For each spatial bin containing both plants and pollinators, compute a normalized
overlap score:

    score = (plant_count / max_plant_count) * (pollinator_count / max_pollinator_count)

Rank bins by score, take the top 20, and reverse-geocode each bin center to a
US state using geopy.Nominatim. Bins on coastal water are manually attributed
to the adjacent state.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from geopy.geocoders import Nominatim

PLANT_PATH = Path("/scratch/ariana.l/plant_flowering_events.parquet")
POLLINATOR_PATH = Path("/scratch/ariana.l/pollinator_observations.csv")
OUTPUT_PATH = Path("/scratch/ariana.l/top20_overlap_bins.csv")

BIN_SIZE = 0.5
TOP_N = 20

# Bins on coastal water that should be attributed to the adjacent state.
MANUAL_STATE_OVERRIDES = {
    19: "California",  # bin centered just west of San Francisco
}


def add_bins(df: pd.DataFrame, bin_size: float) -> pd.DataFrame:
    df = df.copy()
    df["lat_bin"] = (df["lat"] // bin_size) * bin_size
    df["lon_bin"] = (df["lon"] // bin_size) * bin_size
    return df


def bin_counts(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["lat_bin", "lon_bin"]).size().reset_index(name="count")


def compute_top_overlap(
    plant_counts: pd.DataFrame, pol_counts: pd.DataFrame
) -> pd.DataFrame:
    overlap = plant_counts.merge(
        pol_counts, on=["lat_bin", "lon_bin"], suffixes=("_plant", "_pol")
    )
    overlap["score"] = (
        overlap["count_plant"] / overlap["count_plant"].max()
    ) * (overlap["count_pol"] / overlap["count_pol"].max())
    return (
        overlap.sort_values("score", ascending=False)
        .head(TOP_N)
        .reset_index(drop=True)
    )


def reverse_geocode_states(top: pd.DataFrame) -> pd.DataFrame:
    geolocator = Nominatim(user_agent="bin_locator")
    rows = []
    for i, (_, row) in enumerate(top.iterrows(), 1):
        lon_min = row["lon_bin"]
        lon_max = row["lon_bin"] + BIN_SIZE
        lat_min = row["lat_bin"]
        lat_max = row["lat_bin"] + BIN_SIZE
        lat_center = lat_min + BIN_SIZE / 2
        lon_center = lon_min + BIN_SIZE / 2

        try:
            location = geolocator.reverse(
                f"{lat_center}, {lon_center}", language="en", timeout=10
            )
            if location and "address" in location.raw:
                state = location.raw["address"].get("state", "Ocean/Unknown")
            else:
                state = "Ocean/Unknown"
        except Exception:
            state = "Error"

        rows.append(
            {
                "rank": i,
                "lon_min": lon_min,
                "lon_max": lon_max,
                "lat_min": lat_min,
                "lat_max": lat_max,
                "state": state,
            }
        )
        time.sleep(1)  # Nominatim usage policy: ≤1 request/second

    df = pd.DataFrame(rows)
    for rank, state in MANUAL_STATE_OVERRIDES.items():
        df.loc[df["rank"] == rank, "state"] = state
    return df


def main() -> None:
    if not PLANT_PATH.exists():
        raise FileNotFoundError(f"Plant flowering events not found at {PLANT_PATH}")
    if not POLLINATOR_PATH.exists():
        raise FileNotFoundError(f"Pollinator observations not found at {POLLINATOR_PATH}")

    plants = add_bins(pd.read_parquet(PLANT_PATH), BIN_SIZE)
    pollinators = add_bins(pd.read_csv(POLLINATOR_PATH), BIN_SIZE)

    plant_counts = bin_counts(plants)
    pol_counts = bin_counts(pollinators)

    top = compute_top_overlap(plant_counts, pol_counts)
    print(f"Top {TOP_N} overlap bins by score:")
    print(top[["lat_bin", "lon_bin", "count_plant", "count_pol", "score"]])
    print()

    print("Reverse-geocoding bin centers to US states...")
    top_with_states = reverse_geocode_states(top)
    top_with_states.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {OUTPUT_PATH}")
    print(top_with_states)


if __name__ == "__main__":
    main()
