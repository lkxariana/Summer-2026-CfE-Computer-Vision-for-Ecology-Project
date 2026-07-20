"""
02_zero_fill_and_aggregate_1p5.py -- Stage 2, Step 1, Part 2

PPE's per-(species, 0.5-deg cell) flowering curves have inconsistent
week-of-year coverage (2-52 weeks present out of 52 possible). Naively
averaging across cells when aggregating to 1.5-deg bins breaks the
sum-to-1 normalization, because different weeks get averaged over
different denominators.

Per Dan (PPE's author): a missing week represents genuine species absence
at that location/time, not missing data. Missing weeks are therefore
zero-filled before aggregating, so every (species, cell) curve -- and every
aggregated (species, 1.5-deg bin) curve -- sums to 1.0 over the full year.
"""

import pandas as pd
import numpy as np
from pathlib import Path

PPE_DATA_DIR = Path("/scratch/ariana.l/ppe-outputs/data")
BIN_SIZE_TEMPORAL = 1.5
N_WEEKS = 52


def zero_fill_missing_weeks(flowering_all):
    """Reindex every (species, cell) to the full 0..51 week range via an
    explicit merge (NOT groupby().apply(), which can silently mangle column
    names on reset_index() across pandas versions). Missing weeks -> 0.0."""
    species_cells = flowering_all[["species", "centroid_lat", "centroid_lon"]].drop_duplicates()
    weeks_df = pd.DataFrame({"week": range(N_WEEKS)})
    species_cells["_key"] = 1
    weeks_df["_key"] = 1
    full_grid = species_cells.merge(weeks_df, on="_key").drop(columns="_key")

    filled = full_grid.merge(
        flowering_all[["species", "centroid_lat", "centroid_lon", "week", "inat_norm"]],
        on=["species", "centroid_lat", "centroid_lon", "week"],
        how="left",
    )
    filled["inat_norm"] = filled["inat_norm"].fillna(0.0)

    check = filled.groupby(["species", "centroid_lat", "centroid_lon"])["inat_norm"].sum()
    print("Per-(species, cell) curve sum after zero-fill (should be ~1.0):")
    print(check.describe())

    return filled


def aggregate_to_1p5(filled, bin_size=BIN_SIZE_TEMPORAL):
    """Average zero-filled inat_norm across all 0.5-deg cells inside each
    1.5-deg bin, per species per week."""
    filled = filled.copy()
    filled["lat_bin_1p5"] = np.floor(filled["centroid_lat"] / bin_size) * bin_size
    filled["lon_bin_1p5"] = np.floor(filled["centroid_lon"] / bin_size) * bin_size

    ppe_1p5 = (
        filled.groupby(["species", "lat_bin_1p5", "lon_bin_1p5", "week"])["inat_norm"]
        .mean()
        .reset_index()
    )

    check_1p5 = ppe_1p5.groupby(["species", "lat_bin_1p5", "lon_bin_1p5"])["inat_norm"].sum()
    print("\nPer-(species, 1.5-deg bin) curve sum after aggregation (should be ~1.0):")
    print(check_1p5.describe())

    return ppe_1p5


def aggregate_coverage_to_1p5(coverage, bin_size=BIN_SIZE_TEMPORAL):
    """Aggregate coverage.csv's n_weeks_present (cell-level) up to 1.5-deg
    bins, by averaging across the contributing 0.5-deg cells. Used downstream
    as the per-bin weight for the PPE-based weighted mean overlap."""
    coverage = coverage.copy()
    coverage["lat_bin_1p5"] = np.floor(coverage["centroid_lat"] / bin_size) * bin_size
    coverage["lon_bin_1p5"] = np.floor(coverage["centroid_lon"] / bin_size) * bin_size

    coverage_1p5 = (
        coverage.groupby(["lat_bin_1p5", "lon_bin_1p5"])["n_weeks_present"]
        .mean()
        .reset_index()
        .rename(columns={"n_weeks_present": "weight_n_weeks_present"})
    )
    return coverage_1p5


if __name__ == "__main__":
    from importlib import import_module
    mod = import_module("01_load_and_validate_ppe_coverage")
    flowering_all, coverage = mod.load_ppe_data()

    filled = zero_fill_missing_weeks(flowering_all)
    ppe_1p5 = aggregate_to_1p5(filled)
    coverage_1p5 = aggregate_coverage_to_1p5(coverage)

    ppe_1p5.to_parquet(PPE_DATA_DIR / "ppe_1p5_aggregated.parquet", index=False)
    coverage_1p5.to_csv(PPE_DATA_DIR / "coverage_1p5_aggregated.csv", index=False)
    print(f"\nSaved ppe_1p5_aggregated.parquet ({len(ppe_1p5):,} rows)")
    print(f"Saved coverage_1p5_aggregated.csv ({len(coverage_1p5):,} rows)")
