"""
01_load_and_validate_ppe_coverage.py -- Stage 2, Step 1, Part 1

Load Dan's PPE flowering-curve output and Step 7's top-100 Jaccard pairs,
reconstruct each pair's full shared-bin set (only the bin *count* was saved
originally), resolve the corner-vs-center grid offset between Step 7's
hotspot coordinates and PPE's grid centroids, and validate that PPE
coverage actually reaches every pair's shared bins at the species level
(not just the cell level).
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("/scratch/ariana.l/Plant Pollinator Initial Analysis")
PPE_DATA_DIR = Path("/scratch/ariana.l/ppe-outputs/data")

CORNER_TO_CENTER_OFFSET = 0.25  # Step 7 hotspot coords (corner) -> PPE centroid coords (center)


def load_ppe_data():
    flowering_all = pd.read_parquet(PPE_DATA_DIR / "flowering_curves_all.parquet")
    coverage = pd.read_csv(PPE_DATA_DIR / "coverage.csv")
    print(f"flowering_curves_all.parquet: {flowering_all.shape}, "
          f"{flowering_all['cell_idx'].nunique()} cells, "
          f"{flowering_all['species'].nunique()} species")
    print(f"coverage.csv: {coverage.shape[0]} cells total, "
          f"{(coverage['n_weeks_present'] > 0).sum()} with >=1 week present")
    return flowering_all, coverage


def load_jaccard_pairs_and_caches():
    top100 = pd.read_csv(DATA_DIR / "top100_jaccard_pairs.csv")
    cache_plant = pd.read_parquet(DATA_DIR / "cache_plant_counts.parquet")
    cache_pollinator = pd.read_parquet(DATA_DIR / "cache_pollinator_counts.parquet")
    return top100, cache_plant, cache_pollinator


def reconstruct_shared_bins(top100, cache_plant, cache_pollinator):
    """Rebuild each pair's full shared-bin set from Step 7's cached per-(species,
    bin) counts. top100_jaccard_pairs.csv only stores the bin *count*, not the
    bin list itself -- this reconstructs the list and verifies the count matches."""
    plant_bins = cache_plant.groupby("species").apply(
        lambda df: set(zip(df["lat_bin"], df["lon_bin"])), include_groups=False
    ).to_dict()
    pollinator_bins = cache_pollinator.groupby("pollinator_species").apply(
        lambda df: set(zip(df["lat_bin"], df["lon_bin"])), include_groups=False
    ).to_dict()

    shared_bin_sets = {}
    mismatches = []
    for _, row in top100.iterrows():
        p_bins = plant_bins.get(row["species"], set())
        q_bins = pollinator_bins.get(row["pollinator_species"], set())
        shared = p_bins & q_bins
        shared_bin_sets[(row["species"], row["pollinator_species"])] = shared
        if len(shared) != row["shared_bins"]:
            mismatches.append((row["species"], row["pollinator_species"],
                                len(shared), row["shared_bins"]))

    if mismatches:
        print(f"WARNING: {len(mismatches)} pairs have a reconstructed shared-bin "
              f"count that does not match the original 'shared_bins' column:")
        for p, q, recon, orig in mismatches:
            print(f"  {p} / {q}: reconstructed={recon}, original={orig}")
    else:
        print("Reconstructed shared-bin counts match the original 'shared_bins' "
              "column for all 100 pairs.")

    return shared_bin_sets


def validate_ppe_coverage(top100, shared_bin_sets, flowering_all):
    """Check, per pair, what fraction of its shared bins have a PPE prediction
    for the correct (cell, species) -- not just (cell) alone."""
    ppe_species_cells = set(zip(
        flowering_all["centroid_lat"], flowering_all["centroid_lon"], flowering_all["species"]
    ))

    results = []
    for _, row in top100.iterrows():
        shared = shared_bin_sets[(row["species"], row["pollinator_species"])]
        covered = [
            (lat + CORNER_TO_CENTER_OFFSET, lon + CORNER_TO_CENTER_OFFSET)
            for (lat, lon) in shared
            if (lat + CORNER_TO_CENTER_OFFSET, lon + CORNER_TO_CENTER_OFFSET, row["species"]) in ppe_species_cells
        ]
        results.append({
            "rank": row["rank"],
            "species": row["species"],
            "pollinator_species": row["pollinator_species"],
            "shared_bins_original": row["shared_bins"],
            "ppe_covered_bins": len(covered),
            "ppe_coverage_frac": len(covered) / len(shared) if shared else 0.0,
        })

    coverage_df = pd.DataFrame(results)
    print(f"\nPairs with >0% PPE coverage: {(coverage_df['ppe_coverage_frac'] > 0).sum()} / 100")
    print(f"Pairs with 100% PPE coverage: {(coverage_df['ppe_coverage_frac'] == 1.0).sum()} / 100")
    print(f"Mean coverage fraction: {coverage_df['ppe_coverage_frac'].mean():.4f}")
    return coverage_df


if __name__ == "__main__":
    flowering_all, coverage = load_ppe_data()
    top100, cache_plant, cache_pollinator = load_jaccard_pairs_and_caches()
    shared_bin_sets = reconstruct_shared_bins(top100, cache_plant, cache_pollinator)
    coverage_df = validate_ppe_coverage(top100, shared_bin_sets, flowering_all)
