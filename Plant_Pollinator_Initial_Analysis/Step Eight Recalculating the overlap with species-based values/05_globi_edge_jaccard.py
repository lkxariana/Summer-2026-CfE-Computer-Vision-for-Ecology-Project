"""
globi_edge_jaccard.py -- Step 8, Part 5 (standalone sub-analysis)

Computes the Jaccard range-overlap score for the cleaned 39-edge GloBI
list (15 self-paired rows removed -- see Step 5 README correction), using
the same plant_bin_sets / pollinator_bin_sets dictionaries built in
Step 7 ("Jaccard Range Overlap").

This is independent of Step 7's top-100 search: it does not rank against
all possible species combinations, it only scores the 39 specific
documented edges and ranks them against each other.

Result: only 12 of 39 edges could be scored at all -- the remaining 27
involve a plant or pollinator species that did not survive Step 7's
MIN_TOTAL_OBS >= 10 filter and so has no entry in the bin-set dictionaries.
Of the 12 that resolved, scores ranged from 0.053 to 0.625, with a mean of
0.274 -- substantially lower than the top-100 searched pairs (which range
0.505-0.654 by construction). Only one edge, Larrea tridentata /
Asphondylia auripila, appears in both the top-100 list and this GloBI
ranking (rank 1 here, rank 2 in the top 100).
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path("/scratch/ariana.l")
EDGES_PATH = DATA_DIR / "CfE2026CVforEcology" / "rawpollinatordata" / "plant_pollinator_edges.csv"


def build_bin_sets_from_cache():
    """Rebuild plant_bin_sets / pollinator_bin_sets from Step 7's
    ("Jaccard Range Overlap") cached parquet files, avoiding a full re-scan
    of the raw observation CSVs."""
    plant_counts = pd.read_parquet(DATA_DIR / "cache_plant_counts.parquet")
    pollinator_counts = pd.read_parquet(DATA_DIR / "cache_pollinator_counts.parquet")

    plant_bin_sets = (
        plant_counts.groupby("species")
        .apply(lambda df: set(zip(df["lat_bin"], df["lon_bin"])), include_groups=False)
        .to_dict()
    )
    pollinator_bin_sets = (
        pollinator_counts.groupby("pollinator_species")
        .apply(lambda df: set(zip(df["lat_bin"], df["lon_bin"])), include_groups=False)
        .to_dict()
    )
    print(f"Plant species with bin sets: {len(plant_bin_sets)}")
    print(f"Pollinator species with bin sets: {len(pollinator_bin_sets)}")
    return plant_bin_sets, pollinator_bin_sets


def compute_globi_jaccard(plant_bin_sets, pollinator_bin_sets):
    edges_all = pd.read_csv(EDGES_PATH)
    edges_clean = edges_all[
        edges_all["plant_species"] != edges_all["pollinator_species"]
    ].reset_index(drop=True)
    print(
        f"Cleaned edge list: {len(edges_clean)} edges "
        f"(removed {len(edges_all) - len(edges_clean)} self-pairs)"
    )

    results = []
    missing = []

    for _, row in edges_clean.iterrows():
        p_species = row["plant_species"]
        q_species = row["pollinator_species"]

        p_bins = plant_bin_sets.get(p_species)
        q_bins = pollinator_bin_sets.get(q_species)

        if p_bins is None or q_bins is None:
            missing.append((p_species, q_species, p_bins is None, q_bins is None))
            continue

        intersection = p_bins & q_bins
        union = p_bins | q_bins
        jaccard = len(intersection) / len(union) if union else 0.0

        results.append({
            "plant_species": p_species,
            "pollinator_species": q_species,
            "jaccard": jaccard,
            "shared_bins": len(intersection),
        })

    globi_jaccard_df = (
        pd.DataFrame(results).sort_values("jaccard", ascending=False).reset_index(drop=True)
    )
    globi_jaccard_df.insert(0, "rank", globi_jaccard_df.index + 1)

    print(f"\nComputed Jaccard for {len(globi_jaccard_df)} / {len(edges_clean)} edges")
    if missing:
        print(
            f"{len(missing)} edges skipped (species not in plant_bin_sets/"
            f"pollinator_bin_sets -- filtered out in Step 7's MIN_TOTAL_OBS=10 step):"
        )
        for p, q, p_miss, q_miss in missing:
            reason = "plant missing" if p_miss else "pollinator missing"
            print(f"  {p} / {q} ({reason})")

    output_path = DATA_DIR / "globi39_jaccard_ranked.csv"
    globi_jaccard_df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")
    return globi_jaccard_df


if __name__ == "__main__":
    plant_bin_sets, pollinator_bin_sets = build_bin_sets_from_cache()
    compute_globi_jaccard(plant_bin_sets, pollinator_bin_sets)
