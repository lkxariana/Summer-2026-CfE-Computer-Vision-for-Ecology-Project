"""
04_jaccard_overlap.py -- Step 7, Part 4 (core method)

Computes a Jaccard range-overlap score for every (plant species, pollinator
species) combination that shares at least one 0.5deg bin:

    jaccard = |bins(plant) ∩ bins(pollinator)| / |bins(plant) ∪ bins(pollinator)|

This was the SECOND approach tried in this step, after an initial min-count
approach (score = min(plant_count_in_bin, pollinator_count_in_bin),
streamed via a bounded min-heap to avoid materializing the ~17.4M-row full
cross join) produced misleading results: a single locally-abundant plant
species (Bellis perennis) flooded the top of the ranking by acting as a
"ceiling" against every locally-abundant pollinator in one Bay Area bin,
regardless of any real species-pair-specific relationship. That min-count
approach is NOT included in this export, since it was abandoned.

Jaccard fixes this because a pair only scores high if their geographic
RANGES are tightly coincident -- a cosmopolitan species occupying hundreds
of bins is penalized by the union term in the denominator, even if it is
locally abundant in any one of them.

Performance note: an initial version of the per-pair hotspot-bin lookup
used pandas .loc filtering inside the main loop, which was too slow
(estimated 20-40+ minutes for 50 x 13,635 candidate pairs). The version
below precomputes O(1) dictionary lookups for (species, bin) -> count
before the main loop, which brought runtime down to roughly 2-5 minutes.
"""

import heapq
import pandas as pd
from pathlib import Path

TOP_N = 100
DATA_DIR = Path("/scratch/ariana.l")


def build_bin_sets(plant_counts, pollinator_counts):
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


def build_count_dicts(plant_counts, pollinator_counts):
    """O(1) lookup dicts for (species, lat_bin, lon_bin) -> count, used for the
    hotspot-bin selection inside the main loop. Replaces a much slower
    pandas .loc filter that made the naive version of this loop impractically
    slow."""
    plant_count_dict = {
        (row.species, row.lat_bin, row.lon_bin): row.plant_count
        for row in plant_counts.itertuples()
    }
    pollinator_count_dict = {
        (row.pollinator_species, row.lat_bin, row.lon_bin): row.pollinator_count
        for row in pollinator_counts.itertuples()
    }
    print(
        f"Lookup dicts built: {len(plant_count_dict):,} plant entries, "
        f"{len(pollinator_count_dict):,} pollinator entries"
    )
    return plant_count_dict, pollinator_count_dict


def compute_top_jaccard_pairs(plant_bin_sets, pollinator_bin_sets,
                                plant_count_dict, pollinator_count_dict, top_n=TOP_N):
    plant_species_list = list(plant_bin_sets.keys())
    jaccard_heap = []  # min-heap of (jaccard, shared_bins, plant, pollinator, hotspot_lat, hotspot_lon)

    for i, p_species in enumerate(plant_species_list):
        p_bins = plant_bin_sets[p_species]
        for q_species, q_bins in pollinator_bin_sets.items():
            intersection = p_bins & q_bins
            if not intersection:
                continue

            jaccard = len(intersection) / len(p_bins | q_bins)

            # Hotspot bin: the shared bin with the highest min(plant_count, pollinator_count)
            hotspot = max(
                intersection,
                key=lambda b: min(
                    plant_count_dict.get((p_species, b[0], b[1]), 0),
                    pollinator_count_dict.get((q_species, b[0], b[1]), 0),
                ),
            )

            entry = (jaccard, len(intersection), p_species, q_species, hotspot[0], hotspot[1])
            if len(jaccard_heap) < top_n:
                heapq.heappush(jaccard_heap, entry)
            elif jaccard > jaccard_heap[0][0]:
                heapq.heapreplace(jaccard_heap, entry)

        if (i + 1) % 5 == 0:
            print(f"  plant species {i+1} / {len(plant_species_list)} done")

    print(f"Done. Heap has {len(jaccard_heap)} entries.")
    return jaccard_heap


def export_results(jaccard_heap):
    jaccard_results = pd.DataFrame(
        jaccard_heap,
        columns=["jaccard", "shared_bins", "species", "pollinator_species", "hotspot_lat", "hotspot_lon"],
    ).sort_values("jaccard", ascending=False).reset_index(drop=True)

    jaccard_results.insert(0, "rank", jaccard_results.index + 1)
    jaccard_results["jaccard"] = jaccard_results["jaccard"].round(4)

    output_path = DATA_DIR / "top100_jaccard_pairs.csv"
    jaccard_results.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    return jaccard_results


if __name__ == "__main__":
    plant_counts = pd.read_parquet(DATA_DIR / "cache_plant_counts.parquet")
    pollinator_counts = pd.read_parquet(DATA_DIR / "cache_pollinator_counts.parquet")

    plant_bin_sets, pollinator_bin_sets = build_bin_sets(plant_counts, pollinator_counts)
    plant_count_dict, pollinator_count_dict = build_count_dicts(plant_counts, pollinator_counts)

    jaccard_heap = compute_top_jaccard_pairs(
        plant_bin_sets, pollinator_bin_sets, plant_count_dict, pollinator_count_dict
    )
    jaccard_results = export_results(jaccard_heap)
