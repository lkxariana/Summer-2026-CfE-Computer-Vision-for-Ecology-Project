"""
compute_overlap.py -- Step 8, Part 4

For each of the top 100 Jaccard pairs, compute the temporal overlap
coefficient (overlap = sum(min(p_density, q_density)), same formula as
Step 5) at every shared 1.5-degree bin -- not just a single hotspot bin or
a top-20-density restriction, since the Jaccard-derived pairs already
guarantee tight spatial overlap.

Also computes a per-edge summary, including an observation-weighted mean
that weights each bin's overlap by min(n_obs_plant, n_obs_pollinator), so
bins backed by more data on their weaker side count more -- consistent with
the min-count logic used for the Jaccard score itself in Step 7.
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path("/scratch/ariana.l")


def compute_edge_bin_overlap(jaccard_pairs, plant_density, pollinator_density,
                               plant_bin_counts_filtered, pollinator_bin_counts_filtered):
    results = []

    for _, row in jaccard_pairs.iterrows():
        p_species = row["species"]
        q_species = row["pollinator_species"]

        p_bins = {key[1:] for key in plant_density if key[0] == p_species}
        q_bins = {key[1:] for key in pollinator_density if key[0] == q_species}
        shared = p_bins & q_bins

        for bin_key in shared:
            p_dens = plant_density[(p_species, *bin_key)]
            q_dens = pollinator_density[(q_species, *bin_key)]
            overlap = np.sum(np.minimum(p_dens, q_dens))

            n_obs_p = plant_bin_counts_filtered[
                (plant_bin_counts_filtered["species"] == p_species)
                & (plant_bin_counts_filtered["lat_bin"] == bin_key[0])
                & (plant_bin_counts_filtered["lon_bin"] == bin_key[1])
            ]["n_obs"].values[0]
            n_obs_q = pollinator_bin_counts_filtered[
                (pollinator_bin_counts_filtered["pollinator_species"] == q_species)
                & (pollinator_bin_counts_filtered["lat_bin"] == bin_key[0])
                & (pollinator_bin_counts_filtered["lon_bin"] == bin_key[1])
            ]["n_obs"].values[0]

            results.append({
                "jaccard_rank": row["rank"],
                "species": p_species,
                "pollinator_species": q_species,
                "lat_bin": bin_key[0],
                "lon_bin": bin_key[1],
                "overlap": overlap,
                "n_obs_plant": n_obs_p,
                "n_obs_pollinator": n_obs_q,
            })

    overlap_df = pd.DataFrame(results)
    print(f"Total (edge, bin) overlap rows: {len(overlap_df):,}")
    print(
        f"Distinct edges with at least one valid bin: "
        f"{overlap_df[['species', 'pollinator_species']].drop_duplicates().shape[0]} / 100"
    )
    return overlap_df


def weighted_mean(group):
    weights = np.minimum(group["n_obs_plant"], group["n_obs_pollinator"])
    return np.average(group["overlap"], weights=weights)


def summarize_edges(overlap_df):
    edge_summary = (
        overlap_df.groupby(["jaccard_rank", "species", "pollinator_species"])
        .apply(
            lambda g: pd.Series({
                "n_bins": len(g),
                "overlap_mean": g["overlap"].mean(),
                "overlap_weighted_mean": weighted_mean(g),
                "overlap_std": g["overlap"].std(),
                "overlap_min": g["overlap"].min(),
                "overlap_max": g["overlap"].max(),
                "total_obs_plant": g["n_obs_plant"].sum(),
                "total_obs_pollinator": g["n_obs_pollinator"].sum(),
            }),
            include_groups=False,
        )
        .reset_index()
        .sort_values("jaccard_rank")
    )

    output_path = DATA_DIR / "step8_edge_overlap_summary.csv"
    edge_summary.to_csv(output_path, index=False)
    print(f"Saved {len(edge_summary)} edge summaries to {output_path}")
    return edge_summary
