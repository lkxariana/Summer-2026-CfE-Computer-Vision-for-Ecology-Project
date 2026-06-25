"""
03_compute_ppe_overlap.py -- Stage 2, Step 1, Part 3

Compute the temporal overlap coefficient for Step 7's top 100 pairs using
PPE-predicted plant curves (from 02_zero_fill_and_aggregate_1p5.py) on the
plant side, and Step 8's original observation-based logic, unchanged, on
the pollinator side. Mirrors Step 8's 02/03/04 scripts structurally so the
two summaries are directly comparable.
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("/scratch/ariana.l/Plant Pollinator Initial Analysis")
PPE_DATA_DIR = Path("/scratch/ariana.l/ppe-outputs/data")
OUTPUTS_DIR = Path("/scratch/ariana.l/Part2_PhenoField_Predictions/step1_ppe_overlap_outputs")

BIN_SIZE_TEMPORAL = 1.5
N_WEEKS = 52
MIN_OBS = 5


def add_bins(df, bin_size=BIN_SIZE_TEMPORAL):
    """Identical to Step 8's add_bins() -- pollinator side is unchanged."""
    df = df.copy()
    df["lat_bin"] = np.floor(df["lat"] / bin_size) * bin_size
    df["lon_bin"] = np.floor(df["lon"] / bin_size) * bin_size
    return df


def weekly_density(doy_values, n_weeks=N_WEEKS):
    """Identical to Step 8's weekly_density() -- pollinator side is unchanged."""
    weeks = np.clip((doy_values - 1) // 7, 0, n_weeks - 1).astype(int)
    counts = np.bincount(weeks, minlength=n_weeks)
    total = counts.sum()
    return counts / total if total > 0 else counts


def build_plant_density_ppe(ppe_1p5):
    ppe_1p5_sorted = ppe_1p5.sort_values(["species", "lat_bin_1p5", "lon_bin_1p5", "week"])
    plant_density_ppe = {
        key: grp["inat_norm"].values
        for key, grp in ppe_1p5_sorted.groupby(["species", "lat_bin_1p5", "lon_bin_1p5"])
    }
    print(f"PPE plant density entries: {len(plant_density_ppe):,}")
    return plant_density_ppe


def build_pollinator_density(top100, pollinator_obs, min_obs=MIN_OBS):
    """Identical filtering logic to Step 8's 02/03 scripts: bin at 1.5 deg,
    drop (species, bin) cells below MIN_OBS, build weekly_density() per cell."""
    target_pollinators = set(top100["pollinator_species"].unique())
    pollinators_f = pollinator_obs[pollinator_obs["pollinator_species"].isin(target_pollinators)].copy()
    pollinators_f = add_bins(pollinators_f, bin_size=BIN_SIZE_TEMPORAL)

    pollinator_bin_counts = (
        pollinators_f.groupby(["pollinator_species", "lat_bin", "lon_bin"]).size().reset_index(name="n_obs")
    )
    keep_set = set(zip(
        pollinator_bin_counts[pollinator_bin_counts["n_obs"] >= min_obs]["pollinator_species"],
        pollinator_bin_counts[pollinator_bin_counts["n_obs"] >= min_obs]["lat_bin"],
        pollinator_bin_counts[pollinator_bin_counts["n_obs"] >= min_obs]["lon_bin"],
    ))
    pollinators_f["_key"] = list(zip(
        pollinators_f["pollinator_species"], pollinators_f["lat_bin"], pollinators_f["lon_bin"]
    ))
    pollinators_f = pollinators_f[pollinators_f["_key"].isin(keep_set)]

    pollinator_density = {
        key: weekly_density(group["doy"].values)
        for key, group in pollinators_f.groupby(["pollinator_species", "lat_bin", "lon_bin"])
    }
    print(f"Pollinator density entries: {len(pollinator_density):,}")
    return pollinator_density, pollinator_bin_counts


def compute_overlap(top100, plant_density_ppe, pollinator_density, pollinator_bin_counts, coverage_1p5):
    results = []
    for _, row in top100.iterrows():
        p_species, q_species = row["species"], row["pollinator_species"]

        p_bins = {k[1:] for k in plant_density_ppe if k[0] == p_species}
        q_bins = {k[1:] for k in pollinator_density if k[0] == q_species}
        shared = p_bins & q_bins

        for bin_key in shared:
            p_dens = plant_density_ppe[(p_species, *bin_key)]
            q_dens = pollinator_density[(q_species, *bin_key)]
            overlap = np.sum(np.minimum(p_dens, q_dens))

            n_obs_q = pollinator_bin_counts[
                (pollinator_bin_counts["pollinator_species"] == q_species)
                & (pollinator_bin_counts["lat_bin"] == bin_key[0])
                & (pollinator_bin_counts["lon_bin"] == bin_key[1])
            ]["n_obs"].values[0]

            w_row = coverage_1p5[
                (coverage_1p5["lat_bin_1p5"] == bin_key[0]) & (coverage_1p5["lon_bin_1p5"] == bin_key[1])
            ]
            weight_ppe = w_row["weight_n_weeks_present"].values[0] if len(w_row) > 0 else np.nan

            results.append({
                "jaccard_rank": row["rank"], "species": p_species, "pollinator_species": q_species,
                "lat_bin": bin_key[0], "lon_bin": bin_key[1],
                "overlap_ppe": overlap, "n_obs_pollinator": n_obs_q,
                "weight_n_weeks_present": weight_ppe,
            })

    overlap_df_ppe = pd.DataFrame(results)
    print(f"Total (edge, bin) overlap rows: {len(overlap_df_ppe):,}")
    print(f"Distinct edges with >=1 valid bin: "
          f"{overlap_df_ppe[['species', 'pollinator_species']].drop_duplicates().shape[0]} / 100")
    return overlap_df_ppe


def weighted_mean_ppe(g):
    w = g["weight_n_weeks_present"].fillna(0)
    return np.average(g["overlap_ppe"], weights=w) if w.sum() > 0 else np.nan


def summarize_edges(overlap_df_ppe):
    edge_summary_ppe = (
        overlap_df_ppe.groupby(["jaccard_rank", "species", "pollinator_species"])
        .apply(lambda g: pd.Series({
            "n_bins": len(g),
            "overlap_mean": g["overlap_ppe"].mean(),
            "overlap_weighted_mean": weighted_mean_ppe(g),
            "overlap_std": g["overlap_ppe"].std(),
            "overlap_min": g["overlap_ppe"].min(),
            "overlap_max": g["overlap_ppe"].max(),
        }), include_groups=False)
        .reset_index()
        .sort_values("jaccard_rank")
    )

    output_path = OUTPUTS_DIR / "edge_summary_ppe.csv"
    edge_summary_ppe.to_csv(output_path, index=False)
    print(f"Saved {len(edge_summary_ppe)} edge summaries to {output_path}")
    return edge_summary_ppe


if __name__ == "__main__":
    top100 = pd.read_csv(DATA_DIR / "top100_jaccard_pairs.csv")
    pollinator_obs = pd.read_csv(DATA_DIR / "pollinator_observations_v2.csv")
    ppe_1p5 = pd.read_parquet(PPE_DATA_DIR / "ppe_1p5_aggregated.parquet")
    coverage_1p5 = pd.read_csv(PPE_DATA_DIR / "coverage_1p5_aggregated.csv")

    plant_density_ppe = build_plant_density_ppe(ppe_1p5)
    pollinator_density, pollinator_bin_counts = build_pollinator_density(top100, pollinator_obs)
    overlap_df_ppe = compute_overlap(top100, plant_density_ppe, pollinator_density,
                                      pollinator_bin_counts, coverage_1p5)
    edge_summary_ppe = summarize_edges(overlap_df_ppe)
