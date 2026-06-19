"""
visualize_phenology.py -- Step 8, Part 6

KDE-smoothed phenology grid plots, in the same visual style as Step 5
(bw_method=0.20), but built from Step 8's edge x bin overlap data: curves
are averaged across ALL shared bins per edge (not Step 5's top-20-density
restriction), and panels are ordered west-to-east by each edge's mean
longitude across its shared bins.

Three grids were produced from this function in the actual analysis:
  1. Top 12 by overlap_weighted_mean (most temporally synced pairs)
  2. Top 10 by Jaccard rank (most spatially tight pairs)
  3. Top 10 by overlap_weighted_mean (duplicate of #1 at n=10, used for direct
     side-by-side comparison against #2)

Comparing grids #2 and #3 is the main visual finding of Step 8: spatial
range overlap (Jaccard) and temporal phenological overlap are largely
independent signals. Several top-Jaccard pairs show plant and pollinator
activity curves that barely intersect at all.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from pathlib import Path

DATA_DIR = Path("/scratch/ariana.l")
N_WEEKS = 52

PLANT_COLOR = "#3b6fa0"
POLLINATOR_COLOR = "#d9842c"
OVERLAP_COLOR = "#7a5c8e"


def _mean_longitude(edges_df, overlap_df, species_col, pollinator_col):
    mean_lons = []
    for _, row in edges_df.iterrows():
        edge_bins = overlap_df[
            (overlap_df["species"] == row[species_col])
            & (overlap_df["pollinator_species"] == row[pollinator_col])
        ]["lon_bin"]
        mean_lons.append(edge_bins.mean() if len(edge_bins) > 0 else np.nan)
    return mean_lons


def plot_phenology_grid(edges_df, overlap_df, plant_density, pollinator_density,
                          title_fn, output_filename, species_col="species",
                          pollinator_col="pollinator_species", n_panels=10,
                          grid_shape=(4, 3)):
    """
    Generic KDE phenology grid plotter.

    edges_df: dataframe with at least [species_col, pollinator_col] and
        whatever columns title_fn needs.
    title_fn: function(row) -> str, builds the per-panel title from a row of
        edges_df (already merged with any needed summary stats beforehand).
    """
    edges_df = edges_df.head(n_panels).copy()
    edges_df["mean_lon"] = _mean_longitude(edges_df, overlap_df, species_col, pollinator_col)
    edges_df = edges_df.sort_values("mean_lon").reset_index(drop=True)

    weeks = np.arange(N_WEEKS)
    n_rows, n_cols = grid_shape

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 14))
    fig.patch.set_facecolor("white")
    axes = axes.flatten()
    for ax in axes:
        ax.set_facecolor("white")
    for ax in axes[n_panels:]:
        ax.axis("off")

    for i, row in edges_df.iterrows():
        p_species = row[species_col]
        q_species = row[pollinator_col]

        edge_bins = overlap_df[
            (overlap_df["species"] == p_species) & (overlap_df["pollinator_species"] == q_species)
        ][["lat_bin", "lon_bin"]].values

        ax = axes[i]
        if len(edge_bins) == 0:
            ax.set_title(f"{p_species}\n{q_species}\n(no valid overlap bin)", fontsize=8.5)
            ax.axis("off")
            continue

        p_avg = np.mean([plant_density[(p_species, lb, lo)] for lb, lo in edge_bins], axis=0)
        q_avg = np.mean([pollinator_density[(q_species, lb, lo)] for lb, lo in edge_bins], axis=0)

        p_kde = gaussian_kde(weeks, weights=p_avg, bw_method=0.20)
        q_kde = gaussian_kde(weeks, weights=q_avg, bw_method=0.20)
        x_smooth = np.linspace(0, N_WEEKS - 1, 300)
        p_smooth = p_kde(x_smooth)
        q_smooth = q_kde(x_smooth)

        ax.plot(x_smooth, p_smooth, color=PLANT_COLOR, linewidth=1.8, label="Plant")
        ax.plot(x_smooth, q_smooth, color=POLLINATOR_COLOR, linewidth=1.8, label="Pollinator")
        ax.fill_between(x_smooth, np.minimum(p_smooth, q_smooth), color=OVERLAP_COLOR, alpha=0.45)

        ax.set_title(title_fn(row), fontsize=8)
        ax.set_xlabel("Week of year", fontsize=8)
        ax.set_xticks([0, 13, 26, 39, 51])
        ax.tick_params(labelsize=7.5)
        if i == 0:
            ax.legend(fontsize=7)

    plt.tight_layout()
    output_path = DATA_DIR / output_filename
    plt.savefig(output_path, dpi=150, facecolor="white")
    plt.show()
    print(f"Saved to {output_path}")
    return fig


def build_top10_jaccard_grid(jaccard_pairs, edge_summary, overlap_df):
    top10 = jaccard_pairs.head(10).merge(
        edge_summary[["species", "pollinator_species", "overlap_weighted_mean", "n_bins"]],
        on=["species", "pollinator_species"], how="left",
    )

    def title_fn(row):
        wm = row["overlap_weighted_mean"]
        wm_str = f"{wm:.3f}" if pd.notna(wm) else "N/A"
        return f"{row['species']} / {row['pollinator_species']}\njaccard={row['jaccard']:.3f}, temporal={wm_str}"

    return plot_phenology_grid(
        top10, overlap_df, plant_density, pollinator_density,
        title_fn, "step8_phenology_grid_top10_jaccard.png", n_panels=10,
    )


def build_top10_temporal_grid(edge_summary, overlap_df):
    top10 = edge_summary.sort_values("overlap_weighted_mean", ascending=False).head(10)

    def title_fn(row):
        return (
            f"{row['species']} / {row['pollinator_species']}\n"
            f"temporal={row['overlap_weighted_mean']:.3f}, n={int(row['n_bins'])} bins"
        )

    return plot_phenology_grid(
        top10, overlap_df, plant_density, pollinator_density,
        title_fn, "step8_phenology_grid_top10_temporal.png", n_panels=10,
    )


def build_top12_temporal_grid(edge_summary, overlap_df):
    """The original n=12 weighted-overlap grid; kept distinct from the n=10
    version above since it was produced first and is a slightly different cut."""
    top12 = edge_summary.sort_values("overlap_weighted_mean", ascending=False).head(12)

    def title_fn(row):
        return (
            f"{row['species']} / {row['pollinator_species']}\n"
            f"weighted overlap={row['overlap_weighted_mean']:.3f}, "
            f"mean lon={row['mean_lon']:.1f}"
        )

    return plot_phenology_grid(
        top12, overlap_df, plant_density, pollinator_density,
        title_fn, "step8_phenology_grid_top12_temporal.png",
        n_panels=12, grid_shape=(4, 3),
    )
