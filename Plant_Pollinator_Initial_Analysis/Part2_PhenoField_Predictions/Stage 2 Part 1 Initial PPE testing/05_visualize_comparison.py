"""
05_visualize_comparison.py -- Stage 2, Step 1, Part 5

Three figures comparing PPE-based vs. original (observation-based) temporal
overlap: a scatter plot against the y=x line, a histogram/KDE comparing the
two distributions' shapes, and a 9-panel KDE grid (in the same visual style
as Step 8's 06_visualize_phenology.py) showing the plant/pollinator curves
for the pairs with the largest disagreement.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from pathlib import Path

DATA_DIR = Path("/scratch/ariana.l/Plant Pollinator Initial Analysis")
PPE_DATA_DIR = Path("/scratch/ariana.l/ppe-outputs/data")
OUTPUTS_DIR = Path("/scratch/ariana.l/Part2_PhenoField_Predictions/step1_ppe_overlap_outputs")

BIN_SIZE_TEMPORAL = 1.5
N_WEEKS = 52
MIN_OBS = 5

ORIG_COLOR = "#3b6fa0"
PPE_COLOR = "red"
POLLINATOR_COLOR = "#d9842c"
OVERLAP_COLOR = "#7a5c8e"
DIAG_COLOR = "#888888"


def plot_scatter(comparison, output_path):
    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    sc = ax.scatter(
        comparison["overlap_weighted_mean_orig"],
        comparison["overlap_weighted_mean_ppe"],
        c=comparison["n_bins_ppe"],
        cmap="viridis",
        s=40,
        alpha=0.8,
        edgecolors="white",
        linewidths=0.4,
    )
    lims = [0, max(comparison["overlap_weighted_mean_orig"].max(),
                   comparison["overlap_weighted_mean_ppe"].max()) * 1.05]
    ax.plot(lims, lims, color=DIAG_COLOR, linestyle="--", linewidth=1.2, label="y = x (perfect agreement)")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Original (observation-based) weighted mean overlap")
    ax.set_ylabel("PPE (climatology-predicted) weighted mean overlap")
    r = comparison[["overlap_weighted_mean_ppe", "overlap_weighted_mean_orig"]].corr().iloc[0, 1]
    ax.set_title(f"PPE vs. original temporal overlap, all 100 pairs\n(r = {r:.3f})")
    ax.legend(fontsize=9, loc="upper left")
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("n_bins (PPE)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Saved {output_path.name}")
    return lims


def plot_distribution(comparison, lims, output_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    x_smooth = np.linspace(0, lims[1], 300)
    for col, color, label in [
        ("overlap_weighted_mean_orig", POLLINATOR_COLOR, "Original (observation-based)"),
        ("overlap_weighted_mean_ppe", ORIG_COLOR, "PPE (climatology-predicted)"),
    ]:
        vals = comparison[col].dropna().values
        ax.hist(vals, bins=20, range=lims, density=True, alpha=0.25, color=color)
        kde = gaussian_kde(vals, bw_method=0.3)
        ax.plot(x_smooth, kde(x_smooth), color=color, linewidth=2.2, label=label)
        ax.axvline(vals.mean(), color=color, linestyle=":", linewidth=1.5)

    ax.set_xlabel("Weighted mean overlap coefficient")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of weighted mean overlap across 100 pairs:\nOriginal vs. PPE-based")
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Saved {output_path.name}")


def plot_disagreement_grid(comparison, plant_density_orig, plant_density_ppe,
                             pollinator_density, output_path, n_panels=9):
    weeks = np.arange(N_WEEKS)
    disagreement_pairs = comparison.reindex(
        comparison["weighted_mean_diff"].abs().sort_values(ascending=False).index
    ).head(n_panels)

    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    fig.patch.set_facecolor("white")
    axes = axes.flatten()

    for i, (_, row) in enumerate(disagreement_pairs.iterrows()):
        p_species, q_species = row["species"], row["pollinator_species"]
        ax = axes[i]
        ax.set_facecolor("white")

        p_bins_orig = {k[1:] for k in plant_density_orig if k[0] == p_species}
        p_bins_ppe = {k[1:] for k in plant_density_ppe if k[0] == p_species}
        q_bins = {k[1:] for k in pollinator_density if k[0] == q_species}

        shared_orig = p_bins_orig & q_bins
        shared_ppe = p_bins_ppe & q_bins

        if not shared_orig or not shared_ppe:
            ax.set_title(f"{p_species}\n{q_species}\n(missing one version)", fontsize=8.5)
            ax.axis("off")
            continue

        p_avg_orig = np.mean([plant_density_orig[(p_species, *b)] for b in shared_orig], axis=0)
        p_avg_ppe = np.mean([plant_density_ppe[(p_species, *b)] for b in shared_ppe], axis=0)
        q_avg = np.mean([pollinator_density[(q_species, *b)] for b in shared_ppe], axis=0)

        x_smooth = np.linspace(0, N_WEEKS - 1, 300)
        p_orig_smooth = gaussian_kde(weeks, weights=p_avg_orig, bw_method=0.20)(x_smooth)
        p_ppe_smooth = gaussian_kde(weeks, weights=p_avg_ppe, bw_method=0.08)(x_smooth)
        q_smooth = gaussian_kde(weeks, weights=q_avg, bw_method=0.20)(x_smooth)

        ax.plot(x_smooth, p_orig_smooth, color=ORIG_COLOR, linewidth=1.8, linestyle="--", label="Plant (observed)")
        ax.plot(x_smooth, p_ppe_smooth, color=PPE_COLOR, linewidth=1.8, linestyle=":", label="Plant (PPE)")
        ax.plot(x_smooth, q_smooth, color=POLLINATOR_COLOR, linewidth=1.8, label="Pollinator")
        ax.fill_between(x_smooth, np.minimum(p_ppe_smooth, q_smooth), color=OVERLAP_COLOR, alpha=0.35)

        ax.set_title(
            f"{p_species} / {q_species}\norig={row['overlap_weighted_mean_orig']:.3f}, "
            f"ppe={row['overlap_weighted_mean_ppe']:.3f}", fontsize=8.5
        )
        ax.set_xlabel("Week of year", fontsize=8)
        ax.set_xticks([0, 13, 26, 39, 51])
        ax.tick_params(labelsize=7.5)
        if i == 0:
            ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Saved {output_path.name}")


def build_plant_density_orig(top100, plants, min_obs=MIN_OBS, bin_size=BIN_SIZE_TEMPORAL):
    """Identical to Step 8's original plant-side density build, for the
    disagreement-grid figure's 'observed' curve."""
    target_plants = set(top100["species"].unique())
    plants_f = plants[plants["species"].isin(target_plants)].copy()
    plants_f["lat_bin"] = np.floor(plants_f["lat"] / bin_size) * bin_size
    plants_f["lon_bin"] = np.floor(plants_f["lon"] / bin_size) * bin_size

    plant_bin_counts = plants_f.groupby(["species", "lat_bin", "lon_bin"]).size().reset_index(name="n_obs")
    keep = set(zip(
        plant_bin_counts[plant_bin_counts["n_obs"] >= min_obs]["species"],
        plant_bin_counts[plant_bin_counts["n_obs"] >= min_obs]["lat_bin"],
        plant_bin_counts[plant_bin_counts["n_obs"] >= min_obs]["lon_bin"],
    ))
    plants_f["_key"] = list(zip(plants_f["species"], plants_f["lat_bin"], plants_f["lon_bin"]))
    plants_f = plants_f[plants_f["_key"].isin(keep)]

    def weekly_density(doy_values, n_weeks=N_WEEKS):
        weeks = np.clip((doy_values - 1) // 7, 0, n_weeks - 1).astype(int)
        counts = np.bincount(weeks, minlength=n_weeks)
        total = counts.sum()
        return counts / total if total > 0 else counts

    return {
        key: weekly_density(group["doy"].values)
        for key, group in plants_f.groupby(["species", "lat_bin", "lon_bin"])
    }


if __name__ == "__main__":
    from importlib import import_module

    comparison = pd.read_csv(OUTPUTS_DIR / "ppe_vs_original_comparison.csv")

    lims = plot_scatter(comparison, OUTPUTS_DIR / "ppe_vs_original_scatter.png")
    plot_distribution(comparison, lims, OUTPUTS_DIR / "ppe_vs_original_distribution.png")

    # Rebuild the density dicts needed for the disagreement grid
    overlap_mod = import_module("03_compute_ppe_overlap")
    top100 = pd.read_csv(DATA_DIR / "top100_jaccard_pairs.csv")
    pollinator_obs = pd.read_csv(DATA_DIR / "pollinator_observations_v2.csv")
    ppe_1p5 = pd.read_parquet(PPE_DATA_DIR / "ppe_1p5_aggregated.parquet")
    plants = pd.read_parquet(DATA_DIR / "plant_flowering_events.parquet")

    plant_density_ppe = overlap_mod.build_plant_density_ppe(ppe_1p5)
    pollinator_density, _ = overlap_mod.build_pollinator_density(top100, pollinator_obs)
    plant_density_orig = build_plant_density_orig(top100, plants)

    plot_disagreement_grid(
        comparison, plant_density_orig, plant_density_ppe, pollinator_density,
        OUTPUTS_DIR / "ppe_vs_original_disagreement_grid.png",
    )
