"""Generate density maps for plants, pollinators, and their overlap across CONUS.

Produces three PNGs:
- plant_map.png: plant flowering density per spatial bin
- pollinator_map.png: pollinator observation density per spatial bin
- overlap_map.png: both layers overlaid with top 10 high-overlap bins highlighted
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import Normalize

PLANT_PATH = Path("/scratch/ariana.l/plant_flowering_events.parquet")
POLLINATOR_PATH = Path("/scratch/ariana.l/pollinator_observations.csv")
OUTPUT_DIR = Path("/scratch/ariana.l")

BIN_SIZE = 4.0
TOP_N_OVERLAP = 10
CONUS_LON = (-130, -60)
CONUS_LAT = (24, 50)


def add_bins(df: pd.DataFrame, bin_size: float) -> pd.DataFrame:
    df = df.copy()
    df["lat_bin"] = (df["lat"] // bin_size) * bin_size
    df["lon_bin"] = (df["lon"] // bin_size) * bin_size
    return df


def bin_counts(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["lat_bin", "lon_bin"]).size().reset_index(name="count")


def draw_density_map(
    counts: pd.DataFrame,
    cmap_name: str,
    title: str,
    output_path: Path,
    colorbar_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 8))
    norm = Normalize(vmin=0, vmax=counts["count"].max())
    cmap = plt.get_cmap(cmap_name)

    for _, row in counts.iterrows():
        rect = plt.Rectangle(
            (row["lon_bin"], row["lat_bin"]),
            BIN_SIZE,
            BIN_SIZE,
            color=cmap(norm(row["count"])),
            alpha=0.8,
        )
        ax.add_patch(rect)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    plt.colorbar(sm, ax=ax, label=colorbar_label)

    ax.set_xlim(*CONUS_LON)
    ax.set_ylim(*CONUS_LAT)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved {output_path}")


def draw_overlap_map(
    plant_counts: pd.DataFrame,
    pol_counts: pd.DataFrame,
    output_path: Path,
) -> None:
    overlap = plant_counts.merge(
        pol_counts, on=["lat_bin", "lon_bin"], suffixes=("_plant", "_pol")
    )
    overlap["score"] = (
        overlap["count_plant"] / overlap["count_plant"].max()
    ) * (overlap["count_pol"] / overlap["count_pol"].max())
    top = overlap.sort_values("score", ascending=False).head(TOP_N_OVERLAP).reset_index(drop=True)

    plant_norm = Normalize(vmin=0, vmax=plant_counts["count"].max())
    pol_norm = Normalize(vmin=0, vmax=pol_counts["count"].max())

    fig, ax = plt.subplots(figsize=(16, 9))

    for _, row in plant_counts.iterrows():
        rect = plt.Rectangle(
            (row["lon_bin"], row["lat_bin"]),
            BIN_SIZE,
            BIN_SIZE,
            color=plt.cm.Greens(plant_norm(row["count"])),
            alpha=0.5,
        )
        ax.add_patch(rect)

    for _, row in pol_counts.iterrows():
        rect = plt.Rectangle(
            (row["lon_bin"], row["lat_bin"]),
            BIN_SIZE,
            BIN_SIZE,
            color=plt.cm.Reds(pol_norm(row["count"])),
            alpha=0.5,
        )
        ax.add_patch(rect)

    for idx, row in top.iterrows():
        rect = plt.Rectangle(
            (row["lon_bin"], row["lat_bin"]),
            BIN_SIZE,
            BIN_SIZE,
            fill=False,
            edgecolor="blue",
            linewidth=2,
        )
        ax.add_patch(rect)
        ax.text(
            row["lon_bin"] + BIN_SIZE / 2,
            row["lat_bin"] + BIN_SIZE / 2,
            f"#{idx + 1}",
            color="blue",
            fontsize=8,
            ha="center",
            va="center",
            fontweight="bold",
        )

    ax.set_xlim(*CONUS_LON)
    ax.set_ylim(*CONUS_LAT)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Plant vs Pollinator Density Overlap — Top {TOP_N_OVERLAP} bins highlighted")

    sm_plant = plt.cm.ScalarMappable(cmap="Greens", norm=plant_norm)
    sm_pol = plt.cm.ScalarMappable(cmap="Reds", norm=pol_norm)
    cbar1 = plt.colorbar(sm_plant, ax=ax, location="left", shrink=0.6, pad=0.01)
    cbar1.set_label("Plant count")
    cbar2 = plt.colorbar(sm_pol, ax=ax, location="right", shrink=0.6, pad=0.01)
    cbar2.set_label("Pollinator count")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved {output_path}")

    print(f"\nTop {TOP_N_OVERLAP} overlap bins:")
    print(top[["lat_bin", "lon_bin", "count_plant", "count_pol", "score"]])


def main() -> None:
    if not PLANT_PATH.exists():
        raise FileNotFoundError(f"Plant flowering events not found at {PLANT_PATH}")
    if not POLLINATOR_PATH.exists():
        raise FileNotFoundError(f"Pollinator observations not found at {POLLINATOR_PATH}")

    plants = add_bins(pd.read_parquet(PLANT_PATH), BIN_SIZE)
    pollinators = add_bins(pd.read_csv(POLLINATOR_PATH), BIN_SIZE)

    plant_counts = bin_counts(plants)
    pol_counts = bin_counts(pollinators)

    draw_density_map(
        plant_counts,
        cmap_name="YlGn",
        title=f"All Plant Flowering Observations per {BIN_SIZE}°×{BIN_SIZE}° bin",
        output_path=OUTPUT_DIR / "plant_map.png",
        colorbar_label="Observation count",
    )
    draw_density_map(
        pol_counts,
        cmap_name="YlOrRd",
        title=f"All Pollinator Observations per {BIN_SIZE}°×{BIN_SIZE}° bin",
        output_path=OUTPUT_DIR / "pollinator_map.png",
        colorbar_label="Observation count",
    )
    draw_overlap_map(plant_counts, pol_counts, OUTPUT_DIR / "overlap_map.png")


if __name__ == "__main__":
    main()
