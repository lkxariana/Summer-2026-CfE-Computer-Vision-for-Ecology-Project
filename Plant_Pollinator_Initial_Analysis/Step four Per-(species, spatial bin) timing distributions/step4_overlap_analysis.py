"""Step 4 — spatial binning, density mapping, and overlap region identification.

Runs the full Step 4 pipeline:
  1. EDA on spatial bin sizes for plant observations.
  2. Plant and pollinator density maps across CONUS.
  3. Top 20 highest-overlap bins (score = normalized plant × pollinator counts),
     reverse-geocoded to US states.
  4. Group the top 20 bins into three geographic regions and overlay region
     bounding boxes on the combined density map.

Outputs are written to OUTPUT_DIR; expected files are listed in README_step4.md.
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from geopy.geocoders import Nominatim
from matplotlib.colors import Normalize

PLANT_PATH = Path("/scratch/ariana.l/plant_flowering_events.parquet")
POLLINATOR_PATH = Path("/scratch/ariana.l/pollinator_observations.csv")
OUTPUT_DIR = Path("/scratch/ariana.l")

BIN_SIZE = 0.5
CANDIDATE_BIN_SIZES = [0.25, 0.5, 1.0, 2.0]
TOP_N_OVERLAP = 20
CONUS_LON = (-130, -60)
CONUS_LAT = (24, 50)
DPI = 300

# Bins that fall on coastal water but should be attributed to the adjacent state.
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


# ----------------------------------------------------------------------------
# 4.1 EDA on bin size
# ----------------------------------------------------------------------------

def run_bin_size_eda(plants: pd.DataFrame) -> None:
    print("Bin size EDA on plant flowering observations:")
    print("-" * 70)
    print(f"{'bin_size':>10} | {'n_bins':>8} | {'% bins ≥10 obs':>16} | {'median obs/bin':>16}")
    print("-" * 70)
    for size in CANDIDATE_BIN_SIZES:
        binned = add_bins(plants, size)
        obs_per_bin = binned.groupby(["lat_bin", "lon_bin"]).size()
        n_bins = len(obs_per_bin)
        pct_ge10 = (obs_per_bin >= 10).mean() * 100
        median_obs = obs_per_bin.median()
        print(f"{size:>10.2f}° | {n_bins:>8,} | {pct_ge10:>15.1f}% | {median_obs:>16.0f}")
    print()


# ----------------------------------------------------------------------------
# 4.2 Density maps
# ----------------------------------------------------------------------------

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
    plt.savefig(output_path, dpi=DPI)
    plt.close(fig)
    print(f"Saved {output_path}")


# ----------------------------------------------------------------------------
# 4.3 Top 20 overlap bins + reverse geocoding
# ----------------------------------------------------------------------------

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
        .head(TOP_N_OVERLAP)
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
        time.sleep(1)  # Nominatim asks for ≤1 request/second

    df = pd.DataFrame(rows)
    for rank, state in MANUAL_STATE_OVERRIDES.items():
        df.loc[df["rank"] == rank, "state"] = state
    return df


# ----------------------------------------------------------------------------
# 4.4 Group into regions and plot overlay
# ----------------------------------------------------------------------------

def assign_region(row: pd.Series) -> str:
    if row["lon_min"] >= -90:
        return "East"
    if row["lat_min"] >= 45:
        return "Pacific Northwest"
    return "California"


def compute_regions(top_with_states: pd.DataFrame) -> pd.DataFrame:
    top_with_states = top_with_states.copy()
    top_with_states["region"] = top_with_states.apply(assign_region, axis=1)
    return (
        top_with_states.groupby("region")
        .agg(
            lon_min=("lon_min", "min"),
            lon_max=("lon_max", "max"),
            lat_min=("lat_min", "min"),
            lat_max=("lat_max", "max"),
            n_bins=("rank", "count"),
        )
        .reset_index()
    )


def draw_overlap_map_with_regions(
    plant_counts: pd.DataFrame,
    pol_counts: pd.DataFrame,
    top: pd.DataFrame,
    regions: pd.DataFrame,
    output_path: Path,
) -> None:
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

    for _, row in top.iterrows():
        circle = plt.Circle(
            (row["lon_bin"] + BIN_SIZE / 2, row["lat_bin"] + BIN_SIZE / 2),
            BIN_SIZE / 2 * 1.1,
            fill=False,
            edgecolor="blue",
            linewidth=2,
        )
        ax.add_patch(circle)

    for _, region in regions.iterrows():
        rect = plt.Rectangle(
            (region["lon_min"], region["lat_min"]),
            region["lon_max"] - region["lon_min"],
            region["lat_max"] - region["lat_min"],
            fill=False,
            edgecolor="black",
            linewidth=2,
        )
        ax.add_patch(rect)
        ax.text(
            region["lon_min"],
            region["lat_max"] + 0.5,
            region["region"],
            fontsize=11,
            fontweight="bold",
            color="black",
        )

    ax.set_xlim(*CONUS_LON)
    ax.set_ylim(*CONUS_LAT)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(
        f"Plant vs Pollinator Density Overlap — Top {TOP_N_OVERLAP} bins in 3 regions"
    )

    sm_plant = plt.cm.ScalarMappable(cmap="Greens", norm=plant_norm)
    sm_pol = plt.cm.ScalarMappable(cmap="Reds", norm=pol_norm)
    cbar1 = plt.colorbar(sm_plant, ax=ax, location="left", shrink=0.6, pad=0.08)
    cbar1.set_label("Plant count")
    cbar2 = plt.colorbar(sm_pol, ax=ax, location="right", shrink=0.6, pad=0.01)
    cbar2.set_label("Pollinator count")

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI)
    plt.close(fig)
    print(f"Saved {output_path}")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> None:
    if not PLANT_PATH.exists():
        raise FileNotFoundError(f"Plant flowering events not found at {PLANT_PATH}")
    if not POLLINATOR_PATH.exists():
        raise FileNotFoundError(f"Pollinator observations not found at {POLLINATOR_PATH}")

    plants_raw = pd.read_parquet(PLANT_PATH)
    pollinators_raw = pd.read_csv(POLLINATOR_PATH)

    # 4.1 EDA
    run_bin_size_eda(plants_raw)

    # 4.2 Density maps
    plants = add_bins(plants_raw, BIN_SIZE)
    pollinators = add_bins(pollinators_raw, BIN_SIZE)
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

    # 4.3 Top 20 overlap bins + states
    top = compute_top_overlap(plant_counts, pol_counts)
    top_with_states = reverse_geocode_states(top)
    top_with_states.to_csv(OUTPUT_DIR / "top20_overlap_bins.csv", index=False)
    print(f"Saved {OUTPUT_DIR / 'top20_overlap_bins.csv'}")

    # 4.4 Regions + overlay map
    regions = compute_regions(top_with_states)
    regions.to_csv(OUTPUT_DIR / "top20_regions.csv", index=False)
    print(f"Saved {OUTPUT_DIR / 'top20_regions.csv'}")
    print(regions)

    draw_overlap_map_with_regions(
        plant_counts,
        pol_counts,
        top,
        regions,
        OUTPUT_DIR / "overlap_map_regions.png",
    )


if __name__ == "__main__":
    main()
