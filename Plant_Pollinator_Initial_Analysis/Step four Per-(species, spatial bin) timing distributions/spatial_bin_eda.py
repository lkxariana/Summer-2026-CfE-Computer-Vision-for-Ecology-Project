"""EDA on spatial bin size for plant flowering events.

For each candidate bin size, report the number of bins, the percentage with
at least 10 observations, and the median observation count per bin. This
informs the choice of spatial resolution for downstream timing-distribution
analysis.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PLANT_PATH = Path("/scratch/ariana.l/plant_flowering_events.parquet")
CANDIDATE_BIN_SIZES = [0.25, 0.5, 1.0, 2.0]


def summarize_bins(df: pd.DataFrame, bin_size: float) -> tuple[int, float, float]:
    df = df.copy()
    df["lat_bin"] = (df["lat"] // bin_size) * bin_size
    df["lon_bin"] = (df["lon"] // bin_size) * bin_size
    obs_per_bin = df.groupby(["lat_bin", "lon_bin"]).size()
    n_bins = len(obs_per_bin)
    pct_ge10 = (obs_per_bin >= 10).mean() * 100
    median_obs = obs_per_bin.median()
    return n_bins, pct_ge10, median_obs


def main() -> None:
    if not PLANT_PATH.exists():
        raise FileNotFoundError(f"Plant flowering events not found at {PLANT_PATH}")

    plants = pd.read_parquet(PLANT_PATH)
    print(f"Loaded {len(plants):,} plant flowering records.")
    print("-" * 70)
    print(f"{'bin_size':>10} | {'n_bins':>8} | {'% bins ≥10 obs':>16} | {'median obs/bin':>16}")
    print("-" * 70)

    for bin_size in CANDIDATE_BIN_SIZES:
        n_bins, pct, median = summarize_bins(plants, bin_size)
        print(f"{bin_size:>10.2f}° | {n_bins:>8,} | {pct:>15.1f}% | {median:>16.0f}")


if __name__ == "__main__":
    main()
