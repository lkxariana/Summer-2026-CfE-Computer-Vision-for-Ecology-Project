#!/usr/bin/env python
"""QC + density visuals for the GBIF pollinator download.

Streams the GBIF SIMPLE_CSV archive (tab-delimited, possibly inside the .zip) in
chunks so memory stays flat at any row count, and writes to data/gbif_qc/:

  summary_per_order.csv      rows + unique taxa per order vs. expected counts
  spatial_density_<order>.png  CONUS map, record counts on the 0.5deg analysis grid
  spatial_density_combined.png 5-panel version of the above
  temporal_density.png         week-of-year density per order (normalized)

Usage:
    python gbif_qc_report.py /path/to/0012345-250101.zip
    python gbif_qc_report.py /path/to/occurrence.csv      # already-extracted TSV
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# --- config -----------------------------------------------------------------
EXPECTED = {  # unique taxa you expect per order (for reconciliation, not enforcement)
    "Hemiptera": 813,
    "Passeriformes": 173,
    "Thysanoptera": 9,
    "Neuroptera": 18,
    "Chiroptera": 9,
}
ORDERS = list(EXPECTED)

# CONUS box + 0.5deg grid, matched to the Part 1 spatial binning (BIN_SIZE=0.5)
LON_MIN, LON_MAX = -125.0, -66.5
LAT_MIN, LAT_MAX = 24.5, 49.5
BIN = 0.5
LON_EDGES = np.arange(LON_MIN, LON_MAX + BIN, BIN)
LAT_EDGES = np.arange(LAT_MIN, LAT_MAX + BIN, BIN)
N_WEEKS = 52

USECOLS = ["order", "species", "taxonKey", "decimalLatitude",
           "decimalLongitude", "day", "month", "year"]
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gbif_qc")


def empty_acc():
    return {
        "n_rows": 0,
        "species": set(),
        "taxonkeys": set(),
        "n_no_species": 0,
        "week": np.zeros(N_WEEKS, dtype=np.int64),
        "spatial": np.zeros((len(LAT_EDGES) - 1, len(LON_EDGES) - 1), dtype=np.int64),
    }


def doy_to_week(doy):
    return np.minimum(((doy - 1) // 7) + 1, N_WEEKS).astype(int)


def process(path):
    acc = {o: empty_acc() for o in ORDERS}
    # pandas reads the single member of a .zip directly via compression="zip".
    compression = "zip" if path.lower().endswith(".zip") else "infer"
    reader = pd.read_csv(
        path, sep="\t", usecols=USECOLS, compression=compression,
        chunksize=500_000, on_bad_lines="skip", low_memory=False,
        dtype={"day": "Int64", "month": "Int64", "year": "Int64"},
    )
    for ci, chunk in enumerate(reader):
        chunk = chunk[chunk["order"].isin(ORDERS)]
        for o, g in chunk.groupby("order"):
            a = acc[o]
            a["n_rows"] += len(g)
            sp = g["species"].dropna()
            a["species"].update(sp.unique().tolist())
            a["n_no_species"] += int(g["species"].isna().sum())
            a["taxonkeys"].update(g["taxonKey"].dropna().unique().tolist())

            # temporal: day-of-year -> week
            dt = pd.to_datetime(
                dict(year=g["year"], month=g["month"], day=g["day"]),
                errors="coerce",
            )
            doy = dt.dt.dayofyear.dropna().to_numpy()
            if doy.size:
                w = doy_to_week(doy)
                a["week"] += np.bincount(w - 1, minlength=N_WEEKS)[:N_WEEKS]

            # spatial: 0.5deg histogram, accumulated
            ll = g[["decimalLatitude", "decimalLongitude"]].dropna()
            if len(ll):
                h, _, _ = np.histogram2d(
                    ll["decimalLatitude"].to_numpy(),
                    ll["decimalLongitude"].to_numpy(),
                    bins=[LAT_EDGES, LON_EDGES],
                )
                a["spatial"] += h.astype(np.int64)
        print(f"  chunk {ci}: cumulative rows = {sum(a['n_rows'] for a in acc.values()):,}")
    return acc


def write_summary(acc):
    rows = []
    for o in ORDERS:
        a = acc[o]
        n_sp = len(a["species"])
        rows.append({
            "order": o,
            "n_rows": a["n_rows"],
            "n_unique_species": n_sp,
            "n_unique_taxonKey": len(a["taxonkeys"]),
            "n_rows_no_species": a["n_no_species"],
            "expected_taxa": EXPECTED[o],
            "delta_vs_expected": n_sp - EXPECTED[o],
        })
    df = pd.DataFrame(rows)
    path = os.path.join(OUT_DIR, "summary_per_order.csv")
    df.to_csv(path, index=False)
    print("\n" + df.to_string(index=False))
    print(f"\nwrote {path}")
    return df


def _draw_map(ax, grid, title):
    ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.STATES, linewidth=0.3, edgecolor="0.6")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4)
    masked = np.ma.masked_equal(grid, 0)
    mesh = ax.pcolormesh(
        LON_EDGES, LAT_EDGES, masked, transform=ccrs.PlateCarree(),
        cmap="viridis", norm=matplotlib.colors.LogNorm(),
    )
    ax.set_title(title, fontsize=10)
    return mesh


def plot_spatial(acc):
    for o in ORDERS:
        fig = plt.figure(figsize=(8, 5))
        ax = plt.axes(projection=ccrs.PlateCarree())
        mesh = _draw_map(ax, acc[o]["spatial"], f"{o}  (n={acc[o]['n_rows']:,})")
        fig.colorbar(mesh, ax=ax, shrink=0.7, label="records / 0.5° cell")
        fig.savefig(os.path.join(OUT_DIR, f"spatial_density_{o}.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(16, 8),
                             subplot_kw={"projection": ccrs.PlateCarree()})
    for ax, o in zip(axes.ravel(), ORDERS):
        mesh = _draw_map(ax, acc[o]["spatial"], f"{o} (n={acc[o]['n_rows']:,})")
        fig.colorbar(mesh, ax=ax, shrink=0.6)
    axes.ravel()[-1].axis("off")
    fig.suptitle("GBIF record density, 0.5° grid (log scale)", fontsize=13)
    fig.savefig(os.path.join(OUT_DIR, "spatial_density_combined.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote spatial_density_*.png")


def plot_temporal(acc):
    fig, axes = plt.subplots(len(ORDERS), 1, figsize=(9, 1.7 * len(ORDERS)),
                             sharex=True)
    weeks = np.arange(1, N_WEEKS + 1)
    for ax, o in zip(axes, ORDERS):
        w = acc[o]["week"].astype(float)
        dens = w / w.sum() if w.sum() else w
        ax.fill_between(weeks, dens, step="mid", alpha=0.7)
        ax.set_ylabel(o, fontsize=8, rotation=0, ha="right", va="center")
        ax.set_yticks([])
    axes[-1].set_xlabel("week of year")
    axes[-1].set_xlim(1, N_WEEKS)
    fig.suptitle("Week-of-year activity density (normalized per order)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "temporal_density.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote temporal_density.png")


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: python gbif_qc_report.py <download.zip | occurrence.csv>")
    os.makedirs(OUT_DIR, exist_ok=True)
    acc = process(sys.argv[1])
    write_summary(acc)
    plot_spatial(acc)
    plot_temporal(acc)
    print(f"\nAll outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
