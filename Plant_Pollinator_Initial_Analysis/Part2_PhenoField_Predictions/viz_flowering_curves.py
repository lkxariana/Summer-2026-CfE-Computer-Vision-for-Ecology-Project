"""Visualize PhenoField flowering curves as circular KDEs over week-of-year.

Two figure types:
  A. per-location overlay  — several species with DISTINCT phenology at one
     important location (shows "different species -> different curves").
  B. latitudinal gradient  — one species across south->north locations
     (shows the flowering peak shifting later with latitude).

Curves are the normalized weekly flowering distribution (sum-to-1 over the
year) circularly smoothed (Gaussian, ~1.5 wk) = a circular KDE over
week-of-year, matching the Part-1 Step-4 representation.

Usage:
    python viz_flowering_curves.py --cells used --probe inat
"""
from __future__ import annotations
import argparse, os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

HERE = Path(__file__).resolve().parent
OUTDIR = Path(os.environ.get('PPE_OUT_DIR', HERE.parent / 'Part2_PhenoField_Outputs'))
OUTDATA = OUTDIR / 'data'
FIGS = OUTDIR / 'figures'

RAW_COL, PROBE_LABEL = 'inat_p_flowering', 'iNat 4-class probe'

# important locations (used_in_analysis cells) -> readable names
LOC_NAMES = {
    (38.75, -77.25): 'Washington DC', (39.25, -76.75): 'Baltimore MD',
    (42.25, -71.25): 'Boston MA', (40.75, -73.75): 'New York NY',
    (41.75, -87.75): 'Chicago IL', (44.25, -72.75): 'Vermont',
    (35.75, -78.75): 'Raleigh NC', (37.75, -122.25): 'SF Bay CA',
    (34.25, -118.25): 'Los Angeles CA', (32.75, -117.25): 'San Diego CA',
    (47.75, -122.25): 'Seattle WA',
}

# contrasting eastern-forest species spanning the season
CONTRAST = [
    'Sanguinaria canadensis',    # very early spring ephemeral
    'Erythronium americanum',    # early spring ephemeral
    'Mertensia virginica',       # spring
    'Aquilegia canadensis',      # late spring
    'Asclepias syriaca',         # early summer
    'Chamaenerion angustifolium',# summer
    'Amphicarpaea bracteata',    # late summer
]
# eastern locations where those species occur
EAST_LOCS = ['Washington DC', 'Boston MA', 'New York NY', 'Chicago IL', 'Vermont']
# latitudinal-gradient species + south->north transect
GRADIENT_SPECIES = ['Erythronium americanum', 'Asclepias syriaca']
GRADIENT_LOCS = ['Raleigh NC', 'Washington DC', 'New York NY', 'Boston MA', 'Vermont']

MONTH_STARTS = [0, 4.3, 8.6, 13, 17.3, 21.7, 26, 30.3, 34.7, 39, 43.3, 47.7]
MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


def kde(series_by_week, sigma=1.5):
    """52-week normalized circular KDE from a (week->value) lookup."""
    w = np.zeros(52)
    for wk, v in series_by_week.items():
        if 0 <= int(wk) < 52 and np.isfinite(v):
            w[int(wk)] = v
    w = gaussian_filter1d(w, sigma=sigma, mode='wrap')
    s = w.sum()
    return w / s if s > 0 else w


def name_of(lat, lon):
    return LOC_NAMES.get((round(lat, 2), round(lon, 2)))


def fig_per_location(df, raw_col, loc, species_list, probe_label, out):
    sub = df[df.loc_name == loc]
    if sub.empty:
        print(f"  [skip] {loc}: no data"); return
    fig, ax = plt.subplots(figsize=(9, 4.2))
    cmap = plt.cm.turbo(np.linspace(0.05, 0.95, len(species_list)))
    for sp, c in zip(species_list, cmap):
        s = sub[sub.species == sp]
        if s.empty:
            continue
        curve = kde(dict(zip(s.week, s[raw_col])))
        peak = int(np.argmax(curve))
        ax.plot(np.arange(52), curve, color=c, lw=2.1,
                label=f"$\\it{{{sp.replace(' ', '~')}}}$ (peak {MONTHS[min(11,int(peak/52*12))]})")
        ax.fill_between(np.arange(52), curve, alpha=0.06, color=c)
    ax.set_xticks(MONTH_STARTS); ax.set_xticklabels(MONTHS, fontsize=8)
    ax.set_xlim(0, 51); ax.set_ylim(bottom=0)
    ax.set_ylabel('flowering density (KDE)')
    ax.set_title(f"Predicted flowering curves — {loc}\n{probe_label}", fontsize=11)
    ax.legend(fontsize=7.5, loc='upper right', framealpha=0.85)
    plt.tight_layout(); fig.savefig(out, dpi=180, bbox_inches='tight'); plt.close(fig)
    print(f"  wrote {out.name}")


def fig_gradient(df, raw_col, species, locs, probe_label, out):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(locs)))
    plotted = 0
    for loc, c in zip(locs, cmap):
        s = df[(df.loc_name == loc) & (df.species == species)]
        if s.empty:
            continue
        curve = kde(dict(zip(s.week, s[raw_col])))
        ax.plot(np.arange(52), curve, color=c, lw=2.2, label=loc)
        ax.fill_between(np.arange(52), curve, alpha=0.06, color=c)
        plotted += 1
    if not plotted:
        plt.close(fig); print(f"  [skip] gradient {species}: no data"); return
    ax.set_xticks(MONTH_STARTS); ax.set_xticklabels(MONTHS, fontsize=8)
    ax.set_xlim(0, 51); ax.set_ylim(bottom=0)
    ax.set_ylabel('flowering density (KDE)')
    ax.set_title(f"Latitudinal gradient — $\\it{{{species.replace(' ', '~')}}}$ "
                 f"(south→north)\n{probe_label}", fontsize=11)
    ax.legend(fontsize=8, loc='upper right', framealpha=0.85)
    plt.tight_layout(); fig.savefig(out, dpi=180, bbox_inches='tight'); plt.close(fig)
    print(f"  wrote {out.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cells', choices=['used', 'all'], default='used')
    args = ap.parse_args()
    FIGS.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(OUTDATA / f'flowering_curves_{args.cells}.parquet')
    df['loc_name'] = [name_of(a, b) for a, b in zip(df.centroid_lat, df.centroid_lon)]

    print(f"[inat] {PROBE_LABEL}")
    for loc in EAST_LOCS:
        fig_per_location(df, RAW_COL, loc, CONTRAST, PROBE_LABEL,
                         FIGS / f'curves_{loc.replace(" ", "_")}_inat.png')
    for sp in GRADIENT_SPECIES:
        fig_gradient(df, RAW_COL, sp, GRADIENT_LOCS, PROBE_LABEL,
                     FIGS / f'gradient_{sp.split()[0]}_inat.png')


if __name__ == '__main__':
    main()
