"""
Visualize KDE-smoothed phenology curves for each surviving plant-pollinator
edge in the top 20 bins.

For each edge, averages 52-week density curves across all top-20 bins where
the edge appears, then plots plant and pollinator curves with the overlap
region shaded.

Output: phenology_grid_kde.png
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from math import ceil

# ----- Parameters -----
BIN_SIZE = 1.5
MIN_OBS = 10
N_WEEKS = 52
BW_METHOD = 0.20  # KDE bandwidth — lower = sharper, higher = smoother

# ----- Load data and rebuild bin/week columns -----
plants = pd.read_parquet('/scratch/ariana.l/plant_flowering_events.parquet')
pollinators = pd.read_csv('/scratch/ariana.l/pollinator_observations_v2.csv')
df_top = pd.read_csv('/scratch/ariana.l/overlap_coefficients_top20.csv')

plants['lat_bin'] = (plants['lat'] // BIN_SIZE) * BIN_SIZE
plants['lon_bin'] = (plants['lon'] // BIN_SIZE) * BIN_SIZE
pollinators['lat_bin'] = (pollinators['lat'] // BIN_SIZE) * BIN_SIZE
pollinators['lon_bin'] = (pollinators['lon'] // BIN_SIZE) * BIN_SIZE
plants['week'] = np.minimum(((plants['doy'] - 1) // 7) + 1, N_WEEKS)
pollinators['week'] = np.minimum(((pollinators['doy'] - 1) // 7) + 1, N_WEEKS)


def weekly_density(df, species_col, species, lat_bin, lon_bin):
    """Return normalized weekly histogram (52 weeks, sums to 1)."""
    subset = df[
        (df[species_col] == species)
        & (df['lat_bin'] == lat_bin)
        & (df['lon_bin'] == lon_bin)
    ]
    if len(subset) < MIN_OBS:
        return None
    counts = np.zeros(N_WEEKS)
    for w, c in subset['week'].value_counts().items():
        counts[int(w) - 1] = c
    return counts / counts.sum()


# ----- Plot grid -----
edges_top = df_top[['plant', 'pollinator']].drop_duplicates().values
n = len(edges_top)
ncols = 2
nrows = ceil(n / ncols)

fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 4))
axes = axes.flatten()
weeks = np.arange(1, N_WEEKS + 1)
weeks_smooth = np.linspace(1, N_WEEKS, 300)

for i, (plant, pollinator) in enumerate(edges_top):
    ax = axes[i]
    edge_bins = df_top[
        (df_top['plant'] == plant) & (df_top['pollinator'] == pollinator)
    ][['lat_bin', 'lon_bin']].values

    p_curves, q_curves = [], []
    for lat_b, lon_b in edge_bins:
        p = weekly_density(plants, 'species', plant, lat_b, lon_b)
        q = weekly_density(
            pollinators, 'pollinator_species', pollinator, lat_b, lon_b
        )
        if p is not None and q is not None:
            p_curves.append(p)
            q_curves.append(q)

    if not p_curves:
        continue

    p_mean = np.mean(p_curves, axis=0)
    q_mean = np.mean(q_curves, axis=0)

    p_kde = gaussian_kde(weeks, weights=p_mean, bw_method=BW_METHOD)
    q_kde = gaussian_kde(weeks, weights=q_mean, bw_method=BW_METHOD)
    p_smooth = p_kde(weeks_smooth)
    q_smooth = q_kde(weeks_smooth)

    ax.plot(weeks_smooth, p_smooth, color='green', label=f'{plant}')
    ax.plot(weeks_smooth, q_smooth, color='orange', label=f'{pollinator}')
    ax.fill_between(
        weeks_smooth,
        np.minimum(p_smooth, q_smooth),
        alpha=0.3,
        color='purple',
        label='overlap',
    )
    ax.set_title(f'{plant}\n× {pollinator}', fontsize=9)
    ax.set_xlabel('Week of Year', fontsize=8)
    ax.set_ylabel('Relative Frequency', fontsize=8)
    ax.legend(fontsize=7)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle(
    'Phenological Overlap — KDE Smoothed (Top 20 Bins)', fontsize=12, y=1.01
)
plt.tight_layout()
plt.savefig(
    '/scratch/ariana.l/phenology_grid_kde.png', dpi=150, bbox_inches='tight'
)
print("Saved: /scratch/ariana.l/phenology_grid_kde.png")
