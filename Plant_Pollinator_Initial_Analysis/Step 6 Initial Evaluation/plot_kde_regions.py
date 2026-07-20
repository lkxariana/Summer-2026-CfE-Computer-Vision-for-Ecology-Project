"""
plot_kde_regions.py

Generates a single figure with three columns (East, California, Desert Southwest),
each column showing KDE-smoothed phenological overlap curves for all surviving
plant-pollinator edges in that region.

Each panel shows:
  - Plant curve with observation count (n=)
  - Pollinator curve with observation count (n=)
  - Shaded overlap region with mean overlap coefficient
  - Region label at top of each column

Color scheme:
  - East:             ruby red / coral red / light red
  - California:       dark green / medium green / light green
  - Desert Southwest: dark orange / amber / light yellow

Output: phenology_kde_regions_annotated.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# ── Parameters ────────────────────────────────────────────────────────────────
BIN_SIZE = 1.5
MIN_OBS  = 10
N_WEEKS  = 52
BW       = 0.20   # KDE bandwidth — increase for smoother curves

# ── Region definitions ────────────────────────────────────────────────────────
# Each region defines a lat/lon bounding box and a color scheme.
# Colors: plant line, pollinator line, overlap fill, column header label.
# Pacific Northwest is excluded — no surviving edges (target species absent).
regions = {
    'East': {
        'lat': (35.5, 44.5), 'lon': (-88.0, -71.0),
        'plant_color':   '#9B1B30',   # ruby red
        'pol_color':     '#FF6B6B',   # coral red
        'overlap_color': '#FFB3B3',   # light red
        'label_color':   '#9B1B30',
    },
    'California': {
        'lat': (32.5, 38.5), 'lon': (-123.0, -117.0),
        'plant_color':   '#1B5E20',   # dark green
        'pol_color':     '#66BB6A',   # medium green
        'overlap_color': '#C8E6C9',   # light green
        'label_color':   '#1B5E20',
    },
    'Desert Southwest': {
        'lat': (28.0, 36.0), 'lon': (-120.0, -103.0),
        'plant_color':   '#E65100',   # dark orange
        'pol_color':     '#FFB300',   # amber
        'overlap_color': '#FFE082',   # light yellow
        'label_color':   '#E65100',
    },
}

# ── Load data ─────────────────────────────────────────────────────────────────
plants      = pd.read_parquet('/scratch/ariana.l/plant_flowering_events.parquet')
pollinators = pd.read_csv('/scratch/ariana.l/pollinator_observations_v2.csv')
df_results  = pd.read_csv('/scratch/ariana.l/overlap_coefficients.csv')

# Add spatial bin and week columns
plants['lat_bin'] = (plants['lat'] // BIN_SIZE) * BIN_SIZE
plants['lon_bin'] = (plants['lon'] // BIN_SIZE) * BIN_SIZE
pollinators['lat_bin'] = (pollinators['lat'] // BIN_SIZE) * BIN_SIZE
pollinators['lon_bin'] = (pollinators['lon'] // BIN_SIZE) * BIN_SIZE
plants['week']      = np.minimum(((plants['doy'] - 1) // 7) + 1, N_WEEKS)
pollinators['week'] = np.minimum(((pollinators['doy'] - 1) // 7) + 1, N_WEEKS)


def weekly_density(df, species_col, species, lat_bin, lon_bin):
    """Return normalized 52-week histogram for a species in a spatial bin."""
    subset = df[
        (df[species_col] == species) &
        (df['lat_bin'] == lat_bin) &
        (df['lon_bin'] == lon_bin)
    ]
    if len(subset) < MIN_OBS:
        return None
    counts = np.zeros(N_WEEKS)
    for w, c in subset['week'].value_counts().items():
        counts[int(w) - 1] = c
    return counts / counts.sum()


# ── Filter df_results per region ─────────────────────────────────────────────
region_edges = {}
region_dfs   = {}
for region_name, props in regions.items():
    df_region = df_results[
        (df_results['lat_bin'] >= props['lat'][0]) &
        (df_results['lat_bin'] <  props['lat'][1]) &
        (df_results['lon_bin'] >= props['lon'][0]) &
        (df_results['lon_bin'] <  props['lon'][1])
    ]
    region_edges[region_name] = df_region[['plant', 'pollinator']].drop_duplicates().values
    region_dfs[region_name]   = df_region
    print(f"{region_name}: {len(df_region)} edge×bin rows, "
          f"{len(region_edges[region_name])} edges")

# ── Build figure ──────────────────────────────────────────────────────────────
max_edges = max(len(e) for e in region_edges.values())
n_regions = len(regions)
weeks        = np.arange(1, N_WEEKS + 1)
weeks_smooth = np.linspace(1, N_WEEKS, 300)

fig, axes = plt.subplots(
    max_edges, n_regions,
    figsize=(7 * n_regions, 4 * max_edges + 1)
)

# Region column headers via fig.text (more reliable than ax.set_title)
col_positions = [1/(2*n_regions) + i/n_regions for i in range(n_regions)]
for col, (region_name, props) in enumerate(regions.items()):
    fig.text(
        col_positions[col], 0.98, region_name,
        ha='center', va='top',
        fontsize=14, fontweight='bold',
        color=props['label_color'],
        transform=fig.transFigure
    )

# ── Plot each edge per region ─────────────────────────────────────────────────
for col, (region_name, props) in enumerate(regions.items()):
    df_region = region_dfs[region_name]
    edges     = region_edges[region_name]

    for row, (plant, pollinator) in enumerate(edges):
        ax = axes[row, col]

        edge_bins = df_region[
            (df_region['plant']      == plant) &
            (df_region['pollinator'] == pollinator)
        ][['lat_bin', 'lon_bin']].values

        p_curves, q_curves = [], []
        n_plant_total, n_pol_total = 0, 0

        for lat_b, lon_b in edge_bins:
            p = weekly_density(plants, 'species', plant, lat_b, lon_b)
            q = weekly_density(pollinators, 'pollinator_species',
                               pollinator, lat_b, lon_b)
            if p is not None and q is not None:
                p_curves.append(p)
                q_curves.append(q)
                n_plant_total += len(plants[
                    (plants['species']  == plant) &
                    (plants['lat_bin']  == lat_b) &
                    (plants['lon_bin']  == lon_b)
                ])
                n_pol_total += len(pollinators[
                    (pollinators['pollinator_species'] == pollinator) &
                    (pollinators['lat_bin'] == lat_b) &
                    (pollinators['lon_bin'] == lon_b)
                ])

        if not p_curves:
            ax.set_visible(False)
            continue

        mean_overlap = df_region[
            (df_region['plant']      == plant) &
            (df_region['pollinator'] == pollinator)
        ]['overlap'].mean()

        p_mean   = np.mean(p_curves, axis=0)
        q_mean   = np.mean(q_curves, axis=0)
        p_smooth = gaussian_kde(weeks, weights=p_mean, bw_method=BW)(weeks_smooth)
        q_smooth = gaussian_kde(weeks, weights=q_mean, bw_method=BW)(weeks_smooth)

        ax.plot(weeks_smooth, p_smooth,
                color=props['plant_color'], linewidth=1.8,
                label=f'{plant} (n={n_plant_total:,})')
        ax.plot(weeks_smooth, q_smooth,
                color=props['pol_color'], linewidth=1.8,
                label=f'{pollinator} (n={n_pol_total:,})')
        ax.fill_between(
            weeks_smooth, np.minimum(p_smooth, q_smooth),
            alpha=0.5, color=props['overlap_color'],
            label=f'overlap = {mean_overlap:.3f}'
        )
        ax.set_title(f'{plant}\n× {pollinator}', fontsize=8)
        ax.set_xlabel('Week of Year', fontsize=7)
        ax.set_ylabel('Relative Frequency', fontsize=7)
        ax.legend(fontsize=6)

    # Hide unused rows
    for row in range(len(edges), max_edges):
        axes[row, col].set_visible(False)

plt.suptitle(
    'Phenological Overlap by Region — KDE Smoothed (bw=0.20)',
    fontsize=13, y=1.01
)
plt.tight_layout(rect=[0, 0, 1, 0.97])

output_path = '/scratch/ariana.l/phenology_kde_regions_annotated.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {output_path}")
