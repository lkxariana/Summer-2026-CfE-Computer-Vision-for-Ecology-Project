"""
Compute phenological overlap coefficients per (edge, spatial bin).

For each plant-pollinator edge and each 1.5° spatial bin where both species
have at least MIN_OBS observations, compute the area of intersection between
their normalized 52-week activity distributions.

Output: overlap_coefficients.csv, surviving_edges.csv, edge_overlap_summary.csv
"""

import pandas as pd
import numpy as np

# ----- Load data -----
plants = pd.read_parquet('/scratch/ariana.l/plant_flowering_events.parquet')
pollinators = pd.read_csv('/scratch/ariana.l/pollinator_observations_v2.csv')
edges = pd.read_csv(
    '/home/ariana.l/CfE2026CVforEcology/rawpollinatordata/plant_pollinator_edges.csv'
)

# ----- Parameters -----
BIN_SIZE = 1.5
MIN_OBS = 10
N_WEEKS = 52

# ----- Spatial bins (lower-left corner of 1.5° × 1.5° cell) -----
plants['lat_bin'] = (plants['lat'] // BIN_SIZE) * BIN_SIZE
plants['lon_bin'] = (plants['lon'] // BIN_SIZE) * BIN_SIZE
pollinators['lat_bin'] = (pollinators['lat'] // BIN_SIZE) * BIN_SIZE
pollinators['lon_bin'] = (pollinators['lon'] // BIN_SIZE) * BIN_SIZE

# ----- Week of year (1–52, clipping 53 to 52) -----
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


# ----- Compute overlap for every edge × bin -----
results = []
for _, edge in edges.iterrows():
    plant = edge['plant_species']
    pollinator = edge['pollinator_species']

    # Bins where each species has observations
    plant_bins = set(
        zip(
            plants[plants['species'] == plant]['lat_bin'],
            plants[plants['species'] == plant]['lon_bin'],
        )
    )
    pol_bins = set(
        zip(
            pollinators[pollinators['pollinator_species'] == pollinator]['lat_bin'],
            pollinators[pollinators['pollinator_species'] == pollinator]['lon_bin'],
        )
    )

    for (lat_b, lon_b) in plant_bins & pol_bins:
        p_density = weekly_density(plants, 'species', plant, lat_b, lon_b)
        q_density = weekly_density(
            pollinators, 'pollinator_species', pollinator, lat_b, lon_b
        )
        if p_density is None or q_density is None:
            continue
        overlap = np.minimum(p_density, q_density).sum()
        results.append({
            'plant': plant,
            'pollinator': pollinator,
            'lat_bin': lat_b,
            'lon_bin': lon_b,
            'overlap': overlap,
        })

df_results = pd.DataFrame(results)

# ----- Report and save -----
print(
    f"Total (edge, bin) combinations with both sides ≥{MIN_OBS} obs: "
    f"{len(df_results)}"
)
n_edges = df_results[['plant', 'pollinator']].drop_duplicates().shape[0]
print(f"Unique edges represented: {n_edges} out of {len(edges)}")
print("\nOverlap distribution:")
print(df_results['overlap'].describe())

df_results.to_csv('/scratch/ariana.l/overlap_coefficients.csv', index=False)

# Surviving edges
surviving_edges = (
    df_results[['plant', 'pollinator']].drop_duplicates().reset_index(drop=True)
)
surviving_edges.to_csv('/scratch/ariana.l/surviving_edges.csv', index=False)

# Per-edge summary (mean overlap aggregated across bins)
edge_overlap = (
    df_results.groupby(['plant', 'pollinator'])['overlap']
    .agg(mean_overlap='mean', std_overlap='std', n_bins='count')
    .reset_index()
    .sort_values('mean_overlap', ascending=False)
)
edge_overlap.to_csv('/scratch/ariana.l/edge_overlap_summary.csv', index=False)

print("\nSaved:")
print("  /scratch/ariana.l/overlap_coefficients.csv")
print("  /scratch/ariana.l/surviving_edges.csv")
print("  /scratch/ariana.l/edge_overlap_summary.csv")
