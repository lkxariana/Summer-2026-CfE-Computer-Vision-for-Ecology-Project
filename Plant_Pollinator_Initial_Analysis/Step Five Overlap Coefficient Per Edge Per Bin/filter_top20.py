"""
Filter overlap results to the top 20 most observation-dense spatial bins.

The top 20 bins were originally defined at 0.5° resolution for geographic
specificity; this script maps them to their parent 1.5° bins (matching the
resolution used in overlap computation) and restricts df_results accordingly.

Output: df_top with overlap rows from only the top 20 bins.
"""

import pandas as pd

BIN_SIZE = 1.5

# ----- Load -----
df_results = pd.read_csv('/scratch/ariana.l/overlap_coefficients.csv')
top20 = pd.read_csv('/scratch/ariana.l/top20_overlap_bins.csv')

# ----- Map 0.5° top20 bins to their parent 1.5° bins -----
top20['lat_bin'] = (top20['lat_min'] // BIN_SIZE) * BIN_SIZE
top20['lon_bin'] = (top20['lon_min'] // BIN_SIZE) * BIN_SIZE
top20_bins = set(zip(top20['lat_bin'], top20['lon_bin']))

print(f"Mapped {len(top20)} top20 0.5° bins to {len(top20_bins)} unique 1.5° bins")

# ----- Filter -----
df_top = df_results[
    df_results.apply(
        lambda r: (r['lat_bin'], r['lon_bin']) in top20_bins, axis=1
    )
]

n_edges = df_top[['plant', 'pollinator']].drop_duplicates().shape[0]
print(f"Rows after filtering: {len(df_top)}")
print(f"Unique edges remaining: {n_edges}")

df_top.to_csv('/scratch/ariana.l/overlap_coefficients_top20.csv', index=False)
print("\nSaved: /scratch/ariana.l/overlap_coefficients_top20.csv")
