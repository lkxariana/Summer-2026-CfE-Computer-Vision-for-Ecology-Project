"""
03_compute_delta.py
Stage 4 Visualization — Compute per-bin, per-week PPE temporal delta

Delta = min(flowering_curve_value, mean_activity_curve_value)
This is the raw temporal overlap signal, computed without calling the classifier.

Requires variables from 01_load_data.py and bin caches from 02_compute_predictions.py.
Outputs: delta_df_weeks1_52.parquet
"""

import numpy as np
import pandas as pd

PLANT = 'Achillea millefolium'
WEEKS = list(range(1, 53))

print("Computing delta per bin per week...")
delta_results = []

for week in WEEKS:
    if week % 8 == 0:
        print(f"  week {week}/52...")
    f_w = flowering_curves.loc[PLANT, week] if week in flowering_curves.columns else 0

    for _, cell_row in achillea_grid.iterrows():
        bin_key = cell_row['bin']
        pol_N_list = bin_pol_N_cache.get(bin_key, [])

        if not pol_N_list:
            delta_results.append({
                'lat': cell_row['centroid_lat'],
                'lon': cell_row['centroid_lon'],
                'week': week,
                'delta': np.nan
            })
            continue

        bin_deltas = []
        for pol, N_val in pol_N_list:
            a_w = activity_curves.loc[pol, week] if week in activity_curves.columns else 0
            bin_deltas.append(min(f_w, a_w))

        delta_results.append({
            'lat': cell_row['centroid_lat'],
            'lon': cell_row['centroid_lon'],
            'week': week,
            'delta': np.mean(bin_deltas)
        })

delta_df = pd.DataFrame(delta_results)
delta_df.to_parquet(BASE + "delta_df_weeks1_52.parquet", index=False)
print(f"\nDone! Shape: {delta_df.shape}")
print(delta_df.dropna().describe())
