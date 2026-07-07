"""
02_compute_predictions.py
Stage 4 Visualization — Compute per-bin, per-week A2 and A3 predictions

Requires variables from 01_load_data.py to be loaded in the session.
Outputs: pred_df_weeks1_52.parquet
"""

import numpy as np
import pandas as pd

PLANT = 'Achillea millefolium'
BIN_SIZE = 0.5
WEEKS = list(range(1, 53))  # full year

vf_plant = Vf_arr[vf_idx[PLANT]]

# Build bin -> pollinator mapping
# Only include pollinators that have both a Vp embedding and an activity curve
print("Building bin -> pollinator mapping...")
bin_to_pollinators = {}
for bin_col in P.columns:
    pols = P.index[P[bin_col] == 1].tolist()
    pols = [p for p in pols if p in vp_idx and p in activity_curves.index]
    if pols:
        bin_to_pollinators[bin_col] = pols
print(f"  Bins with pollinators: {len(bin_to_pollinators):,}")

# Build base spatial grid from Achillea PPE surface (week-independent geometry)
achillea_grid = (
    achillea[achillea['week'] == 0][['centroid_lat', 'centroid_lon']]
    .drop_duplicates()
    .copy()
)
achillea_grid['bin'] = (
    ((achillea_grid['centroid_lat'] // BIN_SIZE) * BIN_SIZE).astype(str) + "_" +
    ((achillea_grid['centroid_lon'] // BIN_SIZE) * BIN_SIZE).astype(str)
)
print(f"  Grid cells: {len(achillea_grid):,}")

# Precompute A2 predictions and cache (pol, N) per bin
# A2 is week-independent so we only compute it once per bin
print("Precomputing A2 and caching N values...")
bin_a2_cache = {}
bin_pol_N_cache = {}

for _, cell_row in achillea_grid.iterrows():
    bin_key = cell_row['bin']
    if bin_key in bin_a2_cache:
        continue
    pols = bin_to_pollinators.get(bin_key, [])
    if not pols:
        bin_a2_cache[bin_key] = np.nan
        bin_pol_N_cache[bin_key] = []
        continue

    pol_N_list = []
    a2_preds = []
    for pol in pols:
        # N = number of shared bins between plant and pollinator
        # Use common-bin aligned matrices to avoid dimension mismatch
        N_val = int(F_common_arr[f_common_idx[PLANT]] @ P_common_arr[p_common_idx[pol]]) \
                if PLANT in f_common_idx and pol in p_common_idx else 0
        feat = np.concatenate([vf_plant, Vp_arr[vp_idx[pol]], [N_val]])
        a2_preds.append(clf_a2.predict_proba(feat.reshape(1, -1))[0, 1])
        pol_N_list.append((pol, N_val))

    bin_a2_cache[bin_key] = np.mean(a2_preds)
    bin_pol_N_cache[bin_key] = pol_N_list

# Compute A3 per week (adds temporal delta = min(flowering, activity))
print("Computing A3 per week (full year)...")
results = []
for week in WEEKS:
    if week % 8 == 0:
        print(f"  week {week}/52...")
    f_w = flowering_curves.loc[PLANT, week] if week in flowering_curves.columns else 0

    for _, cell_row in achillea_grid.iterrows():
        bin_key = cell_row['bin']
        pol_N_list = bin_pol_N_cache.get(bin_key, [])

        if not pol_N_list:
            results.append({
                'lat': cell_row['centroid_lat'],
                'lon': cell_row['centroid_lon'],
                'week': week,
                'pred_a2': np.nan,
                'pred_a3': np.nan
            })
            continue

        week_preds = []
        for pol, N_val in pol_N_list:
            a_w = activity_curves.loc[pol, week] if week in activity_curves.columns else 0
            delta = min(f_w, a_w)
            feat = np.concatenate([vf_plant, Vp_arr[vp_idx[pol]], [N_val, delta]])
            week_preds.append(clf_a3.predict_proba(feat.reshape(1, -1))[0, 1])

        results.append({
            'lat': cell_row['centroid_lat'],
            'lon': cell_row['centroid_lon'],
            'week': week,
            'pred_a2': bin_a2_cache[bin_key],
            'pred_a3': np.mean(week_preds)
        })

pred_df = pd.DataFrame(results)
pred_df.to_parquet(BASE + "pred_df_weeks1_52.parquet", index=False)
print(f"\nDone! Shape: {pred_df.shape}")
print(pred_df.dropna().describe())
