"""
01_load_data.py
Stage 4 Visualization — Load all data and rebuild curves

Run this first before any other script. All variables are used by
subsequent scripts in the same notebook session.
"""

import numpy as np
import pandas as pd
import pickle
import glob
from scipy.stats import vonmises

BASE = "/scratch/ariana.l/Stage 4 Link Prediction Model/"

# --- Existence matrices ---
print("Loading matrices...")
F = pd.read_csv(BASE + "stage4_F_existence_phenofield.csv", index_col=0)
P = pd.read_csv(BASE + "stage4_P_existence_gbif_combined.csv", index_col=0)

F_arr = F.values
P_arr = P.values
f_idx = {s: i for i, s in enumerate(F.index)}
p_idx = {s: i for i, s in enumerate(P.index)}

# --- Align F and P to common bins for N computation ---
common_bins = F.columns.intersection(P.columns)
F_common = F[common_bins]
P_common = P[common_bins]
F_common_arr = F_common.values
P_common_arr = P_common.values
f_common_idx = {s: i for i, s in enumerate(F_common.index)}
p_common_idx = {s: i for i, s in enumerate(P_common.index)}
print(f"  Common bins: {len(common_bins):,}")

# --- PCA embeddings ---
print("Loading embeddings...")
Vf_df = pd.read_csv(BASE + "stage4_Vf_phenofield.csv", index_col=0)
Vp_df = pd.read_csv(BASE + "stage4_Vp_gbif.csv", index_col=0)
Vf_arr = Vf_df.values
Vp_arr = Vp_df.values
vf_idx = {s: i for i, s in enumerate(Vf_df.index)}
vp_idx = {s: i for i, s in enumerate(Vp_df.index)}

# --- Models ---
print("Loading models...")
with open(BASE + "stage4_A3_logistic.pkl", "rb") as f:
    clf_a3 = pickle.load(f)
with open(BASE + "stage4_A2_logistic.pkl", "rb") as f:
    clf_a2 = pickle.load(f)

# --- Flowering curves from PPE opportunity surface ---
# Reconstructed by averaging norm across all cells per species per week
print("Building flowering curves (averaging norm across cells)...")
files = sorted(glob.glob("/scratch/ariana.l/ppe-outputs/opportunity_surface/part_*.parquet"))
all_curves = []
for f in files:
    df = pd.read_parquet(f, columns=['species', 'week', 'norm'])
    all_curves.append(df.groupby(['species', 'week'])['norm'].mean().reset_index())

flowering_curves = (
    pd.concat(all_curves)
    .groupby(['species', 'week'])['norm'].mean()
    .unstack(fill_value=0)
)
print(f"  flowering_curves: {flowering_curves.shape}")

# --- Pollinator activity curves from GBIF ---
# Weekly observation counts normalized to sum=1 per species
print("Building pollinator activity curves...")

def doy_to_week(doy):
    return ((doy - 1) // 7).clip(0, 51)

GBIF_PATH = "/scratch/ariana.l/Plant Pollinator Initial Analysis/gbif_0007192_observations_v2.csv"
pol_doy = pd.read_csv(GBIF_PATH, usecols=['pollinator_species', 'doy'], low_memory=False)
pol_doy = pol_doy.dropna(subset=['doy', 'pollinator_species'])
pol_doy['week'] = doy_to_week(pol_doy['doy'].astype(int))

week_counts = pol_doy.groupby(['pollinator_species', 'week']).size().unstack(fill_value=0)
activity_curves = week_counts.div(week_counts.sum(axis=1).replace(0, 1), axis=0)
print(f"  activity_curves: {activity_curves.shape}")

# --- Achillea millefolium PPE opportunity surface ---
print("Loading Achillea PPE surface...")
achillea_data = []
for f in files:
    df = pd.read_parquet(f)
    subset = df[df['species'] == 'Achillea millefolium']
    if len(subset) > 0:
        achillea_data.append(subset)
achillea = pd.concat(achillea_data, ignore_index=True)
print(f"  achillea: {achillea.shape}")

# --- Ground truth from GloBI ---
print("Loading ground truth...")
gt = pd.read_csv(BASE + "stage4_globi_conus_broad.csv")
gt = gt[
    (gt['sourceTaxonName'] == 'Achillea millefolium') |
    (gt['targetTaxonName'] == 'Achillea millefolium')
].dropna(subset=['decimalLatitude', 'decimalLongitude'])

print(f"\nAll done!")
print(f"  F: {F_arr.shape}, P: {P_arr.shape}")
print(f"  Vf: {Vf_arr.shape}, Vp: {Vp_arr.shape}")
print(f"  flowering_curves: {flowering_curves.shape}")
print(f"  activity_curves: {activity_curves.shape}")
print(f"  achillea PPE surface: {achillea.shape}")
print(f"  ground truth obs: {len(gt):,}")
