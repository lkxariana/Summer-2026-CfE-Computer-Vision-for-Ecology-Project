"""
Stage 6 — Seed Testing
======================
Validates stability of ANTHEIA link prediction results across 5 random seeds.

Models tested:
  A2     — [Vf (15D), Vp (15D), N (1D)] = 31D  (spatial baseline)
  A3     — [Vf (15D), Vp (15D), N (1D), delta scalar (1D)] = 32D
  A'     — [Vf (15D), Vp (15D), N (1D), V_delta 4D (35D)]
  A*     — [Vf (15D), Vp (15D), N (1D), V_delta 15D (46D)]
  B      — [Vf_prob (15D), Vp (15D), N (1D)] = 31D

Key differences from original Stage 4/5 training:
  - Activity curves (a_curves) now include both GBIF insects AND eBird data
  - Flowering curves (f_curves) built from full 6697-species opportunity surface
    instead of the 50-species flowering_curves_used.parquet
  - Seeds varied: [42, 0, 1, 2, 3]; seed 42 is the original run (sanity check)

Outputs (saved to STAGE6_BASE):
  stage6_all_seed_results.csv   — per-seed ROC-AUC and PR-AUC for all models
  stage6_summary.csv            — mean ± std across 5 seeds per model
  results_table_v6.png          — styled poster table (mean values)
"""

import numpy as np
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

# ── Paths ────────────────────────────────────────────────────────────────────
BASE        = "/scratch/ariana.l/Stage 4 Link Prediction Model/"
STAGE5_BASE = "/scratch/ariana.l/Stage 5 PPE Representation Study/"
STAGE6_BASE = "/scratch/ariana.l/Stage 6 Seed Testing/"
PPE_DIR     = "/scratch/ariana.l/ppe-outputs/opportunity_surface/"
GBIF_INSECT = "/scratch/ariana.l/Plant Pollinator Initial Analysis/gbif_0007192_observations_v2.csv"
GBIF_BIRD   = "/scratch/ariana.l/Plant Pollinator Initial Analysis/gbif_0007204_observations_v2.csv"

os.makedirs(STAGE6_BASE, exist_ok=True)
SEEDS = [42, 0, 1, 2, 3]

# ── Step 1: Build f_curves from full opportunity surface (6697 species) ──────
print("Building f_curves from opportunity surface...")
files = sorted(glob.glob(PPE_DIR + "part_*.parquet"))
print(f"  Total files: {len(files)}")
f_curves_dict = {}
for i, fpath in enumerate(files):
    df = pd.read_parquet(fpath, columns=['species', 'week', 'norm'])
    species = df['species'].iloc[0]
    weekly_mean = df.groupby('week')['norm'].mean().reindex(range(52), fill_value=0)
    total = weekly_mean.sum()
    if total > 0:
        weekly_mean = weekly_mean / total
    f_curves_dict[species] = weekly_mean.values
    if (i + 1) % 500 == 0:
        print(f"  {i+1}/{len(files)} files processed...")
f_curves = pd.DataFrame.from_dict(f_curves_dict, orient='index', columns=list(range(52)))
print(f"  f_curves: {f_curves.shape}")

# ── Step 2: Build a_curves from insects + eBird (chunked) ───────────────────
print("Building a_curves from insects + eBird...")
counts = defaultdict(lambda: defaultdict(int))

def process_file(path, label):
    total = 0
    for chunk in pd.read_csv(path, usecols=['pollinator_species', 'doy'],
                              chunksize=500_000, low_memory=False):
        chunk = chunk.dropna(subset=['doy', 'pollinator_species'])
        chunk['week'] = ((chunk['doy'].astype(int) - 1) // 7).clip(0, 51)
        for row in chunk.itertuples(index=False):
            counts[row.pollinator_species][row.week] += 1
        total += len(chunk)
    print(f"  {label}: {total:,} rows")

process_file(GBIF_INSECT, "insects")
process_file(GBIF_BIRD,   "eBird")

all_species = sorted(counts.keys())
weeks = list(range(52))
a_curves = pd.DataFrame(
    [{w: counts[sp].get(w, 0) for w in weeks} for sp in all_species],
    index=all_species, columns=weeks
)
a_curves = a_curves.div(a_curves.sum(axis=1), axis=0)
del counts
print(f"  a_curves: {a_curves.shape}")

# ── Step 3: Load existence matrices and embeddings ───────────────────────────
print("Loading existence matrices...")
F = pd.read_csv(BASE + "stage4_F_existence_phenofield.csv", index_col=0)
P = pd.read_csv(BASE + "stage4_P_existence_gbif_combined.csv", index_col=0)
common_bins = sorted(set(F.columns) & set(P.columns))
F_common = F[common_bins]
P_common = P[common_bins]
print(f"  Common bins: {len(common_bins)}")

print("Loading embeddings...")
Vf_df   = pd.read_csv(BASE + "stage4_Vf_phenofield.csv", index_col=0)
Vp_df   = pd.read_csv(BASE + "stage4_Vp_gbif.csv", index_col=0)
Vd4_df  = pd.read_csv(STAGE5_BASE + "stage5_Vdelta_ppe.csv", index_col=0)
Vd15_df = pd.read_csv(STAGE5_BASE + "stage5_Vdelta_15d.csv", index_col=0)
Vfp_df  = pd.read_csv(STAGE5_BASE + "stage5_Vf_prob.csv", index_col=0)

print("Loading GloBI...")
globi = pd.read_csv(BASE + "stage4_globi_conus_broad.csv")
print(f"  GloBI records: {len(globi):,}")

# ── Step 4: Precompute array views and index maps ────────────────────────────
F_common_arr = F_common.values
P_common_arr = P_common.values
fc_idx  = {s: i for i, s in enumerate(F_common.index)}
pc_idx  = {s: i for i, s in enumerate(P_common.index)}
vf_idx  = {s: i for i, s in enumerate(Vf_df.index)}
vp_idx  = {s: i for i, s in enumerate(Vp_df.index)}
vd4_idx = {s: i for i, s in enumerate(Vd4_df.index)}
vd_idx  = {s: i for i, s in enumerate(Vd15_df.index)}
vfp_idx = {s: i for i, s in enumerate(Vfp_df.index)}

Vf_arr   = Vf_df.values
Vp_arr   = Vp_df.values
Vd4_arr  = Vd4_df.values
Vd15_arr = Vd15_df.values
Vfp_arr  = Vfp_df.values

# ── Step 5: Species sets ─────────────────────────────────────────────────────
plants_A     = set(Vf_df.index)  & set(F.index)
pols_A       = set(Vp_df.index)  & set(P.index)
plants_A3    = plants_A & set(f_curves.index)
pols_A3      = pols_A   & set(a_curves.index)
plants_Ap    = plants_A & set(Vd4_df.index)
plants_Astar = plants_A & set(Vd15_df.index)
plants_B     = set(Vfp_df.index) & set(F.index)

print(f"Plants  A2/A3/A*/A': {len(plants_A)} / {len(plants_A3)} / {len(plants_Astar)} / {len(plants_Ap)}")
print(f"Plants  B: {len(plants_B)}")
print(f"Pollinators: {len(pols_A)} (A3: {len(pols_A3)})")

# ── Step 6: Helper functions ─────────────────────────────────────────────────
def compute_delta(plant, pollinator):
    f = f_curves.loc[plant].values
    a = a_curves.loc[pollinator].values
    return float(np.minimum(f, a).sum())

def get_pos_pairs(plants_set, pols_set):
    pos = globi[['sourceTaxonName', 'targetTaxonName']].drop_duplicates()
    pos.columns = ['pollinator', 'plant']
    pos = pos[pos['plant'].isin(plants_set) & pos['pollinator'].isin(pols_set)].copy()
    pos['label'] = 1
    return pos

def get_neg_pairs(pos_pairs, plants_set, pols_set, seed):
    pos_set = set(zip(pos_pairs['pollinator'], pos_pairs['plant']))
    rng = np.random.default_rng(seed)
    plants_arr = np.array(list(plants_set))
    pols_arr   = np.array(list(pols_set))
    neg = []
    while len(neg) < len(pos_pairs) * 3:
        pol   = rng.choice(pols_arr)
        plant = rng.choice(plants_arr)
        if (pol, plant) not in pos_set:
            neg.append((pol, plant, 0))
    return pd.DataFrame(neg, columns=['pollinator', 'plant', 'label'])

# ── Step 7: Feature builders ─────────────────────────────────────────────────
def features_A2(plant, pollinator):
    vf = Vf_arr[vf_idx[plant]]
    vp = Vp_arr[vp_idx[pollinator]]
    N  = float(F_common_arr[fc_idx[plant]] @ P_common_arr[pc_idx[pollinator]])
    return np.concatenate([vf, vp, [N]])

def features_A3(plant, pollinator):
    vf    = Vf_arr[vf_idx[plant]]
    vp    = Vp_arr[vp_idx[pollinator]]
    N     = float(F_common_arr[fc_idx[plant]] @ P_common_arr[pc_idx[pollinator]])
    delta = compute_delta(plant, pollinator)
    return np.concatenate([vf, vp, [N, delta]])

def features_Aprime(plant, pollinator):
    vf  = Vf_arr[vf_idx[plant]]
    vp  = Vp_arr[vp_idx[pollinator]]
    N   = float(F_common_arr[fc_idx[plant]] @ P_common_arr[pc_idx[pollinator]])
    vd4 = Vd4_arr[vd4_idx[plant]]
    return np.concatenate([vf, vp, [N], vd4])

def features_Astar(plant, pollinator):
    vf   = Vf_arr[vf_idx[plant]]
    vp   = Vp_arr[vp_idx[pollinator]]
    N    = float(F_common_arr[fc_idx[plant]] @ P_common_arr[pc_idx[pollinator]])
    vd15 = Vd15_arr[vd_idx[plant]]
    return np.concatenate([vf, vp, [N], vd15])

def features_B(plant, pollinator):
    vf = Vfp_arr[vfp_idx[plant]]
    vp = Vp_arr[vp_idx[pollinator]]
    N  = float(F_common_arr[fc_idx[plant]] @ P_common_arr[pc_idx[pollinator]])
    return np.concatenate([vf, vp, [N]])

# ── Step 8: Main seed loop ───────────────────────────────────────────────────
MODELS = {
    'A2':     (features_A2,     plants_A,     pols_A),
    'A3':     (features_A3,     plants_A3,    pols_A3),
    'Aprime': (features_Aprime, plants_Ap,    pols_A),
    'Astar':  (features_Astar,  plants_Astar, pols_A),
    'B':      (features_B,      plants_B,     pols_A),
}

results = []

for model_name, (feat_fn, plants_set, pols_set) in MODELS.items():
    print(f"\n{'='*50}\nModel: {model_name}\n{'='*50}")
    pos_pairs = get_pos_pairs(plants_set, pols_set)
    print(f"Positive pairs: {len(pos_pairs)}")

    for seed in SEEDS:
        print(f"  Seed {seed}...", end=' ', flush=True)
        neg_pairs = get_neg_pairs(pos_pairs, plants_set, pols_set, seed)
        pairs = pd.concat([pos_pairs, neg_pairs], ignore_index=True)

        X = np.vstack([feat_fn(r.plant, r.pollinator) for r in pairs.itertuples()])
        y = pairs['label'].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=seed)

        clf = LogisticRegression(max_iter=1000, random_state=seed)
        clf.fit(X_train, y_train)
        y_prob = clf.predict_proba(X_test)[:, 1]

        roc = roc_auc_score(y_test, y_prob)
        pr  = average_precision_score(y_test, y_prob)
        print(f"ROC-AUC {roc:.4f}  PR-AUC {pr:.4f}")

        results.append({'model': model_name, 'seed': seed,
                        'roc_auc': roc, 'pr_auc': pr,
                        'n_pos': len(pos_pairs), 'n_total': len(pairs)})

# ── Step 9: Save results ─────────────────────────────────────────────────────
results_df = pd.DataFrame(results)
results_df.to_csv(STAGE6_BASE + "stage6_all_seed_results.csv", index=False)

summary = results_df.groupby('model').agg(
    roc_mean=('roc_auc', 'mean'), roc_std=('roc_auc', 'std'),
    pr_mean=('pr_auc', 'mean'),  pr_std=('pr_auc', 'std'),
).round(4).reindex(['A2', 'A3', 'Aprime', 'Astar', 'B'])
summary.to_csv(STAGE6_BASE + "stage6_summary.csv")

print("\n" + "="*60)
print("Seed Testing Summary (mean ± std, 5 seeds)")
print("="*60)
print(f"{'Model':<10} {'ROC-AUC':>20} {'PR-AUC':>20}")
print("-"*60)
for model, row in summary.iterrows():
    label = {"Aprime": "A'", "Astar": "A*"}.get(model, model)
    print(f"{label:<10} {row.roc_mean:.4f} ± {row.roc_std:.4f}   "
          f"{row.pr_mean:.4f} ± {row.pr_std:.4f}")
print("="*60)
print(f"\nSaved: stage6_all_seed_results.csv, stage6_summary.csv")
