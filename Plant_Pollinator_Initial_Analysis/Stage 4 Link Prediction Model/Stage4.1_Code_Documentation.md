# Stage 4 Code Documentation
# Stage_4_Link_Prediction_Model.ipynb

---

## Step 1 — Header sniff

Inspect GloBI column names before extraction.

```python
import pandas as pd

GLOBI_PATH = "/scratch/ariana.l/CfE2026CVforEcology/rawpollinatordata/interactions.csv.gz"

sample = pd.read_csv(GLOBI_PATH, nrows=2, low_memory=False)
print(f"Total columns: {len(sample.columns)}")
for col in sample.columns:
    print(f"  {col}")
```

**Output:** 92 columns. Key columns confirmed: `sourceTaxonName`, `targetTaxonName`,
`interactionTypeName`, `sourceTaxonOrderName`, `targetTaxonOrderName`,
`decimalLatitude`, `decimalLongitude`.

---

## Step 2 — Extract GloBI CONUS broad interactions

Filter the full GloBI dump to CONUS geography and broad pollination interaction types.
Reads in chunks to avoid loading 2.8GB into memory.

```python
import pandas as pd

GLOBI_PATH = "/scratch/ariana.l/CfE2026CVforEcology/rawpollinatordata/interactions.csv.gz"
OUT_PATH = "/scratch/ariana.l/Stage 4 Link Prediction Model/stage4_globi_conus_broad.csv"

BROAD_INTERACTION_TYPES = {
    'pollinates', 'pollinatedBy',
    'visitsFlowersOf', 'flowersVisitedBy',
    'visits', 'visitedBy',
    'hasFlowerVisitor'
}

CONUS_LON = (-125, -66)
CONUS_LAT  = (24, 50)

KEEP_COLS = [
    'sourceTaxonName', 'sourceTaxonOrderName', 'sourceTaxonFamilyName',
    'targetTaxonName', 'targetTaxonOrderName', 'targetTaxonFamilyName',
    'interactionTypeName',
    'decimalLatitude', 'decimalLongitude'
]

chunks = []
for chunk in pd.read_csv(GLOBI_PATH, usecols=KEEP_COLS, chunksize=100_000, low_memory=False):
    chunk = chunk[chunk['interactionTypeName'].isin(BROAD_INTERACTION_TYPES)]
    chunk = chunk.dropna(subset=['decimalLatitude', 'decimalLongitude'])
    chunk = chunk[
        chunk['decimalLongitude'].between(*CONUS_LON) &
        chunk['decimalLatitude'].between(*CONUS_LAT)
    ]
    chunks.append(chunk)

df = pd.concat(chunks, ignore_index=True)
df.to_csv(OUT_PATH, index=False)
print(f"Saved {len(df):,} rows to {OUT_PATH}")
print(df['interactionTypeName'].value_counts())
```

**Output:** 715,215 rows — visitsFlowersOf (577k), visits (72k), pollinates (65k).

---

## Step 3 — Check pollinator order coverage

Identify which taxonomic orders appear on the pollinator side of GloBI interactions
but are absent from our GBIF download.

```python
source_is_pollinator = {'visitsFlowersOf', 'pollinates', 'visits', 'hasFlowerVisitor'}
target_is_pollinator = {'flowersVisitedBy', 'pollinatedBy', 'visitedBy'}

pol_source = globi.loc[globi['interactionTypeName'].isin(source_is_pollinator),
                        ['sourceTaxonName', 'sourceTaxonOrderName']].rename(
                        columns={'sourceTaxonName': 'taxon', 'sourceTaxonOrderName': 'order'})

pol_target = globi.loc[globi['interactionTypeName'].isin(target_is_pollinator),
                        ['targetTaxonName', 'targetTaxonOrderName']].rename(
                        columns={'targetTaxonName': 'taxon', 'targetTaxonOrderName': 'order'})

pol_all = pd.concat([pol_source, pol_target]).drop_duplicates(subset='taxon')
print(pol_all['order'].value_counts(dropna=False).head(20))
```

**Finding:** Original GBIF download (Hymenoptera, Lepidoptera, Diptera, Coleoptera, Apodiformes)
covers the top 4 orders. Additional legitimate pollinator orders identified:
Hemiptera (813), Passeriformes (173), Thysanoptera (9), Neuroptera (18), Chiroptera (9).
Pending Dan's confirmation on whether to supplement.

---

## Step 4 — Build P existence matrix from GBIF

Vectorized chunked read — accumulate unique (species, bin) pairs per chunk,
concatenate and pivot once. No iterrows.

```python
import pandas as pd

BIN_SIZE = 0.5
CONUS_LON = (-125, -66)
CONUS_LAT  = (24, 50)
CHUNKSIZE  = 500_000

all_pairs = []

for path in [
    "/scratch/ariana.l/Plant Pollinator Initial Analysis/gbif_0007204_observations_v2.csv",
    "/scratch/ariana.l/Plant Pollinator Initial Analysis/gbif_0007192_observations_v2.csv"
]:
    for i, chunk in enumerate(pd.read_csv(path, chunksize=CHUNKSIZE, low_memory=False)):
        chunk = chunk.dropna(subset=['lat', 'lon', 'pollinator_species'])
        chunk = chunk[
            chunk['lon'].between(*CONUS_LON) &
            chunk['lat'].between(*CONUS_LAT)
        ]
        chunk['bin'] = (
            ((chunk['lat'] // BIN_SIZE) * BIN_SIZE).astype(str) + "_" +
            ((chunk['lon'] // BIN_SIZE) * BIN_SIZE).astype(str)
        )
        all_pairs.append(chunk[['pollinator_species', 'bin']].drop_duplicates())

all_pairs_df = pd.concat(all_pairs).drop_duplicates()
all_pairs_df['presence'] = 1

P = all_pairs_df.pivot_table(
    index='pollinator_species', columns='bin',
    values='presence', fill_value=0
)

P.to_csv("/scratch/ariana.l/Stage 4 Link Prediction Model/stage4_P_existence_gbif_combined.csv")
print(f"P matrix shape: {P.shape}")   # (4515, 3895)
print(f"Sparsity: {1 - P.values.mean():.3f}")  # 0.967
```

---

## Step 5 — Build F existence matrix from PhenoField

Stream all 107 shards from HuggingFace `dcher95/phenofield`, extract
(species, lat, lon), bin to 0.5° grid, pivot to existence matrix.

```python
import os
os.environ["HF_HOME"] = "/scratch/ariana.l/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "/scratch/ariana.l/hf_cache/datasets"

import time, pandas as pd
from datasets import Dataset
from huggingface_hub import hf_hub_download

REPO_ID = "dcher95/phenofield"
NUM_SHARDS = 107
BIN_SIZE = 0.5
CONUS_LON = (-125, -66)
CONUS_LAT = (24, 50)

# Step 5a: stream occurrences
all_dfs = []
for i in range(NUM_SHARDS):
    path = hf_hub_download(repo_id=REPO_ID,
                           filename=f"train/data-{i:05d}-of-00107.arrow",
                           repo_type="dataset")
    table = Dataset.from_file(path).data.table.select(["species", "latitude", "longitude"])
    df = table.to_pandas().rename(columns={"latitude": "lat", "longitude": "lon"})
    all_dfs.append(df)

plants = pd.concat(all_dfs, ignore_index=True)
plants.to_parquet("/scratch/ariana.l/Stage 4 Link Prediction Model/phenofield_plant_occurrences.parquet", index=False)
# Shape: (2,650,448, 3) | Unique species: 6,466

# Step 5b: build existence matrix
plants = plants.dropna(subset=['lat', 'lon'])
plants = plants[
    plants['lon'].between(*CONUS_LON) &
    plants['lat'].between(*CONUS_LAT)
]
plants['bin'] = (
    ((plants['lat'] // BIN_SIZE) * BIN_SIZE).astype(str) + "_" +
    ((plants['lon'] // BIN_SIZE) * BIN_SIZE).astype(str)
)
plants['presence'] = 1
F = plants[['species', 'bin', 'presence']].drop_duplicates().pivot_table(
    index='species', columns='bin', values='presence', fill_value=0
)
F.to_csv("/scratch/ariana.l/Stage 4 Link Prediction Model/stage4_F_existence_phenofield.csv")
print(f"F matrix shape: {F.shape}")   # (6466, 3162)
print(f"Sparsity: {1 - F.values.mean():.3f}")  # 0.980
```

---

## Step 6 — PCA on F and P

Fit PCA (15 components) separately on each existence matrix.

```python
from sklearn.decomposition import PCA
import pandas as pd

F = pd.read_csv("/scratch/ariana.l/Stage 4 Link Prediction Model/stage4_F_existence_phenofield.csv", index_col=0)
P = pd.read_csv("/scratch/ariana.l/Stage 4 Link Prediction Model/stage4_P_existence_gbif_combined.csv", index_col=0)

pca_f = PCA(n_components=15, random_state=42)
pca_p = PCA(n_components=15, random_state=42)

Vf = pca_f.fit_transform(F.values)
Vp = pca_p.fit_transform(P.values)

Vf_df = pd.DataFrame(Vf, index=F.index, columns=[f'f_pc{i+1}' for i in range(15)])
Vp_df = pd.DataFrame(Vp, index=P.index, columns=[f'p_pc{i+1}' for i in range(15)])

Vf_df.to_csv("/scratch/ariana.l/Stage 4 Link Prediction Model/stage4_Vf_phenofield.csv")
Vp_df.to_csv("/scratch/ariana.l/Stage 4 Link Prediction Model/stage4_Vp_gbif.csv")

print(f"F variance explained: {pca_f.explained_variance_ratio_.sum():.3f}")  # 0.399
print(f"P variance explained: {pca_p.explained_variance_ratio_.sum():.3f}")  # 0.757
```

**Note:** F variance explained (0.399) is lower than P (0.757) due to iNat observation
bias in PhenoField — plant records are concentrated in human-populated areas.
This is a known limitation of the lab's data.

---

## Step 7 — Compute N (shared bin count per pair)

N = number of 0.5° bins where both the plant and pollinator are present.
Computed as the dot product of their binary existence vectors over common bins.

```python
import pandas as pd
import numpy as np

globi = pd.read_csv("/scratch/ariana.l/Stage 4 Link Prediction Model/stage4_globi_conus_broad.csv")
Vf_df = pd.read_csv("/scratch/ariana.l/Stage 4 Link Prediction Model/stage4_Vf_phenofield.csv", index_col=0)
Vp_df = pd.read_csv("/scratch/ariana.l/Stage 4 Link Prediction Model/stage4_Vp_gbif.csv", index_col=0)
F = pd.read_csv("/scratch/ariana.l/Stage 4 Link Prediction Model/stage4_F_existence_phenofield.csv", index_col=0)
P = pd.read_csv("/scratch/ariana.l/Stage 4 Link Prediction Model/stage4_P_existence_gbif_combined.csv", index_col=0)

# Align F and P to common bins (3,160 bins)
common_bins = F.columns.intersection(P.columns)
F_common = F[common_bins]
P_common = P[common_bins]

F_arr = F_common.values
P_arr = P_common.values
f_idx = {s: i for i, s in enumerate(F_common.index)}
p_idx = {s: i for i, s in enumerate(P_common.index)}

# Extract GloBI positive pairs
source_is_pollinator = {'visitsFlowersOf', 'pollinates', 'visits', 'hasFlowerVisitor'}
target_is_pollinator = {'flowersVisitedBy', 'pollinatedBy', 'visitedBy'}

pairs_source = globi[globi['interactionTypeName'].isin(source_is_pollinator)][
    ['targetTaxonName', 'sourceTaxonName']].rename(
    columns={'targetTaxonName': 'plant', 'sourceTaxonName': 'pollinator'})
pairs_target = globi[globi['interactionTypeName'].isin(target_is_pollinator)][
    ['sourceTaxonName', 'targetTaxonName']].rename(
    columns={'sourceTaxonName': 'plant', 'targetTaxonName': 'pollinator'})

pairs = pd.concat([pairs_source, pairs_target]).drop_duplicates()
pairs = pairs[pairs['plant'].isin(Vf_df.index) & pairs['pollinator'].isin(Vp_df.index)]

# Compute N
pairs['N'] = pairs.apply(
    lambda r: int(F_arr[f_idx[r['plant']]] @ P_arr[p_idx[r['pollinator']]]),
    axis=1
)
print(f"Total GloBI pairs: {len(pairs):,}")  # 3,148
print(pairs['N'].describe())
# mean: 93.9 | median: 45 | max: 981
```

---

## Step 8 — Assemble 31D feature vectors

Concatenate [Vf (15D), Vp (15D), N (1D)] for each positive pair.

```python
import numpy as np

Vf_arr = Vf_df.values
Vp_arr = Vp_df.values
vf_idx = {s: i for i, s in enumerate(Vf_df.index)}
vp_idx = {s: i for i, s in enumerate(Vp_df.index)}

X = np.hstack([
    np.array([Vf_arr[vf_idx[r['plant']]] for _, r in pairs.iterrows()]),
    np.array([Vp_arr[vp_idx[r['pollinator']]] for _, r in pairs.iterrows()]),
    pairs['N'].values.reshape(-1, 1)
])
y = np.ones(len(pairs), dtype=int)

print(f"X shape: {X.shape}")  # (3148, 31)
```

---

## Step 9 — Train A2 baseline logistic regression

Sample negatives at 3:1 ratio (~25% positive), train logistic regression, evaluate.

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
import pickle

np.random.seed(42)

all_plants = list(Vf_df.index)
all_pollinators = list(Vp_df.index)
positive_set = set(zip(pairs['plant'], pairs['pollinator']))

# Sample negatives
n_neg = len(pairs) * 3
neg_pairs = []
while len(neg_pairs) < n_neg:
    p = np.random.choice(all_plants)
    pol = np.random.choice(all_pollinators)
    if (p, pol) not in positive_set:
        neg_pairs.append((p, pol))

neg_df = pd.DataFrame(neg_pairs, columns=['plant', 'pollinator'])
neg_df['N'] = neg_df.apply(
    lambda r: int(F_arr[f_idx[r['plant']]] @ P_arr[p_idx[r['pollinator']]]),
    axis=1
)

X_neg = np.hstack([
    np.array([Vf_arr[vf_idx[r['plant']]] for _, r in neg_df.iterrows()]),
    np.array([Vp_arr[vp_idx[r['pollinator']]] for _, r in neg_df.iterrows()]),
    neg_df['N'].values.reshape(-1, 1)
])
y_neg = np.zeros(len(neg_df), dtype=int)

X_all = np.vstack([X, X_neg])
y_all = np.concatenate([y, y_neg])
# Total: 12,592 pairs (3,148 positive, 9,444 negative)

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all)

clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train, y_train)

y_pred_proba = clf.predict_proba(X_test)[:, 1]
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.3f}")   # 0.931
print(f"PR-AUC:  {average_precision_score(y_test, y_pred_proba):.3f}")  # 0.842

with open("/scratch/ariana.l/Stage 4 Link Prediction Model/stage4_A2_logistic.pkl", "wb") as f:
    pickle.dump(clf, f)
```

---

## Next: Step 10 — A3 with PPE (pending)

Once `opportunity_surface/` rclone transfer completes:
- Load PPE parquet files from `/scratch/ariana.l/ppe-outputs/opportunity_surface/`
- Compute Δ (temporal overlap) per (plant, pollinator) pair
- Append Δ to feature vector → 32D `[Vf, Vp, N, Δ]`
- Retrain logistic regression → A3 model
- Compare ROC-AUC / PR-AUC: A2 vs A3
