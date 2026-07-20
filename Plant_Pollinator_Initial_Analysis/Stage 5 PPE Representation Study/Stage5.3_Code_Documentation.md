# Stage 5.3 — Code Documentation
## Probability Pollinator Matrix (B′)

---

## Cell 1 — Imports, paths, and load existence matrices

**Purpose:** Load libraries, define paths, load F and P existence matrices, derive common bins, define coordinate helper functions.

**Key variables:**
- `common_bins`: 3,160 bin keys in `'lat_lon'` format
- `bin_to_idx`: dict mapping bin key → integer index
- `n_bins`: 3160, `n_weeks`: 52

**Helper functions:**
```python
def snap(x):
    # Snap coordinate to nearest 0.5° grid anchor (.0 or .5)
    return round(round(x * 2) / 2, 1)

def fmt(x):
    return f"{x:.1f}"
```

---

## Cell 2 — Load and combine GBIF insect + bird observations

**Purpose:** Load both GBIF files (insects + birds), combine, convert doy to week index, snap coordinates to common bin grid, filter to common bins.

**Column note:** GBIF files use `lat` / `lon` — NOT `decimalLatitude` / `decimalLongitude`. Using the wrong column names raises `ValueError: Usecols do not match columns`.

**Key operations:**
```python
# doy → week index
gbif['week'] = ((gbif['doy'] - 1) // 7).clip(0, 51).astype(int)

# Snap to common bin grid
gbif['bin_key'] = gbif['lat'].map(snap).map(fmt) + '_' + gbif['lon'].map(snap).map(fmt)
gbif = gbif[gbif['bin_key'].isin(bin_to_idx)]
```

**Output sizes:**
- Insects: 5,053,625 rows
- Birds: 542,898,437 rows
- Combined: 547,952,062 rows, 4,515 species
- After filtering to common bins: 504,347,182 rows

---

## Cell 3 — Build PMp (n_species × n_bins × n_weeks)

**Purpose:** Aggregate GBIF observations into weekly counts per (species, bin), apply MIN_OBS filter, normalize, and fill the PMp array.

**Key operations:**
```python
# Aggregate
counts = gbif.groupby(['pollinator_species', 'bin_idx', 'week']).size().reset_index(name='count')

# Total per (species, bin) for filter + normalization
totals = counts.groupby(['pollinator_species', 'bin_idx'])['count'].sum().reset_index(name='total')
counts = counts.merge(totals, on=['pollinator_species', 'bin_idx'])

# MIN_OBS filter
MIN_OBS = 5
counts = counts[counts['total'] >= MIN_OBS]

# Normalize
counts['norm'] = counts['count'] / counts['total']
```

**Key variables:**
- `PMp`: shape `(4425, 3160, 52)`, dtype float32, 2.91 GB
- `pol_species`: sorted list of 4,425 pollinator species (note: 4,515 unique in raw GBIF; 90 dropped due to NaN or no bins surviving MIN_OBS filter)
- `MIN_OBS = 5`: bins with fewer than 5 total observations zero-padded

**Sparsity:** Only 1.1% non-zero (7,822,389 / 727,116,000 entries) — significantly sparser than PMf (~much higher density from PPE model output). This is the core limitation of PMp: GBIF is opportunistic and spatially biased.

---

## Cell 4 — Fit PCA to 15D → Vp_prob

**Purpose:** Flatten PMp to `(4425, 164320)`, fit randomized PCA to 15 components.

**Key variables:**
- `PMp_flat`: shape `(4425, 164320)`
- `pca_pmp`: `PCA(n_components=15, svd_solver='randomized', random_state=42)`
- `Vp_prob`: shape `(4425, 15)`

**PCA results:**
- Variance explained per component: [0.197, 0.059, 0.035, 0.024, 0.016, 0.013, 0.011, 0.010, 0.009, 0.008, 0.008, 0.007, 0.007, 0.007, 0.006]
- Total variance explained: 41.5%

**Comparison with PMf PCA (5.2):** PMf achieved 48.6% variance explained at 15 components; PMp achieves 41.5%. The lower explained variance reflects higher sparsity and noisier structure in the pollinator data. PC1 dominates more strongly (19.7% vs 20.1%) but subsequent components fall off faster, suggesting fewer meaningful axes of variation in PMp.

---

## Cell 5 — Save Vp_prob and free memory

**Purpose:** Save Vp_prob as CSV with species index, save PCA object, free PMp and PMp_flat.

**Outputs saved to** `/scratch/ariana.l/Stage 5 PPE Representation Study/`:
- `stage5_Vp_prob.csv` — Vp_prob embeddings (4425 × 15)
- `stage5_pca_pmp.pkl` — fitted PCA object

---

## Cell 6 — Reconstruct training pairs and assemble 31D feature vectors

**Purpose:** Reconstruct positive/negative pairs and build 31D feature matrix using binary Vf + Vp_prob.

**Positive pairs:** 3,148 (slightly higher than B/A′'s 3,074 — because Vp_prob_df covers more pollinator species from the combined GBIF dataset than the original Vp_df from Stage 4)

**Feature assembly:**
```python
def build_features(row):
    vf = Vf_df.loc[row.plant].values              # 15D — binary plant embedding (Stage 4)
    vp = Vp_prob_df.loc[row.pollinator].values    # 15D — spatiotemporal pollinator embedding
    N  = float(np.dot(F_common.loc[row.plant].values,
                      P_common.loc[row.pollinator].values))  # 1D
    return np.concatenate([vf, vp, [N]])           # 31D
```

**Output:** X shape `(12592, 31)`, positive rate 0.250

---

## Cell 7 — Train/test split and logistic regression

**Split:** `train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)`
- Train: (10073, 31), Test: (2519, 31)

**Model:** `LogisticRegression(max_iter=1000, random_state=42)`

**Results:**
```
B′ (31D, PMp):          ROC-AUC 0.924,  PR-AUC 0.828  ← worst in Stage 5
A2 (31D, binary):       ROC-AUC 0.931,  PR-AUC 0.842
A' (35D, V_δ appended): ROC-AUC 0.937,  PR-AUC 0.855
B  (31D, PMf):          ROC-AUC 0.938,  PR-AUC 0.856
A3 (32D, scalar delta): ROC-AUC 0.950,  PR-AUC 0.868
```

---

## Cell 8 — Save model

**Output saved:** `stage5_Bprime_logistic.pkl`
