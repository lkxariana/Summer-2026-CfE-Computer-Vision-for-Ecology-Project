# Stage 5.2 — Code Documentation
## Probability Flowering Matrix (B)

---

## Cell 1 — Imports and check for species_data in memory

**Purpose:** Load libraries, define paths, load existence matrices, and check whether `species_data` is already in memory from a prior 5.1 session. If not, flags that Cell 3 will rebuild it.

**Key variables:**
- `BASE`: Stage 4 data directory
- `STAGE5_BASE`: Stage 5 output directory
- `PPE_DIR`: PPE opportunity surface parquet files
- `common_bins`: 3,160 bin keys shared by F and P

---

## Cell 2 — Load existence matrices and reconstruct common bins

**Purpose:** Load F and P, derive `common_bins` (3,160 bins), build `bin_to_idx` lookup, define `snap()` and `fmt()` helper functions for PPE coordinate alignment.

**Key variables:**
- `F`: plant binary existence matrix (6466 × 3162)
- `P`: pollinator binary existence matrix (4515 × 3895)
- `common_bins`: 3,160 bin keys in `'lat_lon'` format
- `bin_to_idx`: dict mapping bin key → integer index
- `n_bins`: 3160
- `n_weeks`: 52

**Helper functions:**
```python
def snap(x):
    # Snap PPE centroid coordinate to nearest 0.5° F/P grid anchor
    return round(round(x * 2) / 2, 1)

def fmt(x):
    return f"{x:.1f}"
```

---

## Cell 3 — Read PPE opportunity surface and collect species data

**Purpose:** Iterate over all 6,697 parquet files. Snap PPE coordinates to common bin grid, filter to common bins, and store `(bin_idx, week_idx, norm)` tuples per species in `species_data`.

Identical to Stage 5.1 Cell 3 — see that notebook for full documentation.

**Key variables:**
- `species_data`: `defaultdict(list)` — species → list of `(bin_idx, week_idx, norm)`

**Output:** 6,697 species accumulated.

**Runtime:** ~5–8 minutes.

---

## Cell 4 — Assemble PMf and fit PCA to 15D

**Purpose:** Convert `species_data` into a dense `(6697, 3160, 52)` array (PMf), flatten to `(6697, 164320)`, fit randomized PCA to 15 components → Vf_prob.

**Key difference from 5.1:** `n_components=15` instead of 4 — matching the dimensionality of the original binary Vf embedding, so the feature vector stays 31D.

**Key variables:**
- `PMf`: shape `(6697, 3160, 52)`, dtype float32 — spatiotemporal flowering probability matrix
- `PMf_flat`: shape `(6697, 164320)` — flattened for PCA
- `pca_pmf`: `PCA(n_components=15, svd_solver='randomized', random_state=42)`
- `Vf_prob`: shape `(6697, 15)` — spatiotemporal plant embeddings

**PCA results:**
- Variance explained per component: [0.201, 0.109, 0.052, 0.025, 0.019, 0.016, 0.011, 0.010, 0.009, 0.007, 0.006, 0.005, 0.005, 0.005, 0.005]
- Total variance explained: 48.6%

**Note:** Save Vf_prob and free PMf/PMf_flat from memory after this cell (4.40 GB).

---

## Cell 5 — Save Vf_prob and free memory

**Purpose:** Save Vf_prob as CSV with species index, save PCA object, free large arrays.

**Outputs saved to** `/scratch/ariana.l/Stage 5 PPE Representation Study/`:
- `stage5_Vf_prob.csv` — Vf_prob embeddings (6697 × 15)
- `stage5_pca_pmf.pkl` — fitted PCA object

---

## Cell 6 — Reconstruct training pairs and assemble 31D feature vectors

**Purpose:** Reconstruct positive/negative pairs from GloBI and build 31D feature matrix.

**Important fix vs. earlier attempts:** Positive pair filter must intersect both `Vf_prob_df.index` AND `F.index` — some species appear in PPE but not in F, causing KeyError when computing N via `F_common.loc[row.plant]`.

```python
pos_pairs = pos_pairs[
    pos_pairs['plant'].isin(Vf_prob_df.index) &
    pos_pairs['plant'].isin(F.index) &           # ← required
    pos_pairs['pollinator'].isin(Vp_df.index) &
    pos_pairs['pollinator'].isin(P.index)         # ← required
]

all_plants = list(set(Vf_prob_df.index) & set(F.index))   # ← required
all_pols   = list(set(Vp_df.index) & set(P.index))        # ← required
```

**Feature assembly:**
```python
def build_features(row):
    vf = Vf_prob_df.loc[row.plant].values     # 15D — spatiotemporal plant embedding
    vp = Vp_df.loc[row.pollinator].values     # 15D — binary pollinator embedding
    N  = float(np.dot(F_common.loc[row.plant].values,
                      P_common.loc[row.pollinator].values))  # 1D
    return np.concatenate([vf, vp, [N]])      # 31D
```

**Output:** X shape `(12296, 31)`, positive rate 0.250

---

## Cell 7 — Train/test split and logistic regression

**Split:** `train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)`
- Train: (9836, 31), Test: (2460, 31)

**Model:** `LogisticRegression(max_iter=1000, random_state=42)`

**Results:**
```
B  (31D, PMf):          ROC-AUC 0.938,  PR-AUC 0.856
A2 (31D, binary Vf):    ROC-AUC 0.931,  PR-AUC 0.842
A' (35D, V_δ appended): ROC-AUC 0.937,  PR-AUC 0.855
A3 (32D, scalar delta): ROC-AUC 0.950,  PR-AUC 0.868
```

---

## Cell 8 — Save model

**Output saved:** `stage5_B_logistic.pkl`
