# Stage 5.4 — Code Documentation
## Spatiotemporal Embedding A* (V_δ 15D)

---

## Cell 1 — Imports and paths

**Purpose:** Load libraries and define paths. Identical to 5.1 and 5.2.

**Key paths:**
- `BASE`: Stage 4 data directory
- `STAGE5_BASE`: Stage 5 output directory
- `PPE_DIR`: PPE opportunity surface parquet files

---

## Cell 2 — Load existence matrices and reconstruct common bins

**Purpose:** Load F and P, derive `common_bins` (3,160 bins), build `bin_to_idx`, define coordinate helper functions.

Identical to 5.1, 5.2, 5.3. See those notebooks for full documentation.

**Key variables:**
- `common_bins`: 3,160 bin keys in `'lat_lon'` format
- `n_bins`: 3160, `n_weeks`: 52

---

## Cell 3 — Read PPE opportunity surface and collect species data

**Purpose:** Iterate over 6,697 parquet files, snap PPE coordinates to common bin grid, collect `(bin_idx, week_idx, norm)` tuples per species into `species_data`.

Identical to 5.1 and 5.2. Runtime ~5–8 minutes.

**Output:** `species_data` with 6,697 plant species.

---

## Cell 4 — Assemble delta matrix and fit PCA to 15D

**Purpose:** Convert `species_data` into a `(6697, 3160, 52)` array, flatten to `(6697, 164320)`, fit randomized PCA to **15 components** (expanded from 4 in A′).

**Key difference from 5.1:**
```python
# A' (5.1):
pca_delta = PCA(n_components=4, svd_solver='randomized', random_state=42)

# A* (5.4):
pca_delta_15 = PCA(n_components=15, svd_solver='randomized', random_state=42)
```

**Key variables:**
- `delta_3d`: shape `(6697, 3160, 52)`, dtype float32, 4.40 GB
- `delta_flat`: shape `(6697, 164320)`
- `pca_delta_15`: fitted PCA object
- `V_delta_15`: shape `(6697, 15)`

**PCA results:**
- Total variance explained: 48.6% (vs 38.7% at 4D in A′)
- The additional 11 components (PC5–PC15) contribute ~10% more variance, capturing finer-grained spatiotemporal patterns in plant flowering distributions

---

## Cell 5 — Save V_delta_15 and free memory

**Purpose:** Save V_δ_15 as CSV with species index, save PCA object, free delta_3d and delta_flat.

**Outputs saved to** `/scratch/ariana.l/Stage 5 PPE Representation Study/`:
- `stage5_Vdelta_15d.csv` — V_δ_15 embeddings (6697 × 15)
- `stage5_pca_delta_15d.pkl` — fitted PCA object

---

## Cell 6 — Reconstruct training pairs and assemble 46D feature vectors

**Purpose:** Reconstruct positive/negative pairs and build 46D feature matrix.

**Key difference from A′:** Feature vector is 46D instead of 35D — V_δ contributes 15D instead of 4D.

**Pair filter:** Must intersect `Vf_df.index`, `F.index`, AND `Vd15_df.index` for plants; `Vp_df.index` AND `P.index` for pollinators.

```python
def build_features(row):
    vf  = Vf_df.loc[row.plant].values          # 15D — binary plant embedding
    vp  = Vp_df.loc[row.pollinator].values     # 15D — binary pollinator embedding
    vd  = Vd15_df.loc[row.plant].values        # 15D — spatiotemporal plant embedding
    N   = float(np.dot(F_common.loc[row.plant].values,
                       P_common.loc[row.pollinator].values))  # 1D
    return np.concatenate([vf, vp, [N], vd])   # 46D
```

**Output:** X shape `(12296, 46)`, positive rate 0.250

---

## Cell 7 — Train/test split and logistic regression

**Split:** `train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)`

**Model:** `LogisticRegression(max_iter=1000, random_state=42)`

**Results:**
```
A* (46D, V_δ 15D):      ROC-AUC 0.948,  PR-AUC 0.878  ← best PR-AUC in Stage 5
A3 (32D, scalar delta): ROC-AUC 0.950,  PR-AUC 0.868
B  (31D, PMf):          ROC-AUC 0.938,  PR-AUC 0.856
A' (35D, V_δ 4D):       ROC-AUC 0.937,  PR-AUC 0.855
A2 (31D, binary):       ROC-AUC 0.931,  PR-AUC 0.842
B' (31D, PMp):          ROC-AUC 0.924,  PR-AUC 0.828
```

---

## Cell 8 — Save model

**Output saved:** `stage5_Astar_logistic.pkl`

---

## Cell 9 — Full Stage 5 results table

**Purpose:** Print formatted results table for all Stage 5 models, ordered as A2, A3, A′, A*, B, B′. Rendered as a Markdown table via `IPython.display.Markdown` for clean notebook presentation with bold values for best ROC-AUC (A3) and best PR-AUC (A*).

```python
from IPython.display import display, Markdown

table = """
| Model | Features | ROC-AUC | PR-AUC |
|-------|----------|---------|--------|
| A2  | binary Vf + Vp + N (31D)         | 0.931 | 0.842 |
| A3  | Vf + Vp + N + Δ scalar (32D)     | **0.950** | 0.868 |
| A'  | Vf + Vp + N + V_δ 4D (35D)      | 0.937 | 0.855 |
| A*  | Vf + Vp + N + V_δ 15D (46D)     | 0.948 | **0.878** |
| B   | PMf + Vp + N (31D)               | 0.938 | 0.856 |
| B'  | binary Vf + PMp + N (31D)        | 0.924 | 0.828 |
"""
display(Markdown(table))
```
