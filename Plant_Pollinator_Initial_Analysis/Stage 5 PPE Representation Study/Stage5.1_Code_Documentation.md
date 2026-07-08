# Stage 5.1 — Code Documentation
## Spatiotemporal Embedding (A′)

---

## Cell 1 — Load existence matrices and reconstruct common bins

**Purpose:** Load the Stage 4 plant (F) and pollinator (P) existence matrices and derive the 3,160 common spatial bins shared by both.

**Key variables:**
- `F`: plant existence matrix (6466 species × 3162 bins)
- `P`: pollinator existence matrix (4515 species × 3895 bins)
- `common_bins`: sorted list of 3,160 bin keys in `'lat_lon'` format (e.g. `'24.5_-81.0'`)

**Note:** F has 3162 bins and P has 3895 bins; the intersection yields 3,160 common bins used throughout Stage 5.

---

## Cell 2 — Inspect PPE opportunity surface column names and bin key format

**Purpose:** Diagnostic cell. Read one parquet file to confirm column names and check whether PPE bin coordinate format matches `common_bins`.

**Finding:** PPE uses `centroid_lat` / `centroid_lon` (not `lat` / `lon`). PPE centroids are anchored at `.25`/`.75` while common_bins use `.0`/`.5` — a 0.25° grid offset requiring snapping (see Cell 3).

---

## Cell 3 — Read all PPE opportunity surface files and collect species data

**Purpose:** Iterate over all 6,697 parquet files in the PPE opportunity surface directory. For each file, snap PPE centroid coordinates to the nearest common bin, filter to common bins only, and collect `(bin_idx, week_idx, norm)` tuples per species into `species_data`.

**Key functions:**
```python
def snap(x):
    # Snap float coordinate to nearest 0.5° grid anchor (.0 or .5)
    return round(round(x * 2) / 2, 1)

def fmt(x):
    return f"{x:.1f}"
```

**Key variables:**
- `species_data`: `defaultdict(list)` mapping species name → list of `(bin_idx, week_idx, norm)` tuples
- `bin_to_idx`: dict mapping bin key string → integer index into `common_bins`

**Output:** `species_data` with 6,697 plant species (1 file = 1 species in PPE)

**Runtime:** ~5–8 minutes for 6,697 files.

---

## Cell 4 — Assemble 3D delta matrix and fit PCA

**Purpose:** Convert `species_data` into a dense `(6697, 3160, 52)` numpy array (float32), flatten to `(6697, 164320)`, and fit randomized PCA to 4 components to produce V_δ.

**Memory:** 4.40 GB for `delta_3d` (float32). Freed immediately after PCA in Cell 5.

**PCA choice:** `svd_solver='randomized'` avoids full SVD on a 164K-dimensional matrix, computing only the top-4 components directly.

**Key variables:**
- `delta_3d`: shape `(6697, 3160, 52)`, dtype float32 — per-species spatiotemporal flowering matrix
- `delta_flat`: shape `(6697, 164320)` — flattened for PCA input
- `pca_delta`: fitted `PCA(n_components=4, svd_solver='randomized', random_state=42)`
- `V_delta`: shape `(6697, 4)` — spatiotemporal embeddings

**Results:**
- Variance explained: [0.201, 0.109, 0.052, 0.025]
- Total variance explained: 38.7%

---

## Cell 5 — Save V_delta and free memory

**Purpose:** Save V_δ as a CSV with species index and the PCA object as a pickle. Free `delta_3d` and `delta_flat` from memory.

**Outputs saved to** `/scratch/ariana.l/Stage 5 PPE Representation Study/`:
- `stage5_Vdelta_ppe.csv` — V_δ embeddings (6697 × 4)
- `stage5_pca_delta.pkl` — fitted PCA object

---

## Cell 6 — Load GloBI interactions

**Purpose:** Load Stage 4 GloBI interactions to reconstruct training pairs.

**File:** `stage4_globi_conus_broad.csv` (715,215 rows, 9 columns)

**Key columns:**
- `sourceTaxonName` → pollinator species
- `targetTaxonName` → plant species

---

## Cell 7 — Reconstruct training pairs and assemble 35D feature matrix

**Purpose:** Reconstruct positive/negative training pairs (same logic as Stage 4, same random seed) and build the 35D feature matrix X.

**Positive pairs:** unique (pollinator, plant) pairs from GloBI, filtered to species present in Vf, Vp, and V_δ → 3,074 pairs

**Negative pairs:** randomly sampled (pollinator, plant) pairs not in GloBI, at 1:3 ratio → 9,222 pairs

**Feature assembly:**
```python
def build_features(row):
    vf = Vf_df.loc[row.plant].values          # 15D — plant spatial embedding
    vp = Vp_df.loc[row.pollinator].values     # 15D — pollinator spatial embedding
    vd = Vd_df.loc[row.plant].values          # 4D  — plant spatiotemporal embedding
    N  = np.dot(F_common.loc[row.plant].values,
                P_common.loc[row.pollinator].values)  # 1D — shared bin count
    return np.concatenate([vf, vp, [N], vd])  # 35D
```

**Important:** F and P must be aligned to `common_bins` before computing N (F has 3162 cols, P has 3895 cols; direct dot product raises ValueError).

**Output:** `X` shape `(12296, 35)`, `y` shape `(12296,)`, positive rate 0.250

---

## Cell 8 — Train/test split and logistic regression

**Purpose:** Train A′ logistic regression on 80% of pairs, evaluate on 20%.

**Split:** `train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)`
- Train: (9836, 35), Test: (2460, 35)

**Model:** `LogisticRegression(max_iter=1000, random_state=42)`

**Results:**
```
A′ (35D):  ROC-AUC 0.937,  PR-AUC 0.855
A2 (31D):  ROC-AUC 0.931,  PR-AUC 0.842   [Stage 4 baseline]
A3 (32D):  ROC-AUC 0.950,  PR-AUC 0.868   [Stage 4 best]
```

---

## Cell 9 — Save model and print summary

**Purpose:** Save trained model and print a structured summary of Stage 5.1 findings.

**Output saved:** `stage5_Aprime_logistic.pkl`
