# Stage 5 — PPE Representation Study
## 5.1 — Spatiotemporal Embedding (A′)

**Date:** July 7, 2026
**Author:** ariana.l
**Cluster:** crow.engr.wustl.edu
**Notebook:** `Stage_5_1_Spatiotemporal_Embedding.ipynb`
**Output dir:** `/scratch/ariana.l/Stage 5 PPE Representation Study/`

---

## Motivation

Stage 4 established two link prediction models:
- **A2** (31D): spatial co-occurrence baseline — `[Vf, Vp, N]`
- **A3** (32D): adds a scalar PPE temporal overlap Δ = `mean(min(f_w, a_w))` across weeks — `[Vf, Vp, N, Δ]`

A3 outperformed A2 (ROC-AUC 0.950 vs 0.931), confirming that phenological timing adds predictive signal beyond spatial co-occurrence alone. Stage 5 asks: **can we encode PPE more richly than a single scalar?**

Stage 5.1 tests **A′**: replacing the scalar Δ with a full spatiotemporal embedding V_δ derived from the PPE opportunity surface — capturing not just *when* a plant flowers, but *where and when*.

---

## Architecture

```
Per-plant delta matrix (3160 bins × 52 weeks)
    └── norm values from PPE opportunity surface
    └── zero-padded for bins outside plant's range
    └── stacked across all 6697 plant species → (6697, 3160, 52)
    └── flattened → (6697, 164320)
    └── PCA (randomized, 4 components) → V_δ (6697, 4)

Feature vector: [Vf (15D), Vp (15D), N (1D), V_δ (4D)] = 35D
                    ↓
           Logistic Regression
                    ↓
        P(interaction | F, P) ∈ [0, 1]
```

**V_δ is per plant species** — same embedding for all pairs involving the same plant F. Analogous to how Vf is per plant and Vp is per pollinator. The pollinator temporal signal remains captured implicitly by Vp (spatial distribution).

---

## Data Sources

All inherited from Stage 4:

| Data | Path on crow |
|------|-------------|
| Plant existence matrix | `/scratch/ariana.l/Stage 4 Link Prediction Model/stage4_F_existence_phenofield.csv` |
| Pollinator existence matrix | `/scratch/ariana.l/Stage 4 Link Prediction Model/stage4_P_existence_gbif_combined.csv` |
| Plant PCA embeddings (Vf) | `/scratch/ariana.l/Stage 4 Link Prediction Model/stage4_Vf_phenofield.csv` |
| Pollinator PCA embeddings (Vp) | `/scratch/ariana.l/Stage 4 Link Prediction Model/stage4_Vp_gbif.csv` |
| PPE opportunity surface | `/scratch/ariana.l/ppe-outputs/opportunity_surface/part_*.parquet` (6697 files) |
| GloBI interactions | `/scratch/ariana.l/Stage 4 Link Prediction Model/stage4_globi_conus_broad.csv` |

---

## Known Issues

### Grid offset between F/P and PPE opportunity surface
F and P existence matrices use 0.5° bin centroids anchored at `.0` and `.5` (e.g. `24.5`, `25.0`). PPE opportunity surface centroids are anchored at `.25` and `.75` (e.g. `37.75`, `38.25`). Both are 0.5° grids, offset by 0.25°.

**Fix applied:** PPE centroids snapped to nearest F/P bin via `round(round(x * 2) / 2, 1)`. Maximum spatial error ≤ 0.25°, acceptable at 0.5° resolution. **Flag for Dan.**

### Training pairs not saved in Stage 4
`stage4_training_pairs.csv` was not saved to disk. Pairs were reconstructed from `stage4_globi_conus_broad.csv` using the same logic and random seed (`seed=42`) as Stage 4: positive pairs from unique GloBI (pollinator, plant) pairs; negatives sampled at 1:3 ratio.

---

## Outputs

| File | Description |
|------|-------------|
| `stage5_Vdelta_ppe.csv` | V_δ embeddings (6697 species × 4 PCs) |
| `stage5_pca_delta.pkl` | Fitted PCA object (randomized, 4 components) |
| `stage5_Aprime_logistic.pkl` | Trained A′ logistic regression model |

---

## Results

| Model | Features | ROC-AUC | PR-AUC |
|-------|----------|---------|--------|
| A2 | Vf + Vp + N (31D) | 0.931 | 0.842 |
| **A′** | **Vf + Vp + N + V_δ (35D)** | **0.937** | **0.855** |
| A3 | Vf + Vp + N + Δ scalar (32D) | 0.950 | 0.868 |

**V_δ PCA variance explained:** 38.7% total (PC1: 20.1%, PC2: 10.9%, PC3: 5.2%, PC4: 2.5%)

**Training pairs:** 3,074 positive, 9,222 negative (25% positive rate)
**Train/test split:** 80/20, stratified, random_state=42

---

## Finding

A′ beats the spatial baseline A2 (+0.006 ROC-AUC) but falls short of A3 (−0.013 ROC-AUC).

**Why the scalar wins:** A3's Δ = `mean(min(f_w, a_w))` directly encodes the phenological alignment between a specific plant-pollinator pair. A′'s V_δ captures only the plant-side spatiotemporal flowering structure — the pollinator temporal signal is absent from V_δ, and logistic regression (being linear) cannot reconstruct the plant-pollinator alignment from V_δ and Vp independently.

In other words: A3 hands the model the *interaction* directly. A′ hands the model the *ingredients* and asks it to infer the interaction — which a linear classifier cannot do.

**Alternative representations considered and rejected:**
- Per-pair delta matrix `min(norm_F(bin, week), a_A(week))`: makes V_δ pair-specific, breaking the PCA approach (PCA must be fit across training examples, not per pair)
- Spatially resolved pollinator activity (per-bin weekly histograms from GBIF): significant extra build step, high sparsity, uncertain gain given A′ result

---

## Next Steps

Stage 5.2 onwards will explore the B variants from the design diagram: replacing the binary existence matrix with probability matrices (PMf → Vf replacement) and testing whether continuous occupancy probability adds signal over binary existence.
