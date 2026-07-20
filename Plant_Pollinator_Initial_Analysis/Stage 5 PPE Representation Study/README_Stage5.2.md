# Stage 5 — PPE Representation Study
## 5.2 — Probability Flowering Matrix (B)

**Date:** July 8, 2026
**Author:** ariana.l
**Cluster:** crow.engr.wustl.edu
**Notebook:** `Stage_5_2_Probability_Flowering_Matrix.ipynb`
**Output dir:** `/scratch/ariana.l/Stage 5 PPE Representation Study/`

---

## Motivation

Stage 5.1 (A′) tested appending a spatiotemporal plant embedding V_δ (4D) to the existing binary Vf. It beat A2 but fell short of A3. The core limitation was that V_δ only captured the plant side — the pollinator temporal signal was absent, and logistic regression could not reconstruct the plant-pollinator alignment from plant and pollinator embeddings independently.

Stage 5.2 asks a different question: instead of appending PPE information as an extra feature, what if we **replace the binary plant existence matrix F entirely** with a richer probability matrix PMf that encodes both spatial and temporal flowering information? The feature vector dimensionality stays the same as A2 (31D), but the plant embedding is now spatiotemporally informed.

---

## Architecture

```
PMf[species, bin, week] = norm (flowering probability from PPE opportunity surface)
Shape: (6697 species, 3160 bins, 52 weeks)
    └── flattened → (6697, 164320)
    └── PCA (randomized, 15 components) → Vf_prob (6697, 15)

Feature vector: [Vf_prob (15D), Vp (15D), N (1D)] = 31D
                        ↓
               Logistic Regression
                        ↓
           P(interaction | F, P) ∈ [0, 1]
```

**Key design choice:** PMf retains the full `(bin × week)` structure — it is not collapsed to a mean across weeks. This preserves the temporal dimension, encoding *where and when* the plant flowers rather than just *where*. Vp remains the binary pollinator existence matrix from Stage 4 (unchanged).

---

## Data Sources

All inherited from Stage 4:

| Data | Path on crow |
|------|-------------|
| Plant existence matrix (for common bins + N) | `/scratch/ariana.l/Stage 4 Link Prediction Model/stage4_F_existence_phenofield.csv` |
| Pollinator existence matrix | `/scratch/ariana.l/Stage 4 Link Prediction Model/stage4_P_existence_gbif_combined.csv` |
| Pollinator PCA embeddings (Vp) | `/scratch/ariana.l/Stage 4 Link Prediction Model/stage4_Vp_gbif.csv` |
| PPE opportunity surface | `/scratch/ariana.l/ppe-outputs/opportunity_surface/part_*.parquet` (6697 files) |
| GloBI interactions | `/scratch/ariana.l/Stage 4 Link Prediction Model/stage4_globi_conus_broad.csv` |

**Note:** Binary F is still loaded to derive `common_bins` and compute N (shared bin count). It is not used as the plant embedding in this model.

---

## Known Issues

### Grid offset (inherited from 5.1)
PPE centroids use a 0.25° offset grid (`.25`/`.75`) relative to F/P bins (`.0`/`.5`). Fixed by snapping PPE centroids to nearest 0.5° bin. Max spatial error ≤ 0.25°. Flag for Dan.

### Species present in PPE but absent from F
Some plant species appear in the PPE opportunity surface but not in the binary existence matrix F. These must be excluded from training pairs to avoid KeyErrors when computing N. Fixed by intersecting `Vf_prob_df.index` with `F.index` during pair construction.

### Training pairs not saved in Stage 4
Reconstructed from `stage4_globi_conus_broad.csv` using same logic and random seed (`seed=42`) as Stage 4.

---

## Outputs

| File | Description |
|------|-------------|
| `stage5_Vf_prob.csv` | Vf_prob embeddings (6697 species × 15 PCs) |
| `stage5_pca_pmf.pkl` | Fitted PCA object (randomized, 15 components) |
| `stage5_B_logistic.pkl` | Trained B logistic regression model |

---

## Results

| Model | Features | ROC-AUC | PR-AUC |
|-------|----------|---------|--------|
| A2 | binary Vf + Vp + N (31D) | 0.931 | 0.842 |
| A′ | binary Vf + Vp + N + V_δ (35D) | 0.937 | 0.855 |
| **B** | **PMf + Vp + N (31D)** | **0.938** | **0.856** |
| A3 | binary Vf + Vp + N + Δ scalar (32D) | 0.950 | 0.868 |

**Vf_prob PCA variance explained:** 48.6% total across 15 components
(PC1: 20.1%, PC2: 10.9%, PC3: 5.2%, PC4: 2.5%, PC5–15: 1.9% down to 0.5%)

**Training pairs:** 3,074 positive, 9,222 negative (25% positive rate)
**Train/test split:** 80/20, stratified, random_state=42

---

## Finding

B (ROC-AUC 0.938) beats A2 (+0.007) and is essentially tied with A′ (0.937 vs 0.938), despite using the same 31D dimensionality as A2. Replacing the binary plant matrix with a continuous spatiotemporal probability matrix provides a meaningful improvement at no cost in feature vector size.

However, B still falls short of A3 (−0.012 ROC-AUC). The pattern is consistent with 5.1: enriching the plant-side representation alone is insufficient. A3's scalar delta encodes the *plant-pollinator interaction* directly — it is a per-pair feature that captures phenological alignment between both species simultaneously, which neither B nor A′ can replicate with a linear classifier.

---

## Next Steps

Stage 5.3 (B′) will test the symmetric case: keep binary Vf, but replace the binary pollinator existence matrix P with a probability pollinator matrix PMp built from spatially and temporally resolved GBIF occurrence data.
