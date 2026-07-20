# Stage 5 — PPE Representation Study
## 5.3 — Probability Pollinator Matrix (B′)

**Date:** July 8, 2026
**Author:** ariana.l
**Cluster:** crow.engr.wustl.edu
**Notebook:** `Stage_5_3_Probability_Pollinator_Matrix.ipynb`
**Output dir:** `/scratch/ariana.l/Stage 5 PPE Representation Study/`

---

## Motivation

Stage 5.2 (B) tested replacing the binary plant existence matrix F with a spatiotemporal probability matrix PMf derived from the PPE opportunity surface. B matched A′ and beat A2, confirming that enriching the plant-side representation adds predictive signal.

Stage 5.3 (B′) tests the symmetric case: keep binary Vf, and replace the binary pollinator existence matrix P with a spatiotemporal probability matrix PMp built from GBIF occurrence records. If plant-side enrichment helps, does pollinator-side enrichment help equally?

---

## Architecture

```
PMp[species, bin, week] = normalized weekly observation count from GBIF
    - Source: GBIF insects (gbif_0007192) + birds/eBird (gbif_0007204), combined
    - doy → week index via (doy - 1) // 7, clipped to [0, 51]
    - Normalization: weekly count / total count per (species, bin)
    - MIN_OBS = 5: bins with fewer than 5 total observations zero-padded
    - Shape: (4425 species, 3160 bins, 52 weeks) = 2.91 GB
    └── flattened → (4425, 164320)
    └── PCA (randomized, 15 components) → Vp_prob (4425, 15)

Feature vector: [Vf (15D), Vp_prob (15D), N (1D)] = 31D
                        ↓
               Logistic Regression
                        ↓
           P(interaction | F, P) ∈ [0, 1]
```

**Key design choice:** Vf remains the binary plant PCA embedding from Stage 4 (unchanged). Only the pollinator side is replaced. This is the direct symmetric counterpart to B.

---

## Data Sources

| Data | Path on crow |
|------|-------------|
| Plant existence matrix | `/scratch/ariana.l/Stage 4 Link Prediction Model/stage4_F_existence_phenofield.csv` |
| Pollinator existence matrix (for common bins + N) | `/scratch/ariana.l/Stage 4 Link Prediction Model/stage4_P_existence_gbif_combined.csv` |
| Plant PCA embeddings (Vf) | `/scratch/ariana.l/Stage 4 Link Prediction Model/stage4_Vf_phenofield.csv` |
| GBIF insects | `/scratch/ariana.l/Plant Pollinator Initial Analysis/gbif_0007192_observations_v2.csv` |
| GBIF birds/eBird | `/scratch/ariana.l/Plant Pollinator Initial Analysis/gbif_0007204_observations_v2.csv` |
| GloBI interactions | `/scratch/ariana.l/Stage 4 Link Prediction Model/stage4_globi_conus_broad.csv` |

**Note:** Binary P is still loaded to derive `common_bins` and compute N. It is not used as the pollinator embedding in this model.

---

## Known Issues

### GBIF column names
GBIF files use `lat` / `lon`, not `decimalLatitude` / `decimalLongitude`. Confirmed during Cell 2 build.

### Grid offset (inherited from 5.1/5.2)
GBIF coordinates snapped to nearest 0.5° common bin grid via `round(round(x * 2) / 2, 1)`. Max spatial error ≤ 0.25°.

### High sparsity in PMp
PMp is only 1.1% non-zero (7,822,389 / 727,116,000 entries) — far sparser than PMf from PPE. GBIF observations are opportunistic and spatially biased toward populated areas, leaving most pollinator × bin combinations empty or below MIN_OBS threshold.

### Training pairs not saved in Stage 4
Reconstructed from `stage4_globi_conus_broad.csv`. Note: B′ recovers 3,148 positive pairs (vs 3,074 in B and A′) because Vp_prob_df covers more pollinator species from the combined GBIF dataset.

---

## Outputs

| File | Description |
|------|-------------|
| `stage5_Vp_prob.csv` | Vp_prob embeddings (4425 species × 15 PCs) |
| `stage5_pca_pmp.pkl` | Fitted PCA object (randomized, 15 components) |
| `stage5_Bprime_logistic.pkl` | Trained B′ logistic regression model |

---

## Results

| Model | Features | ROC-AUC | PR-AUC |
|-------|----------|---------|--------|
| **B′** | **binary Vf + PMp + N (31D)** | **0.924** | **0.828** |
| A2 | binary Vf + binary Vp + N (31D) | 0.931 | 0.842 |
| A′ | binary Vf + binary Vp + N + V_δ (35D) | 0.937 | 0.855 |
| B | PMf + binary Vp + N (31D) | 0.938 | 0.856 |
| A3 | binary Vf + binary Vp + N + Δ scalar (32D) | 0.950 | 0.868 |

**Vp_prob PCA variance explained:** 41.5% total across 15 components
(PC1: 19.7%, PC2: 5.9%, PC3: 3.5%, PC4: 2.4%, PC5–15: 1.6% down to 0.6%)

**Training pairs:** 3,148 positive, 9,444 negative (25% positive rate)
**Train/test split:** 80/20, stratified, random_state=42

---

## Finding

B′ is the worst-performing model across all Stage 5 experiments — it falls **below** the binary baseline A2 (ROC-AUC 0.924 vs 0.931). Replacing binary Vp with PMp introduces noise rather than signal.

**Why B′ underperforms A2:** The fundamental issue is data quality asymmetry between the two sides of the model:

- **Plant side (PMf, B):** Built from the PPE opportunity surface — a model-derived, dense, high-quality flowering probability signal specifically designed for CONUS plant species. Spatially and temporally complete by design.
- **Pollinator side (PMp, B′):** Built from raw GBIF occurrence records — opportunistic citizen science observations, spatially biased toward populated areas, temporally sparse per bin (only 1.1% non-zero after MIN_OBS=5 filter). The resulting PMp is dominated by zeros, and the PCA captures observer effort patterns more than genuine pollinator phenology.

**Broader pattern across Stage 5:** Plant-side enrichment consistently helps (B > A2); pollinator-side enrichment consistently hurts (B′ < A2). The asymmetry reflects the fundamental difference in data provenance: PPE is a curated model output while GBIF is raw observational data.

---

## Next Steps

With A, A′, B, and B′ all tested, the Stage 5 ablation study is complete. A3 (scalar PPE delta, 32D) remains the best model at ROC-AUC 0.950. The results support the following conclusions for the paper:
1. Temporal information from PPE adds meaningful signal over spatial co-occurrence alone.
2. The scalar encoding of plant-pollinator phenological alignment (A3) is more effective than richer plant-side spatiotemporal embeddings for a linear classifier.
3. Data quality on the pollinator side is the primary bottleneck — improving GBIF coverage or using a model-derived pollinator activity surface (analogous to PPE on the plant side) may be necessary to achieve gains symmetric to B.
