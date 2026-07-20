# Stage 4 — Link Prediction Model

## Overview

Stage 4 builds a binary link prediction model for plant-pollinator interactions. Given a (flowering plant, pollinator) species pair, the model outputs P(interaction | F, P) ∈ [0, 1].

This stage replaces the Stage 3 go/no-go framework. The lab's goal is not to test whether PPE is useful in isolation, but to build a full link prediction pipeline and use PPE's temporal signal as an additional feature on top of spatial co-occurrence.

---

## Architecture

```
Flowering plant F  →  binary existence matrix (species × bins)  →  PCA (15D)  →  Vf
Pollinator P       →  binary existence matrix (species × bins)  →  PCA (15D)  →  Vp
                                                                                    ↓
                                          N = shared bin count (F ∩ P)         [Vf, Vp, N]  →  31D feature vector
                                                                                    ↓
                                                                         Logistic Regression
                                                                                    ↓
                                                                     P(interaction | F, P) ∈ [0, 1]
```

**Spatial resolution:** 0.5° × 0.5° bins, CONUS only (lon: -125 to -66, lat: 24 to 50)

**Existence matrix:** binary — 1 if species observed in bin, 0 otherwise

**PCA:** fit once per matrix (F and P separately), each species projected to 15D

---

## Data Sources

| Data | Source | Path on crow |
|---|---|---|
| Plant-pollinator interactions (labels) | GloBI | `/scratch/ariana.l/CfE2026CVforEcology/rawpollinatordata/interactions.csv.gz` |
| Pollinator occurrences (insects) | GBIF gbif_0007192 | `/scratch/ariana.l/Plant Pollinator Initial Analysis/gbif_0007192_observations_v2.csv` |
| Pollinator occurrences (birds/eBird) | GBIF gbif_0007204 | `/scratch/ariana.l/Plant Pollinator Initial Analysis/gbif_0007204_observations_v2.csv` |
| Plant occurrences | PhenoField HuggingFace (`dcher95/phenofield`) | streamed at runtime |
| PPE opportunity surface (pending) | Dan / Box | `/scratch/ariana.l/ppe-outputs/opportunity_surface/` |

---

## Outputs

| File | Description |
|---|---|
| `stage4_globi_conus_broad.csv` | GloBI CONUS interactions, broad types (715,215 rows) |
| `stage4_F_existence_phenofield.csv` | Plant existence matrix (6466 × 3162) |
| `stage4_P_existence_gbif_combined.csv` | Pollinator existence matrix (4515 × 3895) |
| `stage4_Vf_phenofield.csv` | Plant PCA embeddings (6466 × 15), variance explained: 0.399 |
| `stage4_Vp_gbif.csv` | Pollinator PCA embeddings (4515 × 15), variance explained: 0.757 |
| `phenofield_plant_occurrences.parquet` | Raw plant occurrences from PhenoField (2,650,448 rows, 6,466 species) |
| `stage4_A2_logistic.pkl` | Trained A2 baseline logistic regression model |

---

## Models

### A2 — Baseline (no PPE)

**Features:** `[Vf (15D), Vp (15D), N (1D)]` = 31D

**Labels:** GloBI interactions = 1 (positive); randomly sampled non-pairs = 0 (negative)

**Class balance:** ~25% positive (3,148 positive, 9,444 negative)

**Train/test split:** 80/20, stratified

**Results:**
- ROC-AUC: **0.931**
- PR-AUC: **0.842**

### A3 — With PPE (pending)

**Features:** `[Vf (15D), Vp (15D), N (1D), Δ (1D)]` = 32D

Δ = temporal overlap between plant flowering curve (PPE) and pollinator activity curve. PPE helps where N is high but the species are active at different times of year — A2 would predict a false positive, A3 should correct it.

**Status:** pending completion of `opportunity_surface` rclone transfer from Box.

---

## Known Limitations

- **iNat observation bias:** PhenoField plant data is iNat-sourced and biased toward human-populated areas. Variance explained for F PCA (0.399) is lower than P (0.757) as a result. This is a known limitation of the lab's data.
- **Negative sampling:** negatives are randomly sampled from all (plant, pollinator) combinations with no confirmed non-interaction label — open-world assumption.
- **Pollinator orders pending:** 5 orders present in GloBI but absent from GBIF download (Hemiptera, Passeriformes, Thysanoptera, Neuroptera, Chiroptera) — awaiting Dan's confirmation on whether to supplement.

---

## Next Steps

1. Complete `opportunity_surface` rclone transfer
2. Compute Δ per (plant, pollinator) pair from PPE outputs
3. Train A3 model with 32D features
4. Compare A2 vs A3 — PPE adds value if `score(A3) > score(A2)`, especially under spatial hold-out where N collapses but Δ holds
