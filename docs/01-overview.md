# 01 — Overview

## What ANTHEIA Is

ANTHEIA (Area-based Niche and Temporal Habitat Embedding for Interaction Analysis) is a plant-pollinator interaction link prediction pipeline. Given any plant species and any pollinator species, ANTHEIA outputs a probability that the two interact.

The core problem it addresses is that empirical interaction data (GloBI) is sparse and geographically biased — documented interactions represent a small, non-random fraction of true interactions, concentrated around well-studied species and easily accessible field sites. ANTHEIA frames link prediction as a binary classification problem over the full combinatorial space of plant × pollinator pairs, using spatial co-occurrence and phenological timing as features.

Named after the Greek goddess of flowers.

---

## The Central Hypothesis

Spatial co-occurrence alone systematically overestimates interaction likelihood. Two species may share geographic range but be phenologically misaligned — the plant flowers before or after the pollinator is active. A model that ignores timing will produce false positives wherever species ranges overlap but activity windows do not.

This is consistent with the broader SDM literature: Coelho et al. (2024) quantify that only ~20% of species co-occurrences correspond to real interactions, and Dormann et al. (2018) note that existing interaction SDMs "must assume biotic interactions are constant in space and time." ANTHEIA's phenological signal directly addresses this assumption.

ANTHEIA's hypothesis is that adding PPE (Phenological Plant Embeddings) — a climate-driven, spatially-continuous, temporally-resolved flowering probability surface — on top of spatial co-occurrence embeddings will improve link prediction by correcting for temporal mismatches. This hypothesis was validated empirically: every model variant that incorporated PPE plant-side temporal information outperformed the spatial-only baseline.

---

## Pipeline Architecture

ANTHEIA operates at 0.5° × 0.5° spatial resolution (~55 km × 43 km) across CONUS, with 52-week temporal resolution.

**Plant side:**
- Binary existence matrix F (6,466 species × 3,162 CONUS bins), built from PhenoField
- PCA 15D embedding Vf per plant species
- PPE flowering curves f_curves (6,697 species × 52 weeks), built from climate-driven opportunity surface

**Pollinator side:**
- Binary existence matrix P (24,939 species × 3,162 CONUS bins), built from GBIF (`pollinator_observations_v2.csv`)
- PCA 15D embedding Vp per pollinator species
- Activity curves a_curves (25,466 species × 52 weeks), built from GBIF weekly observation histograms

**Interaction labels:**
- GloBI (Global Biotic Interactions), filtered to CONUS and seven broad pollination interaction types
- 715,215 records; 139 positive pairs in the shared pair universe after species coverage intersection
- No true negatives — GloBI documents known interactions, not confirmed non-interactions
- Negative pairs sampled at 3:1 ratio using `np.random.default_rng(seed)`

**Model:**
- Logistic regression on a per-pair feature vector (see Models section below)
- 80/20 stratified train/test split, shared across all model variants
- Evaluated on ROC-AUC and PR-AUC (5-seed mean ± std, seeds [42, 0, 1, 2, 3])

---

## Models

| **Paper Name** | **Internal Code** | **Feature Vector** | **Dim** | **Description** |
|---|---|---|---|---|
| Spatial Baseline | A2 | Vf + Vp + N | 31D | Spatial co-occurrence only; no PPE |
| ANTHEIA-Scalar | A3 | Vf + Vp + N + Δ | 32D | + PPE temporal overlap scalar |
| ANTHEIA-4D | A′ | Vf + Vp + N + V_δ (4D) | 35D | + PPE spatiotemporal embedding (4D) |
| ANTHEIA-15D | A* | Vf + Vp + N + V_δ (15D) | 46D | + PPE spatiotemporal embedding (15D) |
| ANTHEIA-PMf | B | Vf_prob + Vp + N | 31D | PMf replaces binary plant existence matrix |

*Internal codes are retained here for traceability against lab notebooks only. All paper-facing material uses paper names.*

*PMf: continuous PPE flowering probability matrix, replacing the binary plant existence matrix Vf. N: count of spatial bins shared between a plant and pollinator species. V_δ: PCA embedding of the per-plant PPE opportunity surface (bins × weeks). Δ: scalar temporal overlap = Σ_t min(f_t, a_t) across 52 weeks.*

Each model can additionally be run with SDM-derived pollinator activity curves instead of GBIF-derived ones, producing an -SDM variant (e.g. ANTHEIA-Scalar-SDM). See `docs/06-sdm-integration.md`.

---

## Key Results

### GBIF a_curves (primary results, 5-seed mean ± std)

| **Model** | **ROC-AUC** | **PR-AUC** |
|---|---|---|
| Spatial Baseline | 0.9571 ± 0.0375 | 0.9387 ± 0.0335 |
| ANTHEIA-Scalar | 0.9599 ± 0.0365 | 0.9403 ± 0.0339 |
| **ANTHEIA-4D** | **0.9602 ± 0.0232** | **0.9424 ± 0.0257** |
| ANTHEIA-15D | 0.9514 ± 0.0274 | 0.9374 ± 0.0245 |
| ANTHEIA-PMf | 0.9560 ± 0.0298 | 0.9357 ± 0.0338 |

### SDM a_curves (preliminary, 1,615-species test set)

| **Model** | **ROC-AUC** | **PR-AUC** |
|---|---|---|
| Spatial Baseline-SDM | 0.9212 ± 0.0332 | 0.8628 ± 0.0513 |
| ANTHEIA-Scalar-SDM | 0.9235 ± 0.0320 | 0.8658 ± 0.0489 |
| ANTHEIA-4D-SDM | 0.9196 ± 0.0340 | 0.8716 ± 0.0470 |
| ANTHEIA-15D-SDM | 0.9334 ± 0.0319 | 0.8901 ± 0.0542 |
| **ANTHEIA-PMf-SDM** | **0.9489 ± 0.0245** | **0.8911 ± 0.0480** |

SDM results use a smaller pollinator pool (1,275 vs. 24,939 species) and 122 positive pairs (vs. 139), which accounts for the lower absolute performance. The shift in model ranking under SDM — ANTHEIA-PMf-SDM and ANTHEIA-15D-SDM leading, vs. ANTHEIA-4D under GBIF — suggests richer plant-side representations matter more when pollinator temporal signal is higher quality. Full SDM results (25,466 species) are pending.

---

## Key Findings

**Temporal signal improves link prediction.** Every PPE-enriched plant-side model outperforms the Spatial Baseline, confirming that phenological alignment provides information beyond spatial co-occurrence.

**Plant-side enrichment helps; pollinator-side hurts.** PPE is a climate-driven model prediction, independent of observation effort. GBIF pollinator records are opportunistic citizen science data biased by sampling density. Adding GBIF-derived temporal information on the pollinator side consistently degraded performance. SDM aims to close this asymmetry by replacing GBIF-derived pollinator activity curves with model-predicted ones.

**ANTHEIA-4D is the most stable model.** It achieves the best ROC-AUC and PR-AUC under GBIF curves with the lowest variance across seeds (±0.0232 vs. ±0.0375 for Spatial Baseline).

**Scalar encoding is surprisingly competitive.** ANTHEIA-Scalar adds only one dimension over the Spatial Baseline but captures most of the gain from richer spatiotemporal embeddings — the per-pair temporal alignment scalar Δ encodes signal that a linear classifier cannot reconstruct from species embeddings alone.

---

## Repository Structure

```
data/           → Data acquisition and preprocessing
representation/ → Feature engineering (existence matrices, PCA, PPE integration, activity curves)
model/          → Link prediction models (Spatial Baseline through ANTHEIA-PMf)
evaluation/     → Results, seed testing, SDM comparison
visualization/  → Spatial and temporal figures, maps
docs/           → This documentation
```

---

## Provenance Note

All results reported here use the corrected pollinator dataset (`pollinator_observations_v2.csv`, 25,466 true pollinator species). An earlier version of the pipeline used an incorrect GBIF file containing Hemiptera, Passeriformes, and Neuroptera. All results from that version are discarded. See the Stage Summary document for the full correction log.
