# ANTHEIA

**Area-based Niche and Temporal Habitat Embedding for Interaction Analysis**

ANTHEIA is a plant-pollinator interaction link prediction pipeline that combines spatial co-occurrence embeddings with PPE (Phenological Plant Embeddings) spatiotemporal representations to predict the probability that a given plant-pollinator pair interacts.

Named after the Greek goddess of flowers.

---

## Pipeline Structure

```
data/           → Data acquisition and preprocessing
representation/ → Feature engineering (existence matrices, PCA embeddings, PPE integration)
model/          → Link prediction models
evaluation/     → Results, seed testing, SDM comparison
visualization/  → Spatial and temporal figures, maps
docs/           → Extended methodology and reading list
```

---

## Models

| **Model** | **Feature Vector** | **Dim** | **Description** |
|---|---|---|---|
| Spatial Baseline | Vf + Vp + N | 31D | Spatial co-occurrence only; no PPE |
| ANTHEIA-Scalar | Vf + Vp + N + Δ | 32D | + PPE temporal overlap scalar |
| ANTHEIA-4D | Vf + Vp + N + V_δ (4D) | 35D | + PPE spatiotemporal embedding (4D) |
| ANTHEIA-15D | Vf + Vp + N + V_δ (15D) | 46D | + PPE spatiotemporal embedding (15D) |
| ANTHEIA-PMf | Vf_prob + Vp + N | 31D | PMf replaces binary plant existence matrix |

*PMf: continuous PPE flowering probability matrix, replacing the binary plant existence matrix Vf.*

Each model can be run with either GBIF-derived or SDM-derived pollinator activity curves (`a_curves`). See `docs/06-extensions-and-next-steps.md` for details.

---

## Key Results (5-seed mean ± std, GBIF a_curves, corrected pollinator data)

| **Model** | **ROC-AUC** | **PR-AUC** |
|---|---|---|
| Spatial Baseline | 0.9571 ± 0.0375 | 0.9387 ± 0.0335 |
| ANTHEIA-Scalar | 0.9599 ± 0.0365 | 0.9403 ± 0.0339 |
| **ANTHEIA-4D** | **0.9602 ± 0.0232** | **0.9424 ± 0.0257** |
| ANTHEIA-15D | 0.9514 ± 0.0274 | 0.9374 ± 0.0245 |
| ANTHEIA-PMf | 0.9560 ± 0.0298 | 0.9357 ± 0.0338 |

A key finding from the ablation study (Stage 5) is that plant-side temporal enrichment via PPE consistently improves performance, while pollinator-side temporal enrichment using raw GBIF observations consistently hurts it — due to observation-density bias rather than genuine phenological signal. SDM (Dan Cher's model-predicted pollinator activity curves) is designed to address exactly this: replacing noisy GBIF-derived curves with bias-corrected model predictions on the pollinator side. SDM-based results are reported separately in `evaluation/03_sdm_comparison.ipynb`. Full SDM coverage (25,466 species) is pending; current SDM results use a 1,615-species test set. See `docs/06-extensions-and-next-steps.md`.

---

## Data Sources

| **Dataset** | **Description** | **Records** |
|---|---|---|
| PhenoField (dcher95/phenofield) | Plant flowering events, CONUS 2013–2026 | 2.65M rows, 6,466 species |
| GloBI | Documented plant-pollinator interactions | 715,215 CONUS records |
| GBIF (pollinator_observations_v2.csv) | Pollinator occurrence records, CONUS 2013–2026 | 25,466 species |
| PPE opportunity surface | Phenological Plant Embeddings — climate-driven flowering probability per species per bin per week | 6,697 species |
| SDM (Dan Cher) | Model-predicted pollinator activity curves | 1,615 species (full 25,466 pending) |

---

## Authors

Kexing Li, Dan Cher, Nathan Jacobs

MVRL, Washington University in St. Louis
