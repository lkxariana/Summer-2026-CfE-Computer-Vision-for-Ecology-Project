# ANTHEIA — Repository Guide

This guide explains how the repository is organized, what each notebook does,
and in what order to run things. It is intended for anyone picking up this
codebase for the first time, including after a gap.

For the full project overview, model descriptions, and results, see `README.md`.
For extended methodology, limitations, and next steps, see the `docs/` folder.

---

## Quick Reference: Where Is What?

| I want to... | Go to... |
|---|---|
| Understand the project | `README.md` → `docs/01-overview.md` |
| Understand the data sources | `docs/02-dataset.md` |
| Understand the methodology | `docs/03-methodology.md` |
| See the results | `docs/04-experiments.md` |
| Understand scope and limitations | `docs/05-scope-and-limitations.md` |
| Pick up the SDM work in September | `docs/06-extensions-and-next-steps.md` |
| Find citations | `docs/07-reading-list.md` |
| Run the full pipeline from scratch | Follow the order below |
| Find a specific notebook | See the notebook map below |

---

## Running Order

The pipeline has five stages. Each stage depends on the outputs of the previous one.

### Stage 1 — Data

Run `data/full_scale_run/` notebooks in order:

1. `01_plant_flowering.ipynb` → builds `stage4_F_existence_phenofield.csv`
2. `02_interaction_edgelist.ipynb` → builds `globiinteractions_conus.csv`
3. `03_pollinator_occurrences.ipynb` → builds `stage4_P_existence_corrected.csv`, `stage4_Vp_corrected.csv`, `a_curves_corrected.csv`
4. `04_spatial_binning.ipynb` → verifies F/P alignment (no new outputs)
5. `05_phenology_overlap.ipynb` → builds `delta_overlap_pairs.csv`
6. `06_jaccard_range_overlap.ipynb` → builds `top100_jaccard_pairs_fullscale.csv`

### Stage 2 — Representation

Run `representation/` notebooks in order:

1. `01_existence_matrices.ipynb` → verifies F/P, builds index maps
2. `02_pca_embeddings.ipynb` → builds `stage4_Vf_phenofield.csv`, `stage4_Vp_corrected.csv`
3. `03_ppe_integration.ipynb` → builds `f_curves_ppe.csv`, `stage5_Vdelta_ppe.csv`, `stage5_Vdelta_15d.csv`, `stage5_Vf_prob.csv`
4. `04_activity_curves.ipynb` → verifies `a_curves_corrected.csv` (no rebuild needed)

### Stage 3 — Model

Run `model/` notebooks in order:

1. `01_baseline_spatial.ipynb` → single-seed Spatial Baseline result
2. `02_antheia_models.ipynb` → all five primary models + pollinator-side ablations (ANTHEIA-PMp, ANTHEIA-TMp)

### Stage 4 — Evaluation

Run `evaluation/` notebooks in order:

1. `01_seed_testing.ipynb` → **primary results** (5-seed mean ± std, all five models, GBIF a_curves)
2. `02_sdm_comparison.ipynb` → SDM results (5-seed, SDM a_curves, 1,615-species test set — rerun when full SDM arrives)

### Stage 5 — Visualization

1. `visualization/01_spatial_figures.ipynb` → spatial maps + Hovmöller diagram for *Achillea millefolium*

---

## Notebook Map

### `data/small_scale_testing/`
Exploratory go/no-go notebooks using top 50 plant species.
Not used in final results. Run these first if verifying the pipeline on a new machine.

| Notebook | What it does |
|---|---|
| `01_plant_flowering.ipynb` | Loads top 50 species from PhenoField |
| `02_interaction_edgelist.ipynb` | Filters GloBI to 50 target plants → 18 clean edges |
| `03_pollinator_occurrences.ipynb` | Spatial density map of top 50 pollinators |
| `04_spatial_binning.ipynb` | Top 20 overlap bins by density score |
| `05_phenology_overlap.ipynb` | Δ coefficient for 18 edges × bins |
| `06_jaccard_range_overlap.ipynb` | Top 100 pairs by Jaccard range overlap |

### `data/full_scale_run/`
Full pipeline. These produce the files used in all reported results.

| Notebook | What it does | Key output |
|---|---|---|
| `01_plant_flowering.ipynb` | Full PhenoField → F matrix | `stage4_F_existence_phenofield.csv` |
| `02_interaction_edgelist.ipynb` | Full GloBI filter | `globiinteractions_conus.csv` |
| `03_pollinator_occurrences.ipynb` | `pollinator_observations_v2.csv` → P, Vp, a_curves | `stage4_P_existence_corrected.csv`, `a_curves_corrected.csv` |
| `04_spatial_binning.ipynb` | F/P bin alignment check | — |
| `05_phenology_overlap.ipynb` | Δ for full pair universe | `delta_overlap_pairs.csv` |
| `06_jaccard_range_overlap.ipynb` | Jaccard for all 6,466 × 24,939 pairs | `top100_jaccard_pairs_fullscale.csv` |

### `representation/`

| Notebook | What it does | Key output |
|---|---|---|
| `01_existence_matrices.ipynb` | Load F/P, verify alignment, build index maps | — |
| `02_pca_embeddings.ipynb` | PCA 15D on F → Vf; PCA 15D on P → Vp | `stage4_Vf_phenofield.csv` |
| `03_ppe_integration.ipynb` | PPE opportunity surface → f_curves, V_δ 4D, V_δ 15D, Vf_prob | `f_curves_ppe.csv`, `stage5_Vdelta_ppe.csv`, `stage5_Vdelta_15d.csv`, `stage5_Vf_prob.csv` |
| `04_activity_curves.ipynb` | Verify a_curves coverage and distribution | — |

### `model/`

| Notebook | What it does |
|---|---|
| `01_baseline_spatial.ipynb` | Spatial Baseline: [Vf, Vp, N] = 31D, single seed |
| `02_antheia_models.ipynb` | All 5 primary models + ANTHEIA-PMp/TMp ablations, with SDM motivation |

### `evaluation/`

| Notebook | What it does | Key output |
|---|---|---|
| `01_seed_testing.ipynb` | 5-seed evaluation, all primary models, GBIF a_curves | `corrected_model_results.csv` |
| `02_sdm_comparison.ipynb` | 5-seed evaluation, all models, SDM a_curves | `sdm_model_results.csv` |

### `visualization/`

| Notebook | What it does | Key output |
|---|---|---|
| `01_spatial_figures.ipynb` | Spatial maps + Hovmöller diagram (*Achillea millefolium*) | `fig_antheia_combined.png` |

---

## Key Technical Facts

**Bin format:** `"lat_lon"` e.g. `"34.5_-120.0"` at 0.5° resolution. F and P must share the same column set before computing N.

**N computation:** Always `F_common @ P_common` where both are restricted to the 3,162 common CONUS bins. Never use raw F and P arrays.

**Week formula:** `week = (doy - 1) // 7`, clipped to [0, 51]. Not `doy // 7`.

**Negative sampling:** `np.random.default_rng(seed)` — not `np.random.seed()`.

**Seeds:** [42, 0, 1, 2, 3].

**f_curves source:** Always build from the full PPE opportunity surface parquet files (`part_*.parquet`). Not from `flowering_curves_used.parquet` — that file covers only 50 species.

**SDM coordinate offset:** `lat_bin = centroid_lat - 0.25`, `lon_bin = centroid_lon - 0.25`. Required to match F/P bin convention.

---

## What To Do First in September

1. Get Dan's full SDM activity curves (25,466-species coverage)
2. Rerun `evaluation/02_sdm_comparison.ipynb` with the full SDM
3. Update results tables in `docs/01-overview.md` and `docs/04-experiments.md`
4. Begin paper writing

See `docs/06-extensions-and-next-steps.md` for full context on the SDM integration and the four longer-term archival paper directions.

---

## Data Locations (on crow)

All data lives under `/scratch/ariana.l/`. The repo itself is at
`/scratch/ariana.l/CfE2026CVforEcology`, branch `Clean-for-ANTHEIA`.

| File | Path |
|---|---|
| F matrix | `/scratch/ariana.l/Stage 4 Link Prediction Model/stage4_F_existence_phenofield.csv` |
| P matrix | `/scratch/ariana.l/New Stage 4 Link Prediction Model/stage4_P_existence_corrected.csv` |
| Vf | `/scratch/ariana.l/Stage 4 Link Prediction Model/stage4_Vf_phenofield.csv` |
| Vp | `/scratch/ariana.l/New Stage 4 Link Prediction Model/stage4_Vp_corrected.csv` |
| f_curves | `/scratch/ariana.l/New Stage 4 Link Prediction Model/f_curves_ppe.csv` |
| a_curves (GBIF) | `/scratch/ariana.l/New Stage 4 Link Prediction Model/a_curves_corrected.csv` |
| a_curves (SDM) | `/scratch/ariana.l/Stage 6 Seed Testing/a_curves_sdm.csv` |
| V_δ 4D | `/scratch/ariana.l/Stage 5 PPE Representation Study/stage5_Vdelta_ppe.csv` |
| V_δ 15D | `/scratch/ariana.l/Stage 5 PPE Representation Study/stage5_Vdelta_15d.csv` |
| Vf_prob (PMf) | `/scratch/ariana.l/Stage 5 PPE Representation Study/stage5_Vf_prob.csv` |
| PPE opportunity surface | `/scratch/ariana.l/ppe-outputs/opportunity_surface/part_*.parquet` |
| SDM activity curves | `/scratch/ariana.l/Stage 6 Seed Testing/pollinator_activity_curves.parquet` |
| SDM manifest | `/scratch/ariana.l/Stage 6 Seed Testing/species_manifest.csv` |
