# 02 — Dataset

ANTHEIA draws on four primary data sources: PhenoField (plant flowering), GBIF (pollinator occurrences), GloBI (documented interactions), and the PPE opportunity surface (climate-driven flowering probability). The pipeline passed through three distinct data tiers as the project evolved — an initial small-scale exploratory phase, the full analysis, and a pending update that will complete the SDM integration.

---

## Tier 1 — Small-Scale Exploratory Data (Go/No-Go Phase)

These datasets were used during the initial exploratory phase to test pipeline feasibility and validate methodological choices before committing to full-scale processing. They are **not** used in the final model or reported results.

**Plants:** Top 50 species from PhenoField by observation frequency (ranks 2–51, excluding unknown rank 1). 208,567 flowering records. Species like *Achillea millefolium*, *Asclepias syriaca*, *Larrea tridentata*.

**Pollinators:** Original GBIF download (~23.7M records, 13,635 species after filtering), covering Hymenoptera, Lepidoptera, Diptera, Coleoptera, and Trochilidae. Later discovered to be an incorrect file — see Tier 2.

**Interaction edges:** GloBI filtered to the 50 target plants.
- Initial pull: **54 edges** (raw)
- After removing 15 self-paired artifacts: **39 edges**
- After Stage 3 cleaning (removing 18 plant species mislabeled as pollinators in the pollinator column, and 1 biologically implausible pair — a coho salmon): **18 edges**

The 18-edge positive set was used for the go/no-go spatial block hold-out test and the effort-stratified background sampling framework (408 total pairs at ~21:1 background ratio, reduced from ~71× to ~15× observation-effort bias after stratified sampling). This phase was ultimately superseded by the full pipeline; see the Stage Summary document for details.

**Spatial analysis:** Top 20 spatial overlap bins (0.5° resolution), top 100 Jaccard range-overlap pairs across all plant × pollinator combinations.

---

## Tier 2 — Full Analysis Dataset (Current)

These are the datasets used in all reported results.

### Plants

| Source | Records | Species | Bins |
|---|---|---|---|
| PhenoField (dcher95/phenofield, 107 shards) | 2,650,448 | 6,466 | 3,162 CONUS bins |

- Binary existence matrix F: 6,466 species × 3,162 bins, sparsity 0.980
- PCA 15D embedding Vf: 39.9% variance explained
- PPE flowering curves f_curves: 6,697 species × 52 weeks, built by streaming all 6,697 PPE parquet files from `/scratch/ariana.l/ppe-outputs/opportunity_surface/part_*.parquet`

**Important:** f_curves must be built from the full PPE opportunity surface, not from `flowering_curves_used.parquet` (which covers only 50 species from the exploratory phase).

### Pollinators

| Source | File | Records | Species |
|---|---|---|---|
| GBIF | `pollinator_observations_v2.csv` | — | 25,466 |

- Binary existence matrix P: 24,939 species × 3,162 bins, sparsity 0.985
- PCA 15D embedding Vp: 46.2% variance explained
- Activity curves a_curves: 25,466 species × 52 weeks, normalized weekly observation histograms

**Pollinator data correction:** An earlier version of the pipeline used `gbif_0007192_observations_v2.csv`, which contained Hemiptera, Passeriformes, and Neuroptera — zero true pollinators. The correct file (`pollinator_observations_v2.csv`, 25,466 species including Vanessa, Bombylius, Apis, Danaus, etc.) was already on the server but had never been used past the exploratory phase. This was discovered by cross-referencing with Dan's SDM species manifest, which showed only 100 overlap with the wrong file vs. 1,277 with the correct one. All reported results use the corrected file. See the Stage Summary document for the full correction log.

### Interactions

| Source | Records | Positive pairs (shared universe) |
|---|---|---|
| GloBI (CONUS, 7 broad pollination interaction types) | 715,215 | 139 |

- 139 positive pairs after intersecting GloBI with plant and pollinator species coverage across all five models
- No true negatives — GloBI documents known interactions, not confirmed non-interactions
- Negative pairs sampled at 3:1 ratio using `np.random.default_rng(seed)` (seeds [42, 0, 1, 2, 3])
- 80/20 stratified train/test split, shared across all model variants

### PPE Opportunity Surface

| Source | Files | Species |
|---|---|---|
| Dan Cher (dcher95/phenofield PPE outputs) | 6,697 parquet files | 6,697 plant species |

Climate-driven flowering probability per species per 0.5° bin per week. Used to build f_curves and the spatiotemporal plant embeddings V_δ (4D and 15D).

### SDM Activity Curves (Preliminary)

| Source | Species | Positive pairs (SDM universe) |
|---|---|---|
| Dan Cher's species distribution models | 1,615 (1,275 after intersection with Vp) | 122 |

Model-predicted pollinator activity curves replacing GBIF-derived a_curves. Currently a small test set. Used for the SDM comparison results reported in `evaluation/03_sdm_comparison.ipynb`. See `docs/06-sdm-integration.md` for full details.

---

## Tier 3 — Pending Data

### Full SDM Activity Curves (25,466 species)

Dan is building SDM activity curves for the full 25,466-species pollinator list, matching the coverage of `pollinator_observations_v2.csv`. Once this arrives:

- The SDM shared pair universe will expand from 122 to ~139 positive pairs
- The GBIF vs. SDM comparison will become the primary ablation in the paper
- `evaluation/03_sdm_comparison.ipynb` needs to be rerun with the full SDM a_curves
- 5-seed results (mean ± std) need to be recomputed and the results table updated

This is the first task to complete when the semester resumes. See `docs/06-sdm-integration.md`.

---

## Key Data Engineering Notes

**Bin format:** All spatial bins use `"lat_lon"` string format, e.g. `"34.5_-120.0"`, at 0.5° resolution. F and P matrices must share the same column convention before computing N (shared bin count).

**N computation:** Always use `F_common @ P_common` where both matrices are restricted to the 3,162 common CONUS bins. Never use the full F and P arrays directly — mismatched column counts cause silent errors.

**Week formula:** `week = (doy - 1) // 7`, clipped to [0, 51]. Do not use `doy // 7`.

**Shared provenance warning:** GloBI interaction records, GBIF pollinator occurrences, and PhenoField/PPE training data are all largely iNaturalist-sourced. This means the model may partially learn observation-density patterns rather than genuine ecological signal. The SDM comparison is partly motivated by this concern — SDM-derived activity curves are independent of iNaturalist observation effort.
