# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

ANTHEIA — a plant-pollinator interaction link prediction pipeline (MVRL, WashU; authors Kexing Li, Dan Cher, Nathan Jacobs). It predicts the probability that a plant and pollinator species interact, combining spatial co-occurrence embeddings with phenological (PPE) temporal features over CONUS at 0.5° resolution.

This is a **notebook-only research repo**: all code lives in Jupyter notebooks (Python 3 / ipykernel). There is no build system, no tests, no linter, no requirements file. The stack is pandas, numpy, scikit-learn, matplotlib, pyarrow, scipy.

Start with `GUIDE.md` (running order, notebook map, technical facts) and `README ANTHEIA2026.07.md` (models, results). Extended methodology lives in `docs/01-overview.md` through `docs/08-methods-literature.md` (08 = verified methods references for the 2026-08 redesign). Note there is no file literally named `README.md` even though GUIDE.md refers to one.

## Doc Names vs. Actual Filenames

GUIDE.md and docs/ refer to notebooks by idealized names. The actual files on disk differ. Mapping:

| Docs say | Actually on disk |
|---|---|
| `data/full_scale_run/` | `data/Full Scale/` (directory names contain spaces — quote paths) |
| `data/small_scale_testing/` | `data/Small Scale Testing/` |
| `data/.../01_plant_flowering.ipynb` etc. | `01_plant_flowering_full_scale.ipynb` / `..._small_scale.ipynb` suffixed variants |
| `representation/01_existence_matrices.ipynb` etc. | `rep_01_existence_full.ipynb`, `rep_02_pca_full.ipynb`, `rep_03_ppe_full.ipynb`, `rep_04_activity_full.ipynb` |
| `model/01_baseline_spatial.ipynb`, `02_antheia_models.ipynb` | `model_01_baseline.ipynb`, `model_02_antheia.ipynb` |
| `evaluation/01_seed_testing.ipynb`, `02_sdm_comparison.ipynb` (also cited as `03_sdm_comparison.ipynb`) | `eval_01_seed_testing.ipynb`, `eval_02_sdm.ipynb` |
| `visualization/01_spatial_figures.ipynb` | `viz_01_spatial_figures.ipynb` |

## Pipeline Architecture

Five stages; each depends on CSV outputs of the previous stage (see GUIDE.md "Running Order" for the per-notebook output files):

1. **`data/Full Scale/`** — builds existence matrices F (plants, from PhenoField) and P (pollinators, from GBIF `pollinator_observations_v2.csv`), GloBI interaction edgelist, Δ phenology overlap, Jaccard range overlap. `data/Small Scale Testing/` is the superseded top-50-species exploratory phase — not used in reported results.
2. **`representation/`** — PCA 15D embeddings Vf/Vp from F/P; PPE integration producing f_curves, V_δ (4D and 15D), Vf_prob.
3. **`model/`** — logistic regression link predictors. Five variants that differ only in plant-side/temporal features: Spatial Baseline [Vf,Vp,N], ANTHEIA-Scalar (+Δ), ANTHEIA-4D (+V_δ 4D), ANTHEIA-15D (+V_δ 15D), ANTHEIA-PMf (Vf_prob replaces Vf).
4. **`evaluation/`** — `eval_01_seed_testing.ipynb` is the primary 5-seed result; `eval_02_sdm.ipynb` swaps GBIF pollinator activity curves for SDM-predicted ones.
5. **`visualization/`** — spatial maps and Hovmöller figures.

To execute a notebook headlessly: `jupyter nbconvert --to notebook --execute --inplace "<path>.ipynb"` (quote paths; several directories contain spaces).

## Data Locations

No data lives in the repo (`.gitignore` excludes `*.parquet`, `*.zip`, `*.gz`, raw data dirs — keep it that way). Notebooks read and write absolute paths under `/scratch/ariana.l/` (the lab machine "crow"); they will not run elsewhere. The full path table is at the bottom of GUIDE.md. Key inputs: F/P matrices under `Stage 4.../New Stage 4...`, PPE opportunity surface at `/scratch/ariana.l/ppe-outputs/opportunity_surface/part_*.parquet`, SDM curves under `Stage 6 Seed Testing/`.

Known gaps (verified 2026-08-31):
- `f_curves_ppe.csv` is absent from `New Stage 4 Link Prediction Model/`, though seven notebooks load it from there. The `f_curves_corrected.csv` in that directory is a different artifact (6,348 intersected species) — but substituting it reproduces the published Spatial Baseline numbers exactly.
- `globiinteractions_conus.csv` (data/02's declared output) is also absent; the model/eval notebooks actually read `Stage 4 Link Prediction Model/stage4_globi_conus_broad.csv`.
- The notebooks are post-hoc reconstructions and several cannot run as written: the PPE parquet schema uses `centroid_lat`/`centroid_lon` (not `lat`/`lon`), so `data/Full Scale/01_plant_flowering_full_scale.ipynb` and rep_03's V_δ/PMf cells crash on the real files. The on-disk F matrix is observation-derived (36–191 bins/species), not PPE-surface-derived as data/01 claims (each species' PPE surface covers all 3,162 bins). Trust on-disk artifacts over notebook provenance claims.

## Package (branch antheia-package, 2026-08-31)

The pipeline is being rebuilt as `src/antheia/` (installable; see `src/antheia/README.md`) with the orientation-corrected edge list (`scripts/rebuild_edges.py` → `artifacts/edges_v1.parquet`, 62,832 pairs), a frozen degree-stratified plant split (`artifacts/split_v1.json`), and a plant-grouped eval harness (`eval/`). Run everything with `/home/cher/miniconda3/envs/donuts/bin/python3`; tests: `python3 tests/test_invariants.py`. Corrected baselines: `artifacts/baselines_v1.csv`. New work should build on this package, not the notebooks.

## Critical Data Bug (found 2026-08-31, resolved in antheia package)

The GloBI file's three interaction types (`visitsFlowersOf`, `visits`, `pollinates`) are all pollinator→plant oriented, but every notebook renames `sourceTaxonName → plant_species` without swapping. Only 159 of 715,215 records (GloBI's own reversed/dirty entries) survive the coverage intersection, producing the reported 139 positive pairs. Membership-based orientation fixing (swap when source ∈ P-species and target ∈ F-species; F∩P species overlap is 0 so this is unambiguous) yields 60,702 positive pairs within existing feature coverage. Also verified: with the current 139-pair benchmark, an N-only logistic regression scores 0.9903 ROC-AUC, beating all published models (~0.96) — the current evaluation is dominated by range-size/effort shortcuts. Do not build on the 139-pair results without discussing this with Dan Cher.

## Invariants (violating these silently corrupts results)

- **N computation:** always `F_common @ P_common` restricted to the 3,162 shared CONUS bins — never raw F/P arrays.
- **Bin format:** `"lat_lon"` strings at 0.5° resolution, e.g. `"34.5_-120.0"`. F and P must share the same column set before computing N.
- **Week formula:** `week = (doy - 1) // 7`, clipped to [0, 51]. Not `doy // 7`.
- **Randomness:** `np.random.default_rng(seed)`, never `np.random.seed()`. Seeds are `[42, 0, 1, 2, 3]`. Negative pairs resampled per seed at 3:1; the 80/20 stratified split is fixed across models.
- **f_curves:** build from the full PPE opportunity surface parquets, never from `flowering_curves_used.parquet` (50 species only).
- **SDM coordinates:** subtract 0.25 from both lat and lon (`lat_bin = centroid_lat - 0.25`) to match the F/P bin convention.
- **Never mix GBIF and SDM a_curves within a single run.** Results using the pre-correction pollinator file (anything other than `pollinator_observations_v2.csv`) are discarded.

## Current State (as of Aug 2026)

Primary GBIF results are final; SDM results are preliminary (1,615 of 25,466 pollinator species). The queued next step: once Dan Cher's full SDM activity curves arrive, rerun `evaluation/eval_02_sdm.ipynb`, recompute the shared pair universe, and update results tables in `README ANTHEIA2026.07.md`, `docs/01-overview.md`, and `docs/04-experiments.md`. Details in `docs/06-extensions-and-next-steps.md`.

Git: work happens on branch `Clean-for-ANTHEIA` (default branch is `main`; origin is `lkxariana/Summer-2026-CfE-Computer-Vision-for-Ecology-Project`).
