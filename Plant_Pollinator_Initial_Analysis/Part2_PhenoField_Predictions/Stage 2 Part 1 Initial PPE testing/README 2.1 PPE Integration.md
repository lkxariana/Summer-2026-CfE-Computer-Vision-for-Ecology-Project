# Stage 2, Step 1 — PPE Predictions Initial Pipeline

Integrates Dan's PhenoField Predicted Encoder (PPE) flowering curves
(`Part2_PhenoField_Predictions/`) into the temporal overlap analysis from
**Stage 1, Step 8** (Jaccard-pair temporal overlap), replacing the
observation-based plant flowering histogram with a climate-only,
model-predicted flowering curve. Produces a second, PPE-based version of
`step8_edge_overlap_summary.csv` and compares it directly against the
original.

## Motivation

Step 8's plant-side temporal density comes from real flowering observations,
filtered to `MIN_OBS=5` per (species, 1.5° bin). This means coverage is
gated by *where people happened to observe and report flowering*, the same
density bias Step 7 (Jaccard Range Overlap) was designed to escape on the
spatial side. PPE replaces the plant side with a model-predicted weekly
flowering probability, generated from climate data alone, available
anywhere PRISM has coverage — independent of observation density. This step
tests whether using PPE's plant curves changes Step 8's temporal overlap
results, and if so, how and why.

## Data Inputs

| File | Description |
|------|-------------|
| `ppe-outputs/data/flowering_curves_all.parquet` | PPE output: one row per (0.5° cell, species, week), columns `cell_idx, centroid_lat, centroid_lon, species, week, doy, inat_p_*, inat_norm`. Covers 3,335/6,136 grid cells, 50 plant species. `inat_norm` is the per-(cell, species) weekly flowering probability normalized to sum to 1 — the PPE analog of Step 8's `weekly_density()` output. |
| `ppe-outputs/data/coverage.csv` | Per-0.5°-cell climate-window coverage: `cell_idx, centroid_lat, centroid_lon, used_in_analysis, n_weeks_present, n_obs, missing_weeks`. Coverage is computed at the cell level (pooled across all species), not per species. |
| `top100_jaccard_pairs.csv` | Step 7 output: 100 plant-pollinator pairs ranked by Jaccard range-overlap score. Columns include `rank, jaccard, shared_bins, species, pollinator_species, hotspot_lat, hotspot_lon`. Note: `shared_bins` is a *count* only — the actual bin list is not saved and must be reconstructed (see Pipeline, step 2). |
| `cache_plant_counts.parquet`, `cache_pollinator_counts.parquet` | Step 7 caches: `(species, lat_bin, lon_bin) -> count` at 0.5° resolution, reused here to reconstruct each pair's full shared-bin set. |
| `pollinator_observations_v2.csv` | Same file used in Step 8: `pollinator_species, lat, lon, doy, year`. Pollinator side is unchanged from Step 8. |
| `step8_edge_overlap_summary.csv` | Original (observation-based) Step 8 result, used as the comparison baseline. |

All data is stored on the crow server under `/scratch/ariana.l/`, with PPE
outputs specifically under `/scratch/ariana.l/ppe-outputs/data/` (synced
from Box via `rclone`). This step's own outputs are written to a separate
sibling folder, `/scratch/ariana.l/Part2_PhenoField_Predictions/step1_ppe_overlap_outputs/`
— kept distinct from Dan's `Part2_PhenoField_Predictions/outputs/` (his
Box-synced PPE output folder, containing `data/`, `figures/`, `logs/`) to
avoid any risk of collision on the next `rclone` sync.

## Parameters

```python
BIN_SIZE_TEMPORAL = 1.5   # matches Step 8's resolution
N_WEEKS = 52
MIN_OBS = 5               # pollinator-side filter, unchanged from Step 8
CORNER_TO_CENTER_OFFSET = 0.25  # see step 3 below
```

## Pipeline

### 1. Load and validate PPE coverage
Load `flowering_curves_all.parquet` and `coverage.csv`. Confirm `inat_norm`
(not `inat_p_flowering`, which is a raw, non-normalized weekly probability)
is the correct plant-side density column — it is the only column that
behaves as a true 52-week timing distribution. Spot-checked two species
(*Malosma laurina*, *Larrea tridentata*) at their respective hotspot cells
to confirm predicted curves show real, biologically plausible seasonal
structure rather than degenerate/flat output.

### 2. Reconstruct Step 7's full shared-bin sets
`top100_jaccard_pairs.csv` only stores each pair's shared-bin *count*, not
the bin list itself. Reconstructed the full per-pair shared-bin sets from
`cache_plant_counts.parquet` / `cache_pollinator_counts.parquet` by
intersecting each pair's plant and pollinator bin sets at 0.5° resolution.
Verified the reconstructed counts exactly match the original `shared_bins`
column for all 100 pairs before proceeding.

### 3. Resolve the grid offset between Step 7 and PPE
Step 7's hotspot/bin coordinates use a bin-**corner** convention (`.0`/`.5`
anchored); PPE's grid centroids use a bin-**center** convention (`.25`/`.75`
anchored) at the same 0.5° resolution. Naive nearest-centroid matching
produced a uniform, suspicious distance (0.3536° = the exact diagonal of a
0.25°×0.25° offset) for every single hotspot — a four-way tie, not a real
match. Resolved by adding `+0.25` to both lat and lon before matching
exactly against PPE centroids. After correction, **100/100** Jaccard pairs'
shared bins matched a real PPE cell.

A first coverage check that only matched on `(cell)` returned a suspicious
100% coverage; this was a bug, not a real result — it ignored species
entirely, so any cell with *any* species' PPE prediction counted as
"covered" for every pair. Corrected to match on `(cell, species)` jointly;
true species-aware coverage is still 100/100, since PPE generates
predictions for all 50 species uniformly wherever climate data exists.

### 4. Handle inconsistent per-species week coverage
Individual (species, 0.5° cell) PPE curves have inconsistent week-of-year
coverage — ranging from 2 to 52 weeks present out of 166,750 (species, cell)
combinations (only 11.7% have full 52-week coverage). A naive per-week
average across cells when aggregating to 1.5° bins breaks the sum-to-1
normalization (mean curve sum ≈1.61 instead of 1.0), because different weeks
are averaged over different, inconsistent denominators.

Confirmed with Dan (PPE's author) that missing weeks represent genuine
species absence at that location/time, not missing data — consistent with
how the underlying training observations were collected. Based on this,
missing (species, cell, week) combinations are explicitly zero-filled
before aggregating to 1.5° bins. After this fix, both individual
(species, cell) curves and aggregated (species, 1.5°-bin) curves sum to
~1.0 (std ~1e-8, floating-point noise only).

Separately checked `coverage.csv` and `build_prism_weekly.py`'s logic:
climate-window coverage gaps occur at the **cell level**, pooled across all
species — so the per-species week differences seen in
`flowering_curves_all.parquet` must arise downstream (in the probe-training
or curve-combination stage), not from the climate data itself. Flagged as
an open question for Dan; does not block this analysis.

### 5. Aggregate PPE curves from 0.5° to 1.5°
Average zero-filled `inat_norm` across all 0.5° cells inside each 1.5° bin,
per species per week.

### 6. Compute PPE-based temporal overlap
Pollinator side is **identical to Step 8**: `add_bins()` at 1.5°, `MIN_OBS=5`
filter, `weekly_density()` from real observations. Plant side uses the
zero-filled, 1.5°-aggregated PPE curves from step 5 instead of observed
plant data. Overlap formula is unchanged from Step 5/8:
```
overlap = sum(min(p_density_ppe, q_density))
```

### 7. Per-pair summary with a PPE-appropriate weight
Step 8's original weighted mean used `min(n_obs_plant, n_obs_pollinator)` as
the per-bin weight. This has no PPE analog, since the plant side is no
longer an observation-density quantity. Three options were considered
(pollinator-only weight, unweighted mean, PPE-coverage-based weight); chose
to weight by `n_weeks_present` from `coverage.csv` (averaged to 1.5° bins),
as the closest analog to "trust this bin more if more real climate signal
underlies its prediction."

### 8. Compare against the original Step 8 result
Merged the PPE-based summary against `step8_edge_overlap_summary.csv` on
`(species, pollinator_species)` and compared bin counts, weighted-mean
overlap values, correlation, and the largest disagreements.

## Results

- **100/100 edges** scored under PPE (same as the original), with **9,952
  total (edge, bin) rows** — more rows per edge on average than the
  original (99.5 vs. 75.6), because PPE's plant side is not gated by a
  `MIN_OBS` observation threshold the way the original method's plant side
  was.
- **Correlation between PPE and original weighted-mean overlap: r = 0.74**
  — the relative ranking of pairs is broadly preserved.
- **PPE systematically compresses the distribution of overlap values**
  toward a moderate middle band (~0.25–0.5). Pairs with very high original
  overlap (e.g. 0.766, 0.638) are pulled down under PPE; pairs with very low
  original overlap (e.g. 0.018, 0.032) are pulled up. The original
  distribution has a sharp near-zero spike plus a long high-value tail; the
  PPE distribution is a single, narrower, centrally-peaked hump with neither
  tail (see `ppe_vs_original_distribution.png`).
- **Mechanism, illustrated by *Aquilegia canadensis* / *Polygonia comma***
  (original = 0.217, PPE = 0.509): the pollinator's real activity curve is
  bimodal, with a genuine low point between two peaks. The original plant
  histogram's single sharp peak happened to fall exactly in that low point
  — an artifact of sparse-observation timing, not necessarily a true
  mismatch — producing an artificially low original score. PPE's much
  wider, never-zero curve overlaps with both of the pollinator's peaks
  simultaneously, producing a higher score. This generalizes: PPE's
  smoothness tends to inflate apparent overlap for any pair whose sparse
  observed peak happens to be badly timed, rather than reflecting a more
  accurate measurement (see `ppe_vs_original_disagreement_grid.png`).
- **`overlap_min = 0.000000` appears for several pairs** under PPE. This is
  attributable to sparse pollinator sampling at `MIN_OBS=5` (some bins
  barely clear the threshold, clustering their few observations into a
  handful of weeks), not confirmed temporal non-overlap — consistent with
  the broader point that absence of observation in any single week is never
  proof of true absence.

## Output Files

| File | Description |
|------|-------------|
| `edge_summary_ppe.csv` | Per-edge overlap statistics (mean, weighted mean by `n_weeks_present`, std, min, max, bin count) using PPE plant curves, for Step 7's 100 pairs |
| `ppe_vs_original_comparison.csv` | Merged comparison of PPE-based vs. original Step 8 results, per pair |
| `ppe_vs_original_scatter.png` | Scatter plot, PPE vs. original weighted-mean overlap, colored by PPE bin count |
| `ppe_vs_original_distribution.png` | Histogram + KDE comparing the two methods' overlap-value distribution shapes |
| `ppe_vs_original_disagreement_grid.png` | 9-panel KDE grid of the largest-disagreement pairs: observed plant (dashed blue), PPE plant (dotted red, `bw_method=0.08`), pollinator (orange) |

## File Structure

```
stage2_step1_ppe_pipeline/
├── README.md
├── 01_load_and_validate_ppe_coverage.py     # Load PPE data, reconstruct shared bins, fix grid offset, validate coverage
├── 02_zero_fill_and_aggregate_1p5.py        # Zero-fill missing weeks, aggregate 0.5° -> 1.5°
├── 03_compute_ppe_overlap.py                # Pollinator side (Step 8 logic, unchanged) + PPE plant side -> overlap, per-edge summary
├── 04_compare_to_original.py                # Merge against step8_edge_overlap_summary.csv, compute comparison stats
└── 05_visualize_comparison.py               # Scatter, histogram/KDE, and disagreement-grid figures
```

## Usage

Run on the crow server, in sequence:

```bash
python 01_load_and_validate_ppe_coverage.py
python 02_zero_fill_and_aggregate_1p5.py
python 03_compute_ppe_overlap.py
python 04_compare_to_original.py
python 05_visualize_comparison.py
```

Or run cell-by-cell in Jupyter — this is how the analysis was originally
developed (`2.1 PPE Integration.ipynb`).

## Dependencies

- `pandas`, `numpy`
- `scipy` (`gaussian_kde`)
- `matplotlib >= 3.10`

## Known Limitations

- PPE's plant curves are a multi-year climatological average, not a
  per-year prediction — interannual variation is intentionally dropped,
  same caveat as in Dan's Part 2 README.
- The per-species week-coverage inconsistency within a single PPE cell
  (Step 4 above) is not yet fully explained at the mechanism level; the
  zero-fill treatment is based on Dan's stated modeling assumption
  (missing = absent), which is itself inherited from how PPE's training
  data was collected and has not been independently verified beyond that.
- The `n_weeks_present`-based weighting (step 7 above) is a judgment call,
  not a direct port of Step 8's original weighting logic — there is no
  PPE-side equivalent to "plant observation count," so this is the closest
  available analog rather than an equivalent quantity.
- PPE's systematic compression of overlap values toward a moderate range
  means it should not be treated as a strict improvement over the original
  method — it trades the ability to detect true near-zero or very-high
  overlap for broader spatial coverage. Both versions are retained
  side-by-side for this reason; PPE is not used as a full replacement for
  the original observation-based result.
- As with Step 8, `overlap_min = 0` in some PPE-based bins reflects sparse
  pollinator sampling, not confirmed non-overlap.
