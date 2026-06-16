# Part 2 — PhenoField predicted flowering curves on the grid (iNat probe)

Replaces the **observed** plant flowering histogram (Part 1, Step 4) with
**PhenoField-predicted** weekly flowering probability, per grid cell × species,
using the image-free **field encoder** + a trained **iNat 4-class probe**.

## Code vs. data

- **Code** lives here (`Part2_PhenoField_Predictions/`) → GitHub.
- **Data, figures, logs** are written to a sibling **`Part2_PhenoField_Outputs/`**
  (`data/`, `figures/`, `logs/`) → Box.
- Override the output location with `PPE_OUT_DIR=/path/to/outputs` (default is the
  sibling dir above). Every script reads/writes there.

## Pipeline

For each 0.5° grid cell (`data/grid_centroids_0.5deg.csv`) and each of the 50
plant species (`data/plant_flowering_events.parquet`):

1. **`build_prism_weekly.py`** — for each cell, gather every training observation
   whose (lat, lon) falls *inside* the cell (PhenoField `hf_prism_365.parquet`,
   3.1 M real daily windows, 2013–2025), bin by week-of-year, and **average the
   365-day daily PRISM windows across all years** → one multi-year-mean trailing
   window per (cell, week). Weeks with no in-cell obs are **skipped + reported**
   (`coverage.csv`). → `prism_weekly.npz`
2. **`extract_field_features.py`** — `CrossModalVAE.forward_field` on the **e68**
   backbone → `cat(z_static[576], z_dynamic[192])` = **768-d** field features.
   No image, no Climplicit; day-of-year is not fed (season comes only from the
   PRISM window). AlphaEarth held at 2017 (nearest CONUS cell). NaN PRISM days
   are zeroed. → `grid_field_features_<cells>.npz`
3. **`run_inat_probe.py`** — trains a 4-class field probe (no saved artifact
   existed) on the cached e68 iNat field features
   (`phenofield_cache/e68_large_adaln_both_infonce/inat_test_*.npz`), train on
   `inat_test_random`, validate on `inat_test_spatial`. Flowering = class 2.
   **Validation flowering AP 0.90, AUC 0.88.** → `predictions_inat_<cells>.parquet`
4. **`combine_curves.py`** — adds the weekly curve **normalized to sum 1** per
   (cell, species): the Step-4 plug-in. → `flowering_curves_<cells>.parquet`
5. **`viz_flowering_curves.py`** — circular-KDE figures → `figures/`.

`<cells>` = `used` (the 20 `used_in_analysis` cells, full 52-week coverage) or
`all` (3,335 covered cells, ~4.9 M rows).

## Main output: `flowering_curves_<cells>.parquet`

One row per **(cell, week, species)**. A flowering curve = the 52 rows for one
`cell_idx` + `species`, ordered by `week`.

| column | meaning |
|---|---|
| `cell_idx, centroid_lat, centroid_lon` | 0.5° cell + centroid (location) |
| `species, week (0–51), doy` | species + time |
| `inat_p_flowering` | **weekly P(flowering)** (iNat probe, class 2) |
| `inat_p_no_phenology / budding / fruiting` | other 3 class probabilities |
| `inat_norm` | `inat_p_flowering` normalized to sum 1 over the year (timing distribution / KDE) |

## Reproduce

```bash
PY=/u/cherd/miniconda3/envs/pheno/bin/python
export PPE_OUT_DIR=../Part2_PhenoField_Outputs     # where data/figures go
$PY build_prism_weekly.py
$PY extract_field_features.py --cells used   # GPU recommended; CPU ok for `used`
$PY run_inat_probe.py        --cells used
$PY combine_curves.py        --cells used
$PY viz_flowering_curves.py  --cells used
# full grid: sbatch run_all_cells.sbatch   (gpu_a100, runs all 4 stages for --cells all)
```

`check_inat_probe_lc.py` is a diagnostic: iNat probe flowering-AP learning curve
(shows the probe saturates well below the 177k training set).

## Caveats

- **Climatology, not per-year** — one curve per (cell, species) from multi-year
  mean PRISM; interannual variation intentionally dropped.
- **Coverage** — used cells have all 52 weeks; across the full grid 3,335/6,136
  cells have ≥1 week, many with gaps (`coverage.csv`). Down-weight sparse cells.
- **~0.11% NaN** — a few (cell,week) windows have a NaN PRISM day. The extractor
  now zeroes NaNs (`torch.nan_to_num`); re-run `extract_field_features.py` to
  clear them from the saved outputs (the current `*_all.npz` predate the fix).
- **iNat probe trained on test-split features** — no e68 `inat_train` cache
  exists; fit on `inat_test_random`, validated on disjoint `inat_test_spatial`.
  The flowering AP is saturated, so the deployed predictions are unaffected.
