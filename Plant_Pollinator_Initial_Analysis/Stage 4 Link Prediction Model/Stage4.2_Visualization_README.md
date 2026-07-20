# Stage 4 — ANTHEIA Visualization

## Overview

This module produces the combined visualization figure comparing the ANTHEIA A2 (spatial co-occurrence baseline) and A3 (spatial + PPE temporal overlap) link prediction models for a focal plant species, *Achillea millefolium*, across CONUS.

The figure consists of three rows:
- **Row 1:** Spatial maps — Ground Truth (GloBI), A2 (static), and PPE Δ across four seasons (Spring/Summer/Fall/Winter)
- **Row 2:** A2 latitude profile (static bar chart) + PPE Δ Hovmoller diagram (latitude × week)
- **Row 3:** Mean A3 − A2 difference map (full year average, diverging colormap)

---

## Dependencies

```
numpy
pandas
matplotlib
scikit-learn
scipy
pickle
glob
```

All data lives under `/scratch/ariana.l/Stage 4 Link Prediction Model/` and `/scratch/ariana.l/ppe-outputs/`.

---

## Input Files

| File | Description |
|------|-------------|
| `stage4_F_existence_phenofield.csv` | Plant existence matrix (6,466 species × 3,162 bins), built from PhenoField |
| `stage4_P_existence_gbif_combined.csv` | Pollinator existence matrix (4,515 species × 3,895 bins), built from GBIF |
| `stage4_Vf_phenofield.csv` | Plant PCA embeddings (15D) |
| `stage4_Vp_gbif.csv` | Pollinator PCA embeddings (15D) |
| `stage4_A2_logistic.pkl` | Trained A2 logistic regression model (31D feature: Vf + Vp + N) |
| `stage4_A3_logistic.pkl` | Trained A3 logistic regression model (32D feature: Vf + Vp + N + Δ) |
| `stage4_globi_conus_broad.csv` | GloBI interactions filtered to CONUS |
| `/ppe-outputs/opportunity_surface/part_*.parquet` | PPE opportunity surface (~6,000 parquet files) |
| `/Plant Pollinator Initial Analysis/gbif_0007192_observations_v2.csv` | GBIF pollinator observations |

---

## Output Files

| File | Description |
|------|-------------|
| `pred_df_weeks1_52.parquet` | Per-bin, per-week A2 and A3 predictions (60,268 rows × 5 cols) |
| `delta_df_weeks1_52.parquet` | Per-bin, per-week PPE Δ = min(flowering, activity) (60,268 rows × 4 cols) |
| `antheia_combined_figure_v4.png` | Final combined visualization figure |

---

## Pipeline Steps

### Step 1 — Load all data
Run `01_load_data.py`. Loads existence matrices, PCA embeddings, trained models, rebuilds flowering curves (averaged `norm` across all PPE parquet files) and pollinator activity curves (weekly GBIF observation counts, normalized to sum=1).

### Step 2 — Align F and P to common bins
F and P were built from different sources (PhenoField vs GBIF) and have different bin columns (3,162 vs 3,895). Align to 3,160 common bins before computing N (shared bin count feature).

### Step 3 — Compute pred_df (full year)
Run `02_compute_predictions.py`. For each of 1,159 spatial bins in *Achillea millefolium*'s range, and for each of 52 weeks:
- A2 prediction: cached once per bin (week-independent), feature = [Vf, Vp, N]
- A3 prediction: computed per week, feature = [Vf, Vp, N, Δ] where Δ = min(f_w, a_w)
- Both are averaged over all pollinators present in the bin

### Step 4 — Compute delta_df (full year)
Run `03_compute_delta.py`. For each bin × week, compute Δ = min(flowering curve value, mean activity curve value across bin's pollinators). Does not call the classifier — pure temporal overlap signal.

### Step 5 — Generate figure
Run `04_visualize.py`. Produces the combined three-row figure and saves to `antheia_combined_figure_v4.png`.

---

## Notes

- **F and P bin alignment:** F has 3,162 bins, P has 3,895 bins. Only 3,160 are common. The dot product for N must use the common-bin versions (`F_common`, `P_common`).
- **Flowering curves:** Reconstructed by averaging the `norm` column across all cells per species per week from the PPE opportunity surface. Not stored as a separate file — must be rebuilt from the parquet files on each run.
- **Activity curves:** Reconstructed from GBIF insect observations (`gbif_0007192_observations_v2.csv`) by binning observation DOY into weeks and normalizing per species. Not stored separately.
- **Why PPE Δ instead of A3 for temporal panels:** A3 predictions are compressed into a narrow high-probability range (mean 0.88, std 0.089) by the logistic regression, making seasonal variation invisible to the naked eye. Plotting Δ directly shows the raw temporal signal clearly.
