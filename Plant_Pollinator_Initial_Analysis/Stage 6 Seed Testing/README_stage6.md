# Stage 6 — Seed Testing

## Overview

Stage 6 validates the stability of ANTHEIA's link prediction results across multiple random seeds. All models from Stages 4–5 are retrained and evaluated under 5 independent random seeds to report mean ± std rather than single-run numbers.

This stage also corrects two data coverage issues present in the original Stage 4/5 training:

1. **Activity curves now include eBird data.** Original Stage 4 training used only GBIF insect observations (`gbif_0007192`) for pollinator activity curves. Stage 6 combines insects and eBird (`gbif_0007204`, ~543M records) for complete pollinator temporal coverage.

2. **Flowering curves now cover all 6,697 plant species.** Original Stage 4 used `flowering_curves_used.parquet` (50 species only). Stage 6 builds per-species 52-week curves directly from the full PPE opportunity surface (`part_*.parquet`, 6,697 files).

## Models

| Model | Feature Vector | Dim |
|-------|---------------|-----|
| A2 | binary Vf + Vp + N | 31D |
| A3 | binary Vf + Vp + N + Δ scalar | 32D |
| A' | binary Vf + Vp + N + V_δ 4D | 35D |
| A* | binary Vf + Vp + N + V_δ 15D | 46D |
| B  | Vf_prob (PMf) + Vp + N | 31D |

## Seeds

`[42, 0, 1, 2, 3]` — seed 42 matches the original Stage 4/5 runs and serves as a sanity check. Both negative pair sampling and train/test split use the same seed per run.

## Results

| Model | ROC-AUC | PR-AUC |
|-------|---------|--------|
| A2 | 0.9373 ± 0.0080 | 0.8595 ± 0.0175 |
| A3 | 0.9463 ± 0.0046 | 0.8783 ± 0.0091 |
| A' | 0.9397 ± 0.0051 | 0.8675 ± 0.0087 |
| A* | **0.9489 ± 0.0054** | **0.8805 ± 0.0108** |
| B  | 0.9429 ± 0.0055 | 0.8696 ± 0.0101 |

A* achieves the best mean performance on both metrics. Results are stable across seeds (low std), confirming the findings are not seed-dependent.

## Files

```
Stage 6 Seed Testing/
├── Stage 6 Seed Testing.ipynb       — interactive notebook with all cells
├── stage6_seed_testing.py           — standalone reproducible script
├── stage6_all_seed_results.csv      — per-seed ROC-AUC and PR-AUC (25 rows)
├── stage6_summary.csv               — mean ± std per model (5 rows)
└── results_table_v6.png             — styled poster table (mean values)
```

## Data Dependencies

| File | Source |
|------|--------|
| `stage4_F_existence_phenofield.csv` | Stage 4 |
| `stage4_P_existence_gbif_combined.csv` | Stage 4 |
| `stage4_Vf_phenofield.csv` | Stage 4 |
| `stage4_Vp_gbif.csv` | Stage 4 |
| `stage4_globi_conus_broad.csv` | Stage 4 |
| `stage5_Vdelta_ppe.csv` | Stage 5.1 (A') |
| `stage5_Vdelta_15d.csv` | Stage 5.4 (A*) |
| `stage5_Vf_prob.csv` | Stage 5.2 (B) |
| `ppe-outputs/opportunity_surface/part_*.parquet` | PPE (Dan) |
| `gbif_0007192_observations_v2.csv` | GBIF insects |
| `gbif_0007204_observations_v2.csv` | eBird |
