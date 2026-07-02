# PhenoField Opportunity

Using the released PhenoField model (**PPE-L**, `FVA_v4_b2_supcon_cc_k10_bioclip_q_PPE_L`)
as a phenological-opportunity term for plant flowering across space and time, evaluated
SDM-style (presence-background). Supersedes the earlier "Part 2" deployed-probe pipeline;
see [`FINDINGS.md`](FINDINGS.md) for the full arc and results.

## What's here

| file | role |
|---|---|
| `common.py` | shared helpers: `part1_hist`, `overlap` (Schoener's D), `eff_weeks`, `coord_enc`, `fourier_week` |
| `build_prism_weekly.py` | Stage A — per-(cell, week) multi-year-mean PRISM climate windows → `prism_weekly.npz` |
| `build_prism_weekly_peryear.py` | per-(cell, week, **year**) windows → `prism_weekly_peryear.npz` (for the interannual eval) |
| `extract_ppel_field.py` | **the model** — PPE-L image-free field features + supervised head on the grid → `grid_ppel_features_<cells>.npz` |
| `run_covariate_eval.py` | **main eval** — SINR presence-background covariate ablation (coords / week / GDD / climplicit / PPE) with spatial-block hold-out, data-efficiency sweep, and multi-seed CIs |
| `run_interannual.py` | temporal-transfer eval — train ≤2021, test 2022–24; year-aware (PPE, GDD) vs year-invariant (week, climplicit) |
| `run_sinr_curves.py` | flowering-curve figures — presence-background curves (`data` / `ppe_temporal` / `data_ppe`) vs observed |
| `generate_opportunity.py` | **deliverable** — trains one multi-species SINR on Pheno3M flowering presence and writes the plug-in opportunity surface for all flowering species |
| `slurm/` | SLURM runners (hardcoded cluster paths; `gpu_a100` default, resubmit on `gpu`/`bgxd-tgirails` if full) |
| `outputs/` | metrics CSVs + figures |

## Run order

```bash
PY=/u/cherd/miniconda3/envs/pheno/bin/python
export PPE_OUT_DIR=../Part2_PhenoField_Outputs           # bulk data/features live here
$PY build_prism_weekly.py                                 # + build_prism_weekly_peryear.py --cells used
$PY extract_ppel_field.py --cells all                     # PPE-L grid features (GPU)
$PY run_covariate_eval.py --feat $PPE_OUT_DIR/data/grid_ppel_features_all.npz --tag ppel_ci --seeds 0,1,2,3,4
$PY run_interannual.py
$PY run_sinr_curves.py
# or: sbatch slurm/<stage>.sbatch
```

## Key result (see FINDINGS.md)

On held-out cells, PPE-L is a better flowering covariate than a seasonal clock, GDD, and
**climplicit** (the SOTA climate-location embedding), and its advantage **grows as occurrence
data thins** (+ppe − climplicit AP: +0.089±0.021 at full data → +0.151±0.006 at 3%). PPE
provides a **spatial, data-efficient** opportunity term; it does **not** add interannual signal
(a climatology does as well on held-out years).

## Deliverable — flowering-opportunity surface (for downstream interaction / temporal-SDM work)

`generate_opportunity.py` (→ `sbatch slurm/gen_opportunity.sbatch`) writes a partitioned parquet
to `$PPE_OUT_DIR/data/opportunity_surface/` (one file per species, ~8.9 GB total): **6,697
Pheno3M flowering species × 98,817 covered 0.5° cell-weeks ≈ 662 M rows.**

| column | meaning |
|---|---|
| `species`, `species_id` | Pheno3M plant |
| `cell_idx`, `centroid_lat`, `centroid_lon` | 0.5° grid cell |
| `week` (0–51), `doy` | time |
| `p_flowering` | SINR opportunity score (sharp, PPE-L-informed) |
| `norm` | `p_flowering` normalized to sum 1 per (species, cell) — the weekly timing distribution |

It is a **climatology** and a **relative** score (use `norm` for timing overlap); sharpest where
occurrence records exist, PPE-carried where they're sparse.

## External references

- PhenoField / PPE-L model: `/projects/bdbl/cherd/PhenoField` (checkpoint `checkpoints/FVA_v4_b2_supcon_cc_k10_bioclip_q_PPE_L/last.ckpt`)
- SINR (presence-background SDM): Cole et al., ICML 2023 — https://arxiv.org/abs/2306.02564 ; code `/projects/bdbl/cherd/sinr`
- Climplicit location embedding: `Jobedo/climplicit` (via `rshf`)
