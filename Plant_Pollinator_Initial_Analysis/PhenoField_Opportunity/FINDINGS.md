# Findings — PhenoField opportunity

The experimental arc, so the archived diagnostics' conclusions aren't lost. Evaluations use
presence-background SDM framing on 0.5° grid cells × 50 plant species; held-out metric is
average precision (AP) ranking observed-flowering (cell, week, species) bins vs background.

## 1. The deployed probe readout is "muted" — and it's not fixable downstream

The original pipeline (PPE field features on a **climatological grid** → flowering probe →
weekly curve) produces broad, near-flat curves (effective ≈37–52 weeks vs observed ≈7–11).
The muting survived **every** intervention, so it is intrinsic to reading a 365-day trailing
climate window on a climatological grid, not a probe/label/backbone artifact:

| intervention | result |
|---|---|
| recalibration (NPN probe vs iNat 4-class) | helped on per-visit inputs, no grid effect |
| per-year "average late" vs "average early" | no effect (single-year curves already flat) |
| backbone swap e68 → PPE-L | no effect; PPE-L's own supervised head is flatter |
| shorter climate window (truncation) | flatter, not sharper |
| probe capacity / more training data | flowering AP saturated (~0.90) |
| binary vs 4-class probe target | ~2 eff-weeks, still flat |
| z_dynamic-only probe (drop 576 static dims) | no help, lower AP |

(These conclusions came from a series of diagnostic experiments — probe recalibration, per-year
averaging, backbone swap, window truncation, probe capacity/target/feature-subset — summarized
here; the exploratory scripts are not retained in the clean repo.)

## 2. The SINR presence-background reframe fixes the readout

Recasting flowering as an SDM (positives = observed flowering bins; assume-negative background
across the full year) yields sharp, correctly-timed opportunity curves (eff ≈21, peak-week
corr with observed ≈0.95). This is a real improvement over the deployed probe, but it is
**PPE-agnostic** — a clock + coords + species embedding gets there too. (`run_sinr_curves.py`.)

## 3. PPE-L is a better, data-efficient *spatial* covariate (the main result)

Covariate ablation on spatial-block hold-out, 5 seeds (`run_covariate_eval.py`). Held-out AP:

| train fraction | +week (clock) | +gdd | +climplicit | **+ppe** |
|---|---|---|---|---|
| 1.00 | 0.824 | 0.698 | 0.762 | **0.851** |
| 0.10 | 0.624 | 0.494 | 0.628 | **0.736** |
| 0.03 | 0.500 | 0.358 | 0.499 | **0.651** |

PPE beats the clock, GDD, and **climplicit** (SOTA climate embedding) at every data level, and
the gap **grows as data thins** (paired +ppe − climplicit: +0.089±0.021 at full → +0.151±0.006
at 3%; all far outside noise). This is the foundation-model / "predict where records are sparse"
signature: PPE matters most exactly where occurrence data is thin.

## 4. Honest bound: no interannual signal

Temporal-transfer hold-out (train ≤2021, test 2022–24; `run_interannual.py`). Year-aware
covariates (PPE, GDD) do **not** beat year-invariant ones (clock, climplicit) on held-out years;
`+climplicit` (a climatology) is the best arm (AP 0.822 vs +ppe 0.808). Phenology is
near-stationary year-to-year and PPE's temporal channel carries no interannual signal. So PPE
provides a **climatological, within-year** opportunity term — not year-specific prediction.

## Scope / caveats

- Backbone: **PPE-L** (`FVA_v4_b2_supcon_cc_k10_bioclip_q_PPE_L`), the released model. e68 was an
  earlier experimental InfoNCE ablation (dropped; PPE-L was ≥ e68 everywhere it mattered).
- Downstream plant–pollinator co-occurrence eval is **not done** — the pollinator observations
  (Part-1 Step 3) were on an inaccessible scratch path; GBIF re-download pending.
- Spatial hold-out is block-based; data-efficiency subsamples occurrences globally (not a
  per-rare-species cut).
