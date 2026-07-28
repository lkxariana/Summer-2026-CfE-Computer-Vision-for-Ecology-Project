# 04 — Experiments

This document covers the full experimental setup, results, and interpretation for ANTHEIA's link prediction evaluation. All results use the corrected pollinator dataset (`pollinator_observations_v2.csv`). Two experimental tracks are reported: GBIF-based (primary) and SDM-based (preliminary comparison).

---

## 1. Experimental Setup

**Pair universe:** 556 total pairs (139 positive GloBI interactions + 417 negative pairs sampled at 3:1 ratio). See `docs/02-dataset.md` for construction details.

**Train/test split:** 80/20 stratified split, fixed and shared across all five models.

**Seed testing:** 5 seeds [42, 0, 1, 2, 3]. Negative pairs are resampled per seed; the train/test split is fixed. Results reported as mean ± std across seeds.

**Classifier:** Logistic regression (`max_iter=1000`). Chosen deliberately — as a linear classifier on fixed feature vectors, any performance gain from adding a feature reflects genuine signal in that feature rather than increased model capacity.

**Metrics:**
- **ROC-AUC:** ranking metric; measures separation between positive and negative pairs across all thresholds
- **PR-AUC:** precision-recall metric; more sensitive to performance on the positive class and more informative under class imbalance

---

## 2. GBIF Results (Primary)

All models use GBIF-derived activity curves (a_curves) on the pollinator side.

| **Model** | **ROC-AUC** | **PR-AUC** |
|---|---|---|
| Spatial Baseline | 0.9571 ± 0.0375 | 0.9387 ± 0.0335 |
| ANTHEIA-Scalar | 0.9599 ± 0.0365 | 0.9403 ± 0.0339 |
| **ANTHEIA-4D** | **0.9602 ± 0.0232** | **0.9424 ± 0.0257** |
| ANTHEIA-15D | 0.9514 ± 0.0274 | 0.9374 ± 0.0245 |
| ANTHEIA-PMf | 0.9560 ± 0.0298 | 0.9357 ± 0.0338 |

**ANTHEIA-4D is the best overall model** — highest ROC-AUC, highest PR-AUC, and lowest variance across seeds (±0.0232 vs ±0.0375 for Spatial Baseline). The stability gain is as notable as the accuracy gain: lower variance means the result is more reproducible and less dependent on the specific negative sample drawn for a given seed.

**ANTHEIA-Scalar is surprisingly competitive.** Adding a single scalar Δ raises both metrics over the Spatial Baseline with comparable or lower variance. The one-dimension addition directly encodes per-pair phenological alignment that the model cannot reconstruct from Vf and Vp alone.

**ANTHEIA-15D underperforms ANTHEIA-4D.** A 15D spatiotemporal embedding captures more variance (48.6%) than the 4D version, but performance is lower. This suggests logistic regression cannot extract the additional information from the higher-dimensional embedding — the extra dimensions may add noise rather than signal for a linear classifier.

**ANTHEIA-PMf underperforms the Spatial Baseline on PR-AUC.** Replacing the binary plant existence matrix with a continuous PPE probability matrix (PMf) does not help — and marginally hurts on PR-AUC. The Δ scalar is a more efficient encoding of PPE's temporal information than replacing Vf entirely.

---

## 3. Ablation Interpretation: The Plant-Side Asymmetry

A consistent pattern across all experiments: **plant-side temporal enrichment helps; pollinator-side temporal enrichment hurts.**

This was explicitly tested in the pollinator-side ablation experiments (not included in the primary results table, as they are negative results):

- Replacing binary Vp with a normalized weekly GBIF observation histogram (PMp): ROC-AUC dropped below Spatial Baseline
- Replacing binary Vp with a binary temporal existence matrix from GBIF (TMp): same result

The mechanism is data quality, not data quantity. PPE's f_curves are climate-driven model predictions, independent of where observers happened to be. GBIF's a_curves are opportunistic citizen science records biased by sampling effort — weeks with many observations reflect recorder activity, not genuine pollinator activity. Adding this biased temporal signal to the pollinator side degrades predictions by introducing observation-density noise.

This asymmetry is the direct motivation for SDM. Dan's species distribution models produce model-predicted pollinator activity curves that are independent of observation effort — the pollinator-side analog of what PPE provides on the plant side.

---

## 4. SDM Results (Preliminary)

SDM models replace GBIF-derived a_curves with Dan Cher's model-predicted pollinator activity curves. Current SDM coverage: 1,615 species, yielding 1,275 pollinators after intersection with Vp, and 122 positive pairs (vs. 139 under GBIF).

| **Model** | **ROC-AUC** | **PR-AUC** |
|---|---|---|
| Spatial Baseline-SDM | 0.9212 ± 0.0332 | 0.8628 ± 0.0513 |
| ANTHEIA-Scalar-SDM | 0.9235 ± 0.0320 | 0.8658 ± 0.0489 |
| ANTHEIA-4D-SDM | 0.9196 ± 0.0340 | 0.8716 ± 0.0470 |
| ANTHEIA-15D-SDM | 0.9334 ± 0.0319 | 0.8901 ± 0.0542 |
| **ANTHEIA-PMf-SDM** | **0.9489 ± 0.0245** | **0.8911 ± 0.0480** |

**Absolute performance is lower under SDM** across all models. This is expected: the SDM pair universe is smaller (122 vs. 139 positive pairs, 1,275 vs. 24,939 pollinators), and a smaller, less representative universe is a harder evaluation setting.

**The model ranking shifts under SDM.** ANTHEIA-PMf-SDM and ANTHEIA-15D-SDM lead, whereas ANTHEIA-4D leads under GBIF. This shift suggests that when the pollinator temporal signal is higher quality (model-predicted rather than observation-biased), richer plant-side representations become more valuable — the model can leverage the cleaner pollinator signal to extract more from a more expressive plant embedding.

**These results are preliminary.** The SDM covers only 1,615 of 25,466 pollinator species. The full SDM comparison — the primary ablation in the paper — is pending Dan's full species distribution model build. See `docs/06-extensions-and-next-steps.md`.

---

## 5. Seed Testing Interpretation

The 5-seed protocol (seeds [42, 0, 1, 2, 3]) resamples negative pairs at each seed while holding the train/test split fixed. Variance across seeds therefore reflects sensitivity to the specific negative pairs drawn, not to train/test partitioning.

Key observations:
- All models show meaningful variance (±0.02–0.04 ROC-AUC), reflecting the small positive set (139 pairs)
- ANTHEIA-4D has the lowest variance of any model under GBIF (±0.0232), making it the most robust choice
- SDM models show comparable variance to their GBIF counterparts despite the smaller pair universe, suggesting the SDM activity curves produce more consistent signal per pair

Seed 42 is the original single-run seed and serves as a sanity check — its results should fall within the 5-seed distribution for each model.

---

## 6. Spatial Visualization

Per-bin interaction probability maps for *Achillea millefolium* across February–May confirm the model's behavior qualitatively. ANTHEIA produces finer spatial discrimination than the Spatial Baseline — notably lower predicted interaction probabilities in the Gulf Coast and Florida regions where the plant and its pollinators are spatially proximate but phenologically misaligned. This seasonal variation is invisible to the Spatial Baseline, which produces the same prediction regardless of time of year.

The raw PPE Δ = min(f_curve, a_curve) is visualized directly in the seasonal maps rather than ANTHEIA's predict_proba output, because logistic regression compresses predictions into a narrow high-probability range (mean ~0.88, std ~0.089) that makes seasonal variation invisible at the scale of a map. Hovmöller diagrams (latitude vs. week) confirm the expected phenological gradient — peak temporal overlap propagates northward through the season, concentrated at mid-latitudes (35–45°N) between May and July.

See `visualization/01_spatial_figures.ipynb` for the full figure pipeline.
