# ANTHEIA: what changed, and why

A narrative summary of the 2026-08-31 → 09-01 work on branch `antheia-package`.
Full per-experiment detail: `EXPERIMENTS.md`. Code: `src/antheia/`, `eval/`, `scripts/`.

---

## Where it started

ANTHEIA predicted whether a plant and a pollinator species interact, from spatial
co-occurrence plus phenological overlap over CONUS at 0.5°.

**Setup.** 139 positive pairs from GloBI. Negatives sampled uniformly at 3:1. An 80/20
random split *over pairs*. Logistic regression on `[Vf, Vp, N]` — 15-D PCA of each species'
occupancy grid, plus `N` = shared occupied bins — with four variants adding phenology
(Δ overlap scalar, V_δ 4-D/15-D, PMf).

**Result.** ROC-AUC 0.951–0.960 across all five models; ANTHEIA-4D nominally best.

Two independent problems made that number meaningless.

---

## Problem 1: the labels were a parsing bug

GloBI's three flower-visitation types (`visitsFlowersOf`, `visits`, `pollinates`) are all
recorded **pollinator → plant**. Every notebook renamed `sourceTaxonName → plant_species`
without swapping direction. Of 715,215 CONUS records, only **159** survived the coverage
intersection — and those were GloBI's own *reversed* entries, i.e. the data's noise tail.
Deduplicated, that is the 139 "positives."

**Fix.** Orient each record by species-set membership (the plant and pollinator sets share
zero names, so it is unambiguous), plus binomial name normalization.

> **139 → 62,832 unique edges** (3,689 plants × 4,302 pollinators, 0.40% connectance).
> `scripts/rebuild_edges.py` → `artifacts/edges_v1.parquet`.

## Problem 2: the evaluation was solvable without ecology

This one is not a data bug, and it did not go away when the data was fixed. Re-run the
original *style* of evaluation — pooled pair classification, sampled negatives — on the
corrected 62,832-edge set and the models still look strong, while the nulls nearly match them:

| on the corrected data | ROC-AUC | PR-AUC | recall@10 |
|---|---|---|---|
| degree null (rank by pollinator range size) | 0.866 | 0.527 | 0.080 |
| **`N` alone — one feature** | 0.881 | 0.501 | **0.117** |
| Spatial Baseline `[Vf, Vp, N]` | 0.904 | 0.636 | 0.074 |
| ANTHEIA-Scalar (+Δ) | 0.910 | 0.652 | 0.077 |
| ANTHEIA-4D / 15-D / PMf | 0.902–0.903 | 0.633–0.635 | 0.072–0.074 |

Read the ROC column alone and the ladder looks healthy. But a null that knows nothing except
how widespread a pollinator is scores 0.866 — the metric is dominated by easy negatives, so
most of that number is not ecology. Uniform negative sampling makes it worse: positives are
well-observed widespread species, negatives are random rare ones, so popularity alone
separates the classes.

And in the column that matches the actual task, the ordering inverts — **one feature beats all
five ANTHEIA variants**: `N` at 0.117 against 0.072–0.077. That is the problem that persisted
into the corrected data and drove the rest of this work.

---

## The new setup

**Task.** Retrieval, not classification: for a plant, rank **all 24,939 pollinators**.
No eval-time negative sampling, so there is no sampling knob to bias the headline.

**Split.** Leave-plant-out (cold start) — every test plant is unseen. Frozen degree-stratified
75/10/15 (`artifacts/split_v1.json`; 553 test plants, 9,210 test positives). *Verified* that
one frozen split suffices: between-holdout std (±0.009) is smaller than the within-split
bootstrap std (±0.012), and 3-fold ≈ 5-fold ≈ repeated holdouts — so k-fold buys cost, not
information.

**Metrics.** Per-plant recall@10 / hit@10 with bootstrap-over-plants CIs; PR-AUC for pooled
comparisons; **null models (degree, range size, N-only) in every table.**

**Second axis.** GloBI records which source dataset contributed each record. 71.3% of CONUS
records are iNaturalist; 23.2% of edges carry ≥1 curated record (USGS bees, Guzmán 2022,
Web of Life, CropPol, museums). Every model is scored on **all labels** and on
**curated (Tier-1) labels** — a free, per-record measure of cross-process generalization.

---

## Why one feature beat the models

When ranking pollinators *for one plant*, every plant-side feature is constant and
cancels. The classifier had spent its capacity on pooled prevalence — "boost common
pollinators everywhere" — which helps a mixed-pile classification metric and actively hurts
per-plant ranking.

Fixing the objective was not enough. A within-plant pairwise ranking loss reproduced N-only
exactly as a sanity check, but **no learned combination beat raw N** (0.076–0.081). Nor was
it the co-occurrence gate: restricted to each plant's co-occurring candidates (median 5,060;
96% of true partners co-occur), N still won. Nor plant-side timing: a spatially-local per-bin
Δ was genuinely new signal (r = 0.84 with the old Δ, best pooled PR-AUC 0.657) but did not
crack the top-10.

The bottleneck was the **features**: nothing encoded *who these species are*.

---

## What actually worked

| model | recall@10 [95% CI] | hit@10 | Tier-1 recall@10 |
|---|---|---|---|
| degree null | 0.080 [0.067–0.093] | 0.458 | 0.029 |
| N only | 0.115 [0.098–0.132] | 0.508 | 0.117 |
| N + taxonomy affinity + local Δ (linear) | 0.209 [0.188–0.233] | 0.658 | 0.303 |
| GBM, geometry + phenology (val-tuned) | 0.245 [0.221–0.270] | 0.700 | 0.471 |
| two-tower (no embeddings) | 0.225 [0.202–0.250] | 0.675 | 0.469 |
| two-tower + BioCLIP | 0.261 [0.238–0.287] | 0.736 | 0.475 |
| **rank-average ensemble (GBM + tower)** | **0.277 [0.251–0.302]** | **0.741** | **0.504** |

1. **Taxonomic affinity** — per-pollinator affinity for plant genera/families, estimated from
   the training interaction matrix. First feature family to beat N (+78%). Works for plants
   whose genus was never seen, via family.
2. **Feature interactions** — gradient boosting on the *original* geometry/phenology features
   doubled N, no taxonomy needed. The linear probe was the ceiling, not the features.
3. **Foundation-model embeddings as tower inputs.** BioCLIP cosine similarity as a feature
   was useless (0.109); the same embeddings fed to a learned tower gave the best single model
   (0.261). The network learns a task-specific projection that raw name-similarity cannot expose.
4. **Ensembling** GBM with the tower (rank-average) — they fail on different plants.

**Two-tower setup:** one MLP encodes the plant (52-week flowering curve, range size,
genus/family embeddings, optional BioCLIP), one encodes the pollinator (activity curve, range,
taxonomy, BioCLIP); score = dot product + a wide skip carrying `[log N, Δ, local Δ, taxonomy
shares]`. Trained with sampled softmax over within-plant candidate pools (half uniform, half
co-occurring). Model selection on val plants only; one test readout per family.

> **Net: recall@10 0.117 → 0.277 (2.4×), hit@10 0.51 → 0.74, Tier-1 0.117 → 0.504 (4.3×).**

---

## Is it ecology or the observer?

Both features and labels largely come from iNaturalist, so this needed testing.

**Probe.** Using only *real* interactions, ask each feature family to predict **who documented
it**. BioCLIP name embeddings 0.914 AUC · pollinator occupancy PCA 0.819 · N 0.681 ·
**taxonomy affinity 0.621 (lowest)**. Feature families rank by process-predictiveness in
reverse order of how well they transfer.

**Control.** Much of that is taxonomic composition — museums collect bees, iNaturalist
photographs butterflies (Hymenoptera 38% curated vs Lepidoptera 2.4%); family alone predicts
source at 0.824. So we re-ran transfer *within Hymenoptera only*:

| model (within one order) | curated | iNat-only | ratio |
|---|---|---|---|
| pollinator occupancy PCA | 0.091 | 0.180 | **0.51× (degrades)** |
| N only | 0.369 | 0.327 | 1.13× |
| GBM geo | 0.534 | 0.402 | **1.33× (improves)** |

A 2.6× spread with taxon held fixed. **Occupancy embeddings are genuinely process-dependent;
structured and relational features are process-robust.** Note occupancy PCA contains no
taxonomic information at all — *where* things get recorded is itself institutional.

**Prospective validation.** Train only on interactions documented ≤2020, then rank pairs
**first documented 2021+** (2,179 plants, known partners masked out): N 0.104 · linear 0.184 ·
GBM 0.215 · **two-tower+BioCLIP 0.244, hit@10 0.729**. Models built from 2020 knowledge put a
genuinely later-documented partner in the top 10 for ~73% of plants.

---

## Negative results worth keeping

- **V_δ (4-D/15-D) and PMf add nothing** (0.633–0.635 vs 0.636 spatial). The construction in
  `rep_03` flattens each species' own (bin × week) pivot and truncates to the shortest, so
  feature position *k* means a different (bin, week) per species — PCA over misaligned features.
- **Pollinator occupancy PCA (Vp) is actively harmful** for ranking and is the most
  process-contaminated feature we have.
- **BioCLIP-2 ties BioCLIP-1** (0.251 vs 0.261, overlapping CIs). A stronger encoder does not
  help when the input is a species *name* — text towers are essentially soft taxonomy.
- **Floral traits are not obtainable at scale** — the best purpose-built database on Earth has
  corolla depth for 42 species. Design around their absence.
- The ensemble gain is marginal (p = 0.054) and the two-tower/GBM difference is not significant
  (p = 0.11). The top tier is "trees and neural rankers, jointly," not a neural win.

---

## Where it's heading

**Metadata conditioning** is the next build: score = `f_ecology(plant, pollinator) +
g_process(source, year, effort, taxon)` at training, with `g_process` **dropped at inference**
(the "shallow tower" trick from position-bias correction; the ecological ancestor is Fithian
et al. 2015, joint likelihood with per-dataset thinning). The bias head absorbs what is
iNaturalist-specific so the ecology term stops learning it — and it is inspectable, i.e. a
model of each database's sampling behaviour.

Success criterion is *not* a higher headline number: it is the within-order curated/iNat gap
closing, occupancy features moving from 0.51× toward 1.0×. The instrument to prove it already
exists.

Also queued: `n_records` as a graded label instead of binary; sparse ops to enable 0.25°
resolution (0.5° ≈ 55 km is weak evidence of co-occurrence); in-batch negatives; and Dan's
SDM curves, which are spatially resolved and would enable a true per-bin Δ on *both* sides.
