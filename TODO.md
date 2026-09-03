# Modelling requirements

What must exist before any model runs. The network is built and verified
(`data/network/`, 198,327 interactions, 12,347 plant taxa × 15,921 pollinator taxa). Nothing below
exists yet for that universe — the current caches are keyed to the superseded species set.

Ordered by dependency. Items 1–3 block everything.

---

## 1. Define the modelled universe

**Decision needed first, because every cache is sized by it.**

The network has 12,347 plant and 15,921 pollinator taxa. Feature coverage is much narrower:

| | taxa in network | with existing features | interactions modellable |
|---|---:|---:|---:|
| species rank only | — | 4,243 plants / 6,217 pollinators | 82,973 |
| **+ genus aggregation** | — | 998 extra plant genera / 607 extra pollinator genera | **117,880** |

Genus aggregation defines a genus node's features as the union (occupancy) or mean (curves) over its
feature-covered congeners. It recovers 34,907 interactions and is the recommended default.

**Output:** `data/network/modelled_universe.json` — ordered lists `plants[]` and `pollinators[]`, each
entry with `id`, `label`, `rank`, and `feature_source` ∈ {direct, genus_aggregate}. Every cache below
is indexed to these orders. Nothing downstream may reindex.

---

## 2. Core feature caches

All indexed to item 1's ordering. Written to `data/features/`.

| # | artifact | shape | built from | notes |
|---|---|---|---|---|
| 2.1 | `F.npy` plant occupancy | n_plants × 3,162, bool | PhenoField / PPE occurrence records | genus rows are the union over congeners |
| 2.2 | `P.npy` pollinator occupancy | n_polls × 3,162, bool | GBIF occurrence extract | same |
| 2.3 | `FC.npy` flowering curves | n_plants × 52, float32 | PPE surface, marginalised over cells | row-normalised |
| 2.4 | `AC.npy` activity curves | n_polls × 52, float32 | GBIF weekly histograms | row-normalised; week = (doy−1)//7 clipped to [0,51] |
| 2.5 | `N.npy` co-occurrence | n_plants × n_polls, uint16 | `F @ P.T` over shared cells | ~390 MB dense; sparse if memory-bound |
| 2.6 | `Frs.npy`, `Prs.npy` range sizes | n_plants, n_polls | row sums of F, P | |
| 2.7 | `taxonomy.parquet` | per taxon | node tables | genus, family, order; integer-coded for embeddings |

**Verification for each:** row count matches item 1; no all-zero rows; curves sum to 1; N is
symmetric-consistent with F and P on a sample of pairs.

---

## 3. Splits

Frozen before any model runs, written to `data/splits/`.

| # | split | definition | file |
|---|---|---|---|
| 3.1 | leave-plant-out (primary) | degree-stratified 75/10/15 over plant taxa | `plants_75_10_15.json` |
| 3.2 | leave-pollinator-out | same over pollinator taxa | `pollinators_75_10_15.json` |
| 3.3 | both-new | intersection of held-out plants × held-out pollinators | derived |
| 3.4 | source holdout — expert field networks | all interactions whose only source is Web of Life removed from training (≈10,483) | `holdout_webofline.json` |
| 3.5 | source holdout — specimen records | same for `gbif-us-bees` (≈6,613) | `holdout_gbifusbees.json` |
| 3.6 | prospective | train on interactions first recorded ≤ cutoff, test on later ones | `temporal_2020.json` |

**Negatives.** Training: rebalanced 1:1–1:3, resampled per seed, mixed uniform / co-occurrence-matched
/ degree-matched. Test pooled: **one frozen set at the network's true connectance (~1:1000)**, shared
by every model. Retrieval: no sampling — score all candidates.

---

## 4. Ablation inputs

Beyond the core caches. Grouped by the panel each populates.

### Panel A — spatial

| # | input | status | source |
|---|---|---|---|
| 4.1 | shared cells, Jaccard | derived from 2.1/2.2 | — |
| 4.2 | occupancy PCA (15D) | derived | — |
| 4.3 | SINR species embeddings (256D) | **to download** | `elijahcole/sinr`, weights at data.caltech.edu; read per-species vectors off the classifier head |
| 4.4 | RANGE location embeddings (1280D), pooled per species | **to download** | `mvrl/RANGE` + `microsoft/SatCLIP-ViT16-L40` + `mvrl/RANGE-database` |

### Panel B — temporal

| # | input | status | source |
|---|---|---|---|
| 4.5 | overlap coefficient Σ min(f,a) | derived from 2.3/2.4 | — |
| 4.6 | seven overlap statistics | derived | already implemented in `eval/run_overlap_summaries.py` |
| 4.7 | raw 52-week curves | = 2.3, 2.4 | — |
| 4.8 | plant per-cell surfaces | **to rebuild** for new universe | PPE parquets → n_plants × 3,162 × 52 float16 (~2 GB at previous size) |
| 4.9 | pollinator per-cell surfaces | **partial** — SDM covers 1,275 taxa only | Dan's SDM output |
| 4.10 | phenology-model embeddings | **blocked** — see item 7 | PPE / SDM internal representations |

### Panel C — identity

| # | input | status | source |
|---|---|---|---|
| 4.11 | taxonomic affinity | derived from training edges | must be computed inside `fit` |
| 4.12 | BioCLIP text embeddings (512D) | **to rebuild** for new taxa | `imageomics/bioclip`, ~15 s on GPU |
| 4.13 | BioCLIP-2 image embeddings (768D) | **to rebuild** for new taxa | `imageomics/TreeOfLife-200M-Embeddings`, taxon centroids |
| 4.14 | TaxaBind embeddings (512D) | **to build** | `MVRL/taxabind-vit-b-16` |

---

## 5. Evaluation harness

| # | item | notes |
|---|---|---|
| 5.1 | metric functions | exist in `src/antheia/metrics.py` — recall@{10,20}, nDCG@{10,20}, rank of first partner, PR-AUC |
| 5.2 | runner over `baselines.REGISTRY` | fit on train, score all candidates per test plant, emit per-plant scores |
| 5.3 | paired bootstrap over test plants | for model-vs-model differences |
| 5.4 | table emitters | one per table in `docs/paper/tables.md` |

Per-plant score vectors must be persisted so significance tests and error analysis do not require refits.

---

## 6. Model changes

| # | change | reason |
|---|---|---|
| 6.1 | logQ correction on sampled softmax | in-batch negatives are popularity-proportional; degree spans 2–2,581 (Yi et al. 2019) |
| 6.2 | mixed negative sampling | pollinators with no training edges never appear as negatives otherwise (Yang et al. 2020) |
| 6.3 | per-cell phenology encoder | the contribution: encode each species' 52-week vector at each shared cell, match, aggregate |

---

## 7. Blocked — needs Dan

| # | item | what is needed |
|---|---|---|
| 7.1 | **phenology-model embeddings** | Does PPE expose per-species internal vectors, and can they be exported? If PPE is SINR-like, the species embedding is the classifier head and is a matrix read-off. |
| 7.2 | **pollinator SDM architecture** | Is the SDM the same architecture as PPE (shared location encoder + per-species head), or independent per-species models? If independent, there is no shared space and a symmetric embedding comparison is not defined. |
| 7.3 | **pollinator feature coverage** | 6,333 pollinator taxa in the network are absent from the occurrence extract entirely. Closing that gap needs a new GBIF download, not reprocessing. Decide whether to scope around it or acquire. |

---

## Critical path

**1 → 2 → 3 → 5 → baselines.** Items 4 and 6 extend the ablation but are not needed for the first
results table. Item 7 gates only the top row of Panel B.
