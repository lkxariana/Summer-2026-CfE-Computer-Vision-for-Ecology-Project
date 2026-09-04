# Modelling requirements

What must exist before any model runs. The network is built and verified
(`data/network/`, 192,945 interactions, 11,031 plant taxa x 15,010 pollinator taxa). Items 1-3 are
done; run `scripts/run_build_pipeline.sh` to regenerate the chain end to end.

Ordered by dependency.

---

## 1. Define the modelled universe — **done**

`data/network/modelled_universe.json`, built by `scripts/build_modelled_universe.py`.

Every taxon in the universe carries a spatially explicit per-cell surface: a supervised species head
where observations exist, a text-conditioned zero-shot head otherwise. Genus aggregation is not used
(it lost to the text zero-shot head, 0.493 vs 0.566 curve overlap on held-out covered plants);
`has_congener` marks where it would apply, so the ablation needs no rebuild.

| | in universe | direct | zero-shot |
|---|---:|---:|---:|
| plants | 11,031 | 4,770 | 6,261 |
| pollinators | 13,124 | 5,354 | 7,770 |

1,886 pollinator taxa outside the six core flower-visiting orders are excluded, as are 2 whose order
the checklists disagree on. **180,349 of 192,945 interactions (93.5%) are modellable.**

---

## 2. Core feature caches — **done except FCm**

`data/features/`, built by `scripts/build_feature_caches.py` and `scripts/build_modelled_curves.py`,
indexed to item 1's order. Verified by `tests/test_features.py` (9/9).

| artifact | shape | source |
|---|---|---|
| `F.npy` / `P.npy` | taxa x 3,335 cells, bool | flowering observations (e98 cache) / GBIF occurrences |
| `FCo.npy` / `ACo.npy` | taxa x 52, float32 | weekly histograms of the same records |
| `FCm.npy` / `ACm.npy` | taxa x 52, float32 | per-cell surfaces marginalised over cells |
| `N.npy` | 11,031 x 13,124, uint16 | `F @ P.T` over the shared cells |
| `Frs.npy` / `Prs.npy` | range sizes | row sums |
| `taxonomy.parquet`, `coverage.parquet`, `grid.parquet` | per taxon / per cell | node tables, universe |

The observed caches (`F`, `P`, `FCo`, `ACo`, `N`) exist only for taxa with their own records: 4,835
of 11,031 plants and 5,344 of 13,124 pollinators. That is the explicit end of the feature axis. The
modelled curves cover everything — `ACm` is built; **`FCm` waits on the zero-shot plant surfaces.**

The per-cell surfaces are not densified. 13,124 pollinators x 3,335 cells x 52 weeks is several
terabytes; location-conditioned models read the parquets on demand.

---

## 3. Splits — **done**

`data/splits/`, built by `scripts/build_splits.py`, seed 42. Verified by `tests/test_splits.py` (8/8).

| split | file | size |
|---|---|---|
| leave-plant-out, degree-stratified 75/10/15 | `plants_75_10_15.json` | 8,123 / 1,085 / 1,624 (+199 with no interactions) |
| leave-pollinator-out | `pollinators_75_10_15.json` | 9,843 / 1,312 / 1,969 |
| both-new (Stock et al. Setting D) | `both_new.json` | 4,320 test interactions |
| source holdout — Web of Life | `holdout_webofline.json` | 10,402 sole-supported |
| source holdout — specimen records | `holdout_gbifusbees.json` | 6,126 sole-supported |
| prospective, cutoff 2020 | `temporal_2020.json` | 97,749 train / 71,184 test |

**Negatives.** Training: rebalanced 1:1-1:3, resampled per seed, mixed uniform / co-occurrence-matched
/ degree-matched. Test pooled: one frozen set at the network's true connectance, shared by every
model. Retrieval: no sampling — score all candidates.

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

## 7. Resolved by the HPC handoff

| # | item | resolution |
|---|---|---|
| 7.1 | phenology-model embeddings | Exported. `ppe_embeddings/species_static.npz` [6,825 x 576] is the per-species e98 embedding, constant over space and time; `grid_zdyn_null.npz` [98,817 x 192] is the species-free climate-phenology state per (cell, week), the input for 4.10. |
| 7.2 | pollinator SDM architecture | Shared location encoder with per-species heads, the SINR form, plus an LE-SINR head conditioned on a BioCLIP-2 text embedding for taxa with no occurrences. A symmetric embedding comparison is therefore defined. |
| 7.3 | pollinator feature coverage | Closed by a new GBIF download (7.56M records, 5,822 species). Taxa still without occurrences are covered zero-shot: the deliverable spans all 13,124 universe pollinators. |

---

## Critical path

**1 → 2 → 3 → 5 → baselines.** Items 4 and 6 extend the ablation but are not needed for the first
results table. Item 7 gates only the top row of Panel B.
