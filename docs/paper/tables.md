# Results tables

Paper-ready skeletons. `—` marks a cell to populate. Metrics follow the frozen protocol:
per-plant **recall@10**, **recall@20**, **nDCG@10**, median rank of the first true partner, and
**PR-AUC evaluated at the network's connectance** with the prevalence baseline stated in the caption.
ROC-AUC is reported in the supplement only: at connectance ≈0.1% it is uninformative
(Poisot 2023, *Methods Ecol. Evol.* 14:1333–1345).

Uncertainty: bootstrap over test plants. Model-vs-model comparisons use a paired bootstrap on
per-plant scores.

---

## Table 1 — The interaction network

| | Tier A | Tier A+B |
|---|---:|---:|
| Interactions | — | — |
| Plant taxa (species / genus) | — / — | — / — |
| Pollinator taxa (species / genus) | — / — | — / — |
| Connectance | —% | —% |
| Plant degree — median / max | — / — | — / — |
| Pollinator degree — median / max | — / — | — / — |
| Interactions with ≥2 independent records | —% | —% |
| Interactions with ≥2 source datasets | —% | —% |
| Source datasets | — | — |
| Observation years | —–— | —–— |
| Modelled subgraph (both sides feature-covered) | — | — |

*Constructed from the Global Biotic Interactions database (Poelen, Simons & Mungall 2014, *Ecological
Informatics* 24:148–159), snapshot 2026-08-26. Tier A retains flower-visitation terms
(`visitsFlowersOf`, `pollinates`); Tier B adds general-association terms (`visits`, `interactsWith`).*

---

## Table 2 — Model comparison

Leave-plant-out (Setting B of Stock et al. 2018, *Neural Computation* 30:2245–2283). Three evaluation
sets: all held-out plants; and two held-out **source datasets** removed from training entirely, which
test transfer to independently-assembled data.

| | | | All held-out plants | | Expert field networks | | Specimen records | |
|---|---|:---:|---:|---:|---:|---:|---:|---:|
| **Method** | **Reference** | | R@10 | PR-AUC | R@10 | PR-AUC | R@10 | PR-AUC |
| *Nulls* | | | | | | | | |
| Pollinator popularity | Aiyappa et al. 2025, ICML | ✓ | — | — | — | — | — | — |
| Co-occurrence count | — | ✓ | — | — | — | — | — | — |
| Abundance neutral model | Vázquez et al. 2009, *Ecology* 90:2039 | ✓ | — | — | — | — | — | — |
| *Structured ecological baselines* | | | | | | | | |
| Congeneric transfer | phylogenetic-signal baseline, cf. Strydom et al. 2022 | ✓ | — | — | — | — | — | — |
| Phenology × abundance likelihood | Vizentin-Bugoni et al. 2014, *Proc R Soc B* 281:20132397 | ✓ | — | — | — | — | — | — |
| Trait matching (reduced coverage) | Pichler et al. 2020, *Methods Ecol Evol* 11:281 | ✓ | — | — | — | — | — | — |
| *Learned representations* | | | | | | | | |
| Truncated SVD + taxonomic imputation | Strydom et al. 2022, *Methods Ecol Evol* 13:2308 | ✓ | — | — | — | — | — | — |
| LightFM (WARP) | Kula 2015, arXiv:1507.08439 | ✓ | — | — | — | — | — | — |
| *Feature-based* | | | | | | | | |
| Gradient boosting on pair features | — | ✓ | — | — | — | — | — | — |
| **Two-tower retrieval (ours)** | after Yi et al. 2019, RecSys | ✓ | — | — | — | — | — | — |
| **+ per-cell phenology encoder (ours)** | this work | ✓ | — | — | — | — | — | — |

*All methods shown are cold-start capable: each scores a plant with no training interactions.
Prevalence baseline for PR-AUC: —.*

**Methods not compared against.** Transductive factorisation and graph models — Bayesian personalised
ranking (Rendle et al. 2009), latent factors with implicit feedback (Seo & Hutchinson 2018, AAAI-18),
LightGCN (He et al. 2020) — learn a free latent vector per species and therefore cannot score a plant
absent from training; they are inapplicable to this setting rather than merely weaker at it.
Kronecker kernel ridge regression (Stock et al. 2021), the source of the evaluation framework used
here, requires dense species-by-species kernels on both sides and is impractical at this network's
scale. Morphological trait matching requires corolla depth, which is recorded for fewer than fifty
species worldwide.

---

## Table 3 — Encoding ablation: explicit → implicit, spatial and temporal

One evaluation set. Each row adds or substitutes a single input; Δ is the paired difference against
the row above. This table is the paper's central claim: **the field summarises both axes into scalars
before modelling, and that summarisation is where the signal is lost.**

### Panel A — spatial information

| | Representation | Source | R@10 | Δ | nDCG@10 | PR-AUC |
|---|---|---|---:|---:|---:|---:|
| explicit | shared occupied cells (count) | — | — | — | — | — |
| explicit | range overlap (Jaccard) | — | — | — | — | — |
| implicit | occupancy embedding (PCA) | — | — | — | — | — |
| implicit | learned range embedding | SINR, Cole et al. 2023, ICML | — | — | — | — |
| implicit | pooled location embedding | RANGE, Dhakal et al. 2025, CVPR | — | — | — | — |

### Panel B — temporal information

| | Representation | Source | R@10 | Δ | nDCG@10 | PR-AUC |
|---|---|---|---:|---:|---:|---:|
| explicit | overlap coefficient Σ min(f,a) | Vizentin-Bugoni et al. 2014 | — | — | — | — |
| explicit | seven overlap statistics | Schoener 1970; Pianka 1974; + | — | — | — | — |
| implicit | raw 52-week curves | this work | — | — | — | — |
| implicit | per-cell overlap (hand-computed) | this work | — | — | — | — |
| implicit | **per-cell learned encoder** | **this work** | — | — | — | — |
| implicit | **phenology-model embedding** | **this work (PPE / SDM latent)** | — | — | — | — |

### Panel C — identity

| | Representation | Source | R@10 | Δ | nDCG@10 | PR-AUC |
|---|---|---|---:|---:|---:|---:|
| explicit | taxonomic affinity (genus, family) | — | — | — | — | — |
| implicit | species-name embedding | BioCLIP, Stevens et al. 2024, CVPR | — | — | — | — |
| implicit | species-image embedding | BioCLIP 2, Gu et al. 2025, NeurIPS | — | — | — | — |
| implicit | multimodal species embedding | TaxaBind, Sastry et al. 2025, WACV | — | — | — | — |

*Panels are additive within themselves and combined in the final row of Table 2. A co-occurrence-only
model is included in every panel, since without it any temporal effect is interpretable as an
abundance proxy (Dormann et al. 2025; Blanchet et al. 2020, *Ecology Letters* 23:1050–1063).*

---

# SUPPLEMENTARY

**S1 — Data flow.** Records entering and leaving each construction step, with counts of orientation
violations, unresolved roles, duplicates removed and immature-stage records excluded.

**S2 — Source composition.** Per source dataset: records, interactions, interactions unique to it,
pollinator-order composition, date range.

**S3 — Generalisation settings.** Best model across new plant, new pollinator, both new
(Settings B, C, D of Stock et al. 2018) and prospective (interactions first recorded after a cut-off),
with the caveat that later records differ in kind from earlier ones.

**S4 — Results by plant degree.** Table 2 split into specialists (≤3 partners), intermediate (4–20)
and generalists (>20), with the achievable ceiling min(k, degree)/degree printed per stratum.

**S5 — Metric sensitivity.** Main results at k ∈ {10, 20, 50}, under nDCG and recall, and with
ROC-AUC for comparability with the earlier literature.

**S6 — Trait coverage.** Per trait: species with values, share of the network covered, source.
Documents why trait matching is evaluated on a reduced subset.

**S7 — Negative sampling.** Results under uniform, co-occurrence-matched and degree-matched negatives,
and under a positive-unlabeled correction with a sweep over the assumed class prior
(Elkan & Noto 2008, KDD; Kiryo et al. 2017, NeurIPS).

**S8 — Model configurations.** Architectures, hyperparameters, selection procedure, training cost.
