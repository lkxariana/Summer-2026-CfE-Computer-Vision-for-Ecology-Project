# Table plan

Three main tables, six supplementary. Cells show the format each value will take.

**Design logic.** Models are compared *across evaluation sets*, because generalisation is what
separates them. Ablations are run on the primary set only, because they answer a mechanism question
and adding evaluation sets would multiply rows without adding evidence. Settings are then swept for
the best model alone.

---

# MAIN

## Table 1 — The interaction network

| | Tier A | Tier A+B |
|---|---|---|
| Interactions | — | — |
| Plant taxa (species / genus) | — / — | — / — |
| Pollinator taxa (species / genus) | — / — | — / — |
| Connectance | —% | —% |
| Plant degree (median / max) | — / — | — / — |
| Pollinator degree (median / max) | — / — | — / — |
| Interactions with ≥2 records | —% | —% |
| Interactions with ≥2 sources | —% | —% |
| Source datasets | — | — |
| Observation years | —–— | —–— |

## Table 2 — Model comparison across evaluation sets

Three column groups: the full held-out set, and two held-out *sources* that were removed from
training entirely. Leave-plant-out split throughout.

| | \multicolumn{2}{c}{All held-out plants} | \multicolumn{2}{c}{Expert field networks} | \multicolumn{2}{c}{Specimen records} |
|---|---|---|---|---|---|---|
| Model | recall@10 | PR-AUC | recall@10 | PR-AUC | recall@10 | PR-AUC |
| **Nulls** | | | | | | |
| Pollinator popularity | | | | | | |
| Co-occurrence count | | | | | | |
| Abundance product | | | | | | |
| **Structured baselines** | | | | | | |
| Congeneric transfer | | | | | | |
| Trait matching (reduced coverage) | | | | | | |
| **Learned representations** | | | | | | |
| Matrix factorisation + taxonomic imputation | | | | | | |
| Pretrained species embeddings | | | | | | |
| Graph neural network *(transductive; upper reference)* | | | | | | |
| Two-tower (ours) | | | | | | |
| **Feature-based** | | | | | | |
| Linear pairwise ranker | | | | | | |
| Gradient boosting | | | | | | |

*Bootstrap 95% intervals over test plants. PR-AUC at the network's connectance; prevalence baseline
in the caption. The graph neural network cannot score unseen taxa and is evaluated only where both
partners were observed in training — it bounds what a transductive method achieves, not a comparable
number.*

## Table 3 — Ablation: which representation carries the signal

Primary evaluation set only. Each row adds or removes one input; the paired difference against the
row above isolates its contribution.

| | | recall@10 | Δ vs. previous | PR-AUC | Δ vs. previous |
|---|---|---|---|---|---|
| **Spatial** | occupancy only (explicit) | | — | | — |
| | + learned occupancy embedding (implicit) | | | | |
| **Temporal** | + phenological overlap (explicit) | | | | |
| | + phenology curves (implicit) | | | | |
| **Spatial × temporal** | + per-cell overlap (explicit) | | | | |
| | + per-cell phenology encoder (implicit) | | | | |
| **Identity** | + taxonomy | | | | |

*Paired bootstrap over test plants. The two axes are the kind of information (spatial, temporal, both)
and how it is represented (hand-computed scalar vs. learned encoder). The final row is the only input
that is simultaneously spatial, temporal and learned.*

---

# SUPPLEMENTARY

**S1 — Data flow.** Records entering and leaving each construction step, with counts of orientation
violations, unresolved roles, duplicates removed and immature-stage records excluded.

**S2 — Source composition.** Per source: records, edges, edges unique to it, pollinator-order
composition, date range.

**S3 — Generalisation settings.** Best model swept across new plant, new pollinator, both new, and
prospective (interactions first recorded after a cut-off), with the caveat that later records differ
in kind from earlier ones.

**S4 — Results by plant degree.** Table 2 split into specialists (≤3 partners), intermediate (4–20)
and generalists (>20), with the achievable ceiling min(k, degree)/degree printed per stratum.

**S5 — Metric sensitivity.** Main results at k ∈ {10, 20, 50} and under nDCG.

**S6 — Trait coverage.** Per trait: species with values, share of the network covered, source.
Documents why trait matching is evaluated on a subset.

**S7 — Model configurations.** Architectures, hyperparameters, selection procedure, training cost.
