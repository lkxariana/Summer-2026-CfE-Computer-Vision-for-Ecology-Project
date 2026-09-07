## Table 1 — The interaction network

| | Tier A | Tier A+B |
|---|---:|---:|
| Interactions | 101,070 | 192,945 |
| Plant taxa (species / genus) | 5,716 / 962 | 9,483 / 1,548 |
| Pollinator taxa (species / genus) | 5,140 / 370 | 13,697 / 1,313 |
| Connectance | 0.27% | 0.12% |
| Plant degree — median / max | 4 / 845 | 4 / 1,360 |
| Pollinator degree — median / max | 3 / 2,045 | 2 / 2,492 |
| Interactions with ≥2 independent records | 42.0% | 36.0% |
| Interactions with ≥2 source datasets | 9.2% | 7.9% |
| Source datasets | 41 | 51 |
| Interactions carrying a year | 89.6% | 94.0% |
| First record — median (5–95%) | 2020 (2004–2025) | 2019 (1955–2025) |
| First record — full extent | 1700–2026 | 1700–2026 |
| Modelled subgraph (both sides feature-covered) | 99,033 | 180,349 |

## Table 2 — Model comparison

| | | | All held-out plants |  | Expert field networks |  | Specimen records | |
|---|---|:---:|---:|---:|---:|---:|---:|---:|
| **Method** | **Reference** | | nR@10 | PR-AUC | nR@10 | PR-AUC | nR@10 | PR-AUC |
| *Nulls* | | | | | | | | |
| Pollinator popularity | Aiyappa et al. 2025, ICML (arXiv:2405.14985) | ✓ | — | — | 0.043 | 0.0040 | 0.206 | 0.0275 |
| Co-occurrence count |  |  | — | — | — | — | — | — |
| Abundance neutral model |  |  | — | — | — | — | — | — |
| *Structured ecological baselines* | | | | | | | | |
| Congeneric transfer | phylogenetic-signal baseline; cf. Strydom et al. 2022, Methods Ecol Evol 13:2308 | ✓ | — | — | 0.067 | 0.0034 | 0.276 | 0.0129 |
| Phenology x abundance likelihood |  |  | — | — | — | — | — | — |
| Trait matching (reduced coverage) |  |  | — | — | — | — | — | — |
| *Learned representations* | | | | | | | | |
| Truncated SVD + taxonomic imputation | Strydom et al. 2022, Methods Ecol Evol 13:2308 | ✓ | — | — | 0.073 | 0.0045 | 0.396 | 0.0227 |
| LightFM (WARP) |  |  | — | — | — | — | — | — |
| *Feature-based* | | | | | | | | |
| Gradient boosting on pair features |  |  | — | — | — | — | — | — |
| **Taxonomy + spatial + per-cell (ours)** | this work | ✓ | — | — | 0.060 | 0.0043 | 0.291 | 0.0275 |
| *Learned end-to-end (ours)* | | | | | | | | |
| Two-tower retrieval |  |  | — | — | — | — | — | — |
| **Neural pair ranker (ours)** | this work | ✓ | — | — | 0.067 | 0.0028 | 0.270 | 0.0170 |
| **Embedding two-encoder model (ours)** | this work | ✓ | 0.284 | 0.1590 | 0.028 | 0.0026 | 0.210 | 0.0253 |

*Normalised recall@10 (recall / min(partners, 10)) and PR-AUC at the network's connectance; training uses both evidence tiers, scoring is restricted to Tier A. Cold-start capable methods only: each scores a plant with no training interactions. Held-out plants per set: All held-out plants 0, Expert field networks 46, Specimen records 218. Prevalence baseline for PR-AUC: All held-out plants 0.00132, Expert field networks 0.00141, Specimen records 0.00014.*
