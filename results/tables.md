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
| Pollinator popularity | Aiyappa et al. 2025, ICML (arXiv:2405.14985) | ✓ | 0.244 | 0.0526 | 0.043 | 0.0040 | 0.206 | 0.0275 |
| Co-occurrence count | Blanchet, Cazelles & Gravel 2020, Ecology Letters 23:1050 | ✓ | 0.105 | 0.0264 | 0.020 | 0.0016 | 0.036 | 0.0015 |
| Abundance neutral model | Vázquez, Chacoff & Cagnolo 2009, Ecology 90:2039 | ✓ | 0.140 | 0.0239 | 0.032 | 0.0024 | 0.056 | 0.0040 |
| *Structured ecological baselines* | | | | | | | | |
| Congeneric transfer | phylogenetic-signal baseline; cf. Strydom et al. 2022, Methods Ecol Evol 13:2308 | ✓ | 0.307 | 0.0628 | 0.062 | 0.0045 | 0.280 | 0.0156 |
| Phenology x abundance likelihood | Vizentin-Bugoni, Maruyama & Sazima 2014, Proc R Soc B 281:20132397 | ✓ | 0.158 | 0.0240 | 0.034 | 0.0022 | 0.060 | 0.0043 |
| Trait matching (reduced coverage) |  |  | — | — | — | — | — | — |
| *Learned representations* | | | | | | | | |
| Truncated SVD + taxonomic imputation | Strydom et al. 2022, Methods Ecol Evol 13:2308 | ✓ | 0.289 | 0.0950 | 0.073 | 0.0044 | 0.379 | 0.0232 |
| LightFM (WARP) | Kula 2015, arXiv:1507.08439 | ✓ | 0.010 | 0.0025 | 0.000 | 0.0015 | 0.000 | 0.0002 |
| *Feature-based* | | | | | | | | |
| Gradient boosting on pair features | Pichler et al. 2020, Methods Ecol Evol 11:281 | ✓ | 0.290 | 0.0786 | 0.032 | 0.0036 | 0.229 | 0.0262 |
| **Two-tower retrieval (ours)** |  |  | — | — | — | — | — | — |
| **+ per-cell phenology encoder (ours)** |  |  | — | — | — | — | — | — |

*Normalised recall@10 (recall / min(partners, 10)) and PR-AUC at the network's connectance; training uses both evidence tiers, scoring is restricted to Tier A. Cold-start capable methods only: each scores a plant with no training interactions. Held-out plants per set: All held-out plants 663, Expert field networks 46, Specimen records 218. Prevalence baseline for PR-AUC: All held-out plants 0.00132, Expert field networks 0.00141, Specimen records 0.00014.*
