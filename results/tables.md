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
| **Method** | **Reference** | | R@10 | PR-AUC | R@10 | PR-AUC | R@10 | PR-AUC |
| *Nulls* | | | | | | | | |
| Pollinator popularity | Aiyappa et al. 2025, ICML (arXiv:2405.14985) | ✓ | 0.107 | 0.0335 | 0.041 | 0.0040 | 0.176 | 0.0164 |
| Co-occurrence count | Blanchet, Cazelles & Gravel 2020, Ecology Letters 23:1050 | ✓ | 0.035 | 0.0227 | 0.015 | 0.0016 | 0.028 | 0.0009 |
| Abundance neutral model | Vázquez, Chacoff & Cagnolo 2009, Ecology 90:2039 | ✓ | 0.053 | 0.0140 | 0.028 | 0.0024 | 0.047 | 0.0021 |
| *Structured ecological baselines* | | | | | | | | |
| Congeneric transfer | phylogenetic-signal baseline; cf. Strydom et al. 2022, Methods Ecol Evol 13:2308 | ✓ | 0.206 | 0.0542 | 0.055 | 0.0045 | 0.238 | 0.0119 |
| Phenology x abundance likelihood | Vizentin-Bugoni, Maruyama & Sazima 2014, Proc R Soc B 281:20132397 | ✓ | 0.020 | 0.0133 | 0.013 | 0.0019 | 0.012 | 0.0006 |
| Trait matching (reduced coverage) |  |  | — | — | — | — | — | — |
| *Learned representations* | | | | | | | | |
| Truncated SVD + taxonomic imputation | Strydom et al. 2022, Methods Ecol Evol 13:2308 | ✓ | 0.138 | 0.0717 | 0.060 | 0.0044 | 0.314 | 0.0180 |
| LightFM (WARP) | Kula 2015, arXiv:1507.08439 | ✓ | 0.028 | 0.0072 | 0.011 | 0.0013 | 0.005 | 0.0013 |
| *Feature-based* | | | | | | | | |
| Gradient boosting on pair features | Pichler et al. 2020, Methods Ecol Evol 11:281 | ✓ | 0.117 | 0.0486 | 0.025 | 0.0039 | 0.184 | 0.0172 |
| **Two-tower retrieval (ours)** |  |  | — | — | — | — | — | — |
| **+ per-cell phenology encoder (ours)** |  |  | — | — | — | — | — | — |

*Cold-start capable methods only: each scores a plant with no training interactions. Held-out plants per set: All held-out plants 1085, Expert field networks 46, Specimen records 284. Prevalence baseline for PR-AUC: All held-out plants 0.00139, Expert field networks 0.00141, Specimen records 0.00023.*
