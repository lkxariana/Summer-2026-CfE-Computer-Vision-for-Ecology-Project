## Table 3 — local-network completion (91 surveyed networks; mean connectance 0.135 = chance AUPR; observed NODF 36.0)

| Method | Reference | seeds | mean AUPR | lift | pooled AUPR | pooled AUROC | deg rho plants | deg rho polls | NODF pred | precision@L |
|---|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| *Nulls* | | |  |  |  |  |  |  |  |  |
| Pollinator popularity | Aiyappa et al. 2025, ICML | 1 | 0.192 [0.179,0.207] | 1.517 | 0.091 | 0.621 | -0.020 | 0.267 | 92.3 | 0.209 |
| Co-occurrence count N | Blanchet, Cazelles & Gravel 2020, Ecol Lett | 1 | 0.167 [0.153,0.181] | 1.250 | 0.058 | 0.533 | 0.019 | 0.252 | 92.9 | 0.181 |
| Abundance neutral model | Vázquez, Chacoff & Cagnolo 2009, Ecology | 1 | 0.176 [0.162,0.191] | 1.360 | 0.066 | 0.550 | 0.033 | 0.130 | 92.2 | 0.176 |
| *Ecological baselines* | | |  |  |  |  |  |  |  |  |
| Phenology x abundance | Vizentin-Bugoni et al. 2014, Proc R Soc B | 1 | 0.184 [0.169,0.200] | 1.429 | 0.067 | 0.555 | 0.008 | 0.187 | 94.9 | 0.180 |
| Congeneric transfer | cf. Strydom et al. 2022, MEE | 1 | **0.222** [0.205,0.242] | 1.712 | 0.085 | 0.610 | 0.100 | 0.323 | 69.7 | 0.239 |
| Truncated SVD + taxonomic imputation | Strydom et al. 2022, MEE | 1 | 0.179 [0.166,0.192] | 1.368 | 0.078 | 0.607 | 0.133 | 0.279 | 76.0 | 0.199 |
| Co-occurrence PCA-15 + N (roadmap proof-of-concept; ANTHEIA v1) | Strydom et al. 2021, Phil Trans R Soc B; Li, Cher & Jacobs 2026 | 1 | 0.178 [0.164,0.193] | 1.371 | 0.066 | 0.548 | 0.004 | 0.235 | 99.9 | 0.184 |
| Co-occurrence PCA-15 + N + Delta (ANTHEIA v1 scalar) | Li, Cher & Jacobs 2026 | 1 | 0.179 [0.165,0.194] | 1.380 | 0.067 | 0.558 | 0.012 | 0.244 | 99.6 | 0.187 |
| Gradient boosting on pair features | Pichler et al. 2020, MEE | 1 | 0.203 [0.187,0.219] | 1.580 | 0.089 | 0.605 | 0.073 | 0.338 | 83.3 | 0.218 |
| *Published architectures, re-implemented* | | |  |  |  |  |  |  |  |  |
| Wide & Deep | Cheng et al. 2016 | 1 | 0.155 [0.142,0.168] | 1.169 | 0.067 | 0.583 | 0.044 | 0.248 | 44.0 | 0.145 |
| *Ours* | | |  |  |  |  |  |  |  |  |
| Boosted ranker: taxonomy + spatial + per-cell | this work | 1 | 0.222 [0.205,0.239] | 1.725 | 0.095 | 0.614 | **0.197** | 0.347 | 62.4 | 0.252 |
| Genus-routed experts | this work | 1 | 0.222 [0.205,0.239] | **1.726** | **0.095** | 0.614 | 0.197 | **0.347** | 62.5 | **0.252** |
| Embedding model (identity + field blocks) | this work | 1 | 0.163 [0.150,0.177] | 1.239 | 0.078 | **0.632** | 0.002 | 0.261 | **43.1** | 0.161 |

*Each network's plants x pollinators block is scored with every local-network pair removed from training. An absent pair among surveyed species is an observed non-interaction. Predicted networks for the structure columns take the top-L pairs, L = observed links; NODF closest to observed is bold. Bootstrap CI over networks.*
