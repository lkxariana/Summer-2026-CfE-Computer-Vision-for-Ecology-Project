## Table 2 — cold-pollinator validation (cold_poll/val), 1927 plants x 1312 candidates, chance AUPR 0.00231

| Method | Reference | seeds | AUPR | AUPR 1:3 | AUPR 1:1 | AUROC | nR@10 | nR@50 | unseen-genus nR@10 |
|---|---|:---:|---:|---:|---:|---:|---:|---:|---:|
| *Nulls* | | |  |  |  |  |  |  |  |
| Pollinator popularity | Aiyappa et al. 2025, ICML | 1 | 0.002 | 0.252 | 0.503 | 0.500 | 0.009 | 0.060 | 0.000 |
| Co-occurrence count N | Blanchet, Cazelles & Gravel 2020, Ecol Lett | 1 | 0.021 | 0.594 | 0.777 | 0.728 | 0.199 | 0.386 | 0.000 |
| Abundance neutral model | Vázquez, Chacoff & Cagnolo 2009, Ecology | 1 | 0.026 | 0.655 | 0.826 | 0.806 | 0.181 | 0.457 | 0.333 |
| *Ecological baselines* | | |  |  |  |  |  |  |  |
| Phenology x abundance | Vizentin-Bugoni et al. 2014, Proc R Soc B | 1 | 0.027 | 0.673 | 0.835 | 0.812 | 0.231 | 0.476 | 0.222 |
| Congeneric transfer | cf. Strydom et al. 2022, MEE | 1 | 0.002 | 0.252 | 0.503 | 0.500 | 0.009 | 0.060 | 0.000 |
| Truncated SVD + taxonomic imputation | Strydom et al. 2022, MEE | 1 | 0.005 | 0.274 | 0.523 | 0.505 | 0.018 | 0.032 | 0.111 |
| Co-occurrence PCA-15 + N (roadmap proof-of-concept; ANTHEIA v1) | Strydom et al. 2021, Phil Trans R Soc B; Li, Cher & Jacobs 2026 | 1 (pre-fix) | 0.033 | 0.700 | 0.851 | 0.832 | 0.298 | 0.558 | **0.444** |
| Co-occurrence PCA-15 + N + Delta (ANTHEIA v1 scalar) | Li, Cher & Jacobs 2026 | 1 (pre-fix) | **0.037** | 0.709 | 0.856 | 0.837 | **0.306** | **0.573** | 0.222 |
| Gradient boosting on pair features | Pichler et al. 2020, MEE | 1 (pre-fix) | 0.031 | **0.718** | **0.864** | **0.852** | 0.184 | 0.465 | 0.111 |
| Spatial x phenological overlap product | Baiotto et al. 2026 (bioRxiv), Eq. 1 | 1 | 0.011 | 0.573 | 0.789 | 0.807 | 0.062 | 0.289 | 0.111 |
| NECTAR-style plausibility (genus constraint x overlap product) | Baiotto et al. 2026 (bioRxiv) | 1 | 0.011 | 0.573 | 0.789 | 0.807 | 0.062 | 0.289 | 0.111 |
| *Published architectures, re-implemented* | | |  |  |  |  |  |  |  |
| Two-tower retrieval (sampled softmax, logQ) | Yi et al. 2019, RecSys | 1 (pre-fix) | 0.008 | 0.403 | 0.647 | 0.624 | 0.039 | 0.108 | 0.000 |
| Pair MLP (NCF-style) | He et al. 2017, WWW | 1 (pre-fix) | 0.002 | 0.244 | 0.492 | 0.497 | 0.018 | 0.068 | 0.000 |
| Wide & Deep | Cheng et al. 2016 | 1 (pre-fix) | 0.029 | 0.653 | 0.826 | 0.817 | 0.188 | 0.383 | 0.000 |
| DCN-V2 (2 cross layers) | Wang et al. 2021, WWW | 1 (pre-fix) | 0.007 | 0.465 | 0.707 | 0.711 | 0.075 | 0.188 | 0.000 |
| *Hand-engineered features (our earlier system)* | | |  |  |  |  |  |  |  |
| Boosted ranker: taxonomy + spatial + per-cell features | this work, v2 | 1 (pre-fix) | 0.003 | 0.282 | 0.536 | 0.515 | 0.019 | 0.072 | 0.000 |
| Genus-routed experts (booster / SVD by genus) | this work, v2 | 1 (pre-fix) | 0.003 | 0.281 | 0.536 | 0.515 | 0.019 | 0.073 | 0.111 |
| *Ours: graph retriever + re-ranker* | | |  |  |  |  |  |  |  |
| **System v2: R-GCN (symmetric leave-own-edges-out, R3) + identity re-ranker** | this work | 1 (incomplete) (pre-fix) | 0.010 | 0.484 | 0.728 | 0.766 | 0.061 | 0.142 | 0.000 |
|   ablation: R3 retriever alone | this work | 3 (pre-fix) | 0.026 | 0.670 | 0.839 | 0.833 | 0.220 | 0.437 | 0.000 |
| System v1: R-GCN (plant-side leave-own-edges-out) + identity re-ranker | this work | 3 (pre-fix) | 0.013 | 0.539 | 0.757 | 0.765 | 0.233 | 0.359 | 0.074 |
|   ablation: frozen R-GCN retriever alone | this work | 3 (pre-fix) | 0.005 | 0.408 | 0.667 | 0.720 | 0.130 | 0.284 | 0.037 |
|   ablation: embedding-model retriever alone | this work | 1 (incomplete) (pre-fix) | 0.010 | 0.501 | 0.731 | 0.727 | 0.083 | 0.217 | 0.222 |

*AUPR at network prevalence is primary; AUPR 1:3 and 1:1 re-weight negatives from the full ranking (the population version of sampled-negative evaluation). Deterministic baselines are single runs; learned models are averaged over seeds {42, 0, 1}. Bold = column best.*
