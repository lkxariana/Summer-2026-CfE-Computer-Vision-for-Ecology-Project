## Table 2 — cold-pollinator validation (cold_poll/val), 1927 plants x 1312 candidates, chance AUPR 0.00231

| Method | Reference | seeds | AUPR | AUPR 1:3 | AUPR 1:1 | AUROC | nR@10 | nR@50 | unseen-genus nR@10 |
|---|---|:---:|---:|---:|---:|---:|---:|---:|---:|
| *Nulls* | | |  |  |  |  |  |  |  |
| Pollinator popularity | Aiyappa et al. 2025, ICML | 1 | 0.002 | 0.252 | 0.503 | 0.500 | 0.009 | 0.060 | 0.000 |
| Co-occurrence count N | Blanchet, Cazelles & Gravel 2020, Ecol Lett | 1 | 0.021 | 0.594 | 0.777 | 0.728 | 0.199 | 0.386 | 0.000 |
| Abundance neutral model | Vázquez, Chacoff & Cagnolo 2009, Ecology | 1 | 0.026 | 0.655 | 0.826 | 0.806 | 0.181 | 0.457 | **0.333** |
| *Ecological baselines* | | |  |  |  |  |  |  |  |
| Phenology x abundance | Vizentin-Bugoni et al. 2014, Proc R Soc B | 1 | 0.027 | 0.673 | 0.835 | 0.812 | 0.231 | 0.476 | 0.222 |
| Congeneric transfer | cf. Strydom et al. 2022, MEE | 1 | 0.002 | 0.252 | 0.503 | 0.500 | 0.009 | 0.060 | 0.000 |
| Truncated SVD + taxonomic imputation | Strydom et al. 2022, MEE | 1 | 0.005 | 0.274 | 0.523 | 0.505 | 0.018 | 0.032 | 0.111 |
| Co-occurrence PCA-15 + N (roadmap proof-of-concept; ANTHEIA v1) | Strydom et al. 2021, Phil Trans R Soc B; Li, Cher & Jacobs 2026 | 1 | 0.033 | 0.699 | 0.851 | 0.833 | 0.284 | 0.548 | **0.333** |
| Co-occurrence PCA-15 + N + Delta (ANTHEIA v1 scalar) | Li, Cher & Jacobs 2026 | 1 | 0.036 | 0.708 | 0.856 | 0.838 | 0.305 | 0.570 | 0.222 |
| Gradient boosting on pair features | Pichler et al. 2020, MEE | 1 | 0.044 | 0.745 | 0.876 | 0.858 | 0.251 | 0.552 | 0.222 |
| Spatial x phenological overlap product | Baiotto et al. 2026 (bioRxiv), Eq. 1 | 1 | 0.011 | 0.573 | 0.789 | 0.807 | 0.062 | 0.289 | 0.111 |
| NECTAR-style plausibility (genus constraint x overlap product) | Baiotto et al. 2026 (bioRxiv) | 1 | 0.011 | 0.573 | 0.789 | 0.807 | 0.062 | 0.289 | 0.111 |
| *Published architectures, re-implemented* | | |  |  |  |  |  |  |  |
| Two-tower retrieval (sampled softmax, logQ) | Yi et al. 2019, RecSys | 1 | 0.006 | 0.375 | 0.623 | 0.596 | 0.031 | 0.088 | 0.000 |
| Pair MLP (NCF-style) | He et al. 2017, WWW | 1 | 0.003 | 0.281 | 0.539 | 0.548 | 0.028 | 0.116 | 0.000 |
| Wide & Deep | Cheng et al. 2016 | 1 | 0.029 | 0.653 | 0.826 | 0.817 | 0.188 | 0.383 | 0.000 |
| DCN-V2 (2 cross layers) | Wang et al. 2021, WWW | 1 | 0.007 | 0.465 | 0.707 | 0.711 | 0.075 | 0.188 | 0.000 |
| *Hand-engineered features (our earlier system)* | | |  |  |  |  |  |  |  |
| Boosted ranker: taxonomy + spatial + per-cell features | this work, v2 | 1 | 0.003 | 0.297 | 0.553 | 0.527 | 0.026 | 0.099 | 0.000 |
| Genus-routed experts (booster / SVD by genus) | this work, v2 | 1 | 0.003 | 0.285 | 0.539 | 0.513 | 0.026 | 0.097 | 0.111 |
| *Ours: graph retriever + re-ranker* | | |  |  |  |  |  |  |  |
| **System v3: species-node R-GCN (text + interaction edges, symmetric rehearsal, no taxon nodes) + identity re-ranker** | this work | 3 | 0.135 | 0.832 | 0.923 | 0.917 | 0.399 | 0.627 | 0.259 |
|   ablation: v3 retriever alone | this work | 3 | 0.129 | 0.816 | 0.915 | 0.906 | 0.401 | 0.619 | 0.148 |
|   factorised retriever: v3 + explicit co-presence (opportunity) term | this work | 3 | 0.153 | 0.840 | 0.927 | 0.919 | 0.425 | 0.662 | 0.111 |
|   factorised retriever + identity re-ranker | this work | 3 | 0.162 | 0.851 | 0.933 | 0.926 | 0.427 | 0.666 | 0.222 |
|   variant: presence-embedding retriever + re-ranker | this work | 1 (incomplete) | 0.157 | **0.860** | **0.937** | **0.931** | 0.422 | **0.687** | 0.111 |
|   ablation: v3 system without cell x month nodes | this work | 1 (incomplete) | 0.116 | 0.821 | 0.919 | 0.915 | 0.358 | 0.596 | 0.111 |
| System v2: R-GCN (symmetric leave-own-edges-out, R3) + identity re-ranker | this work | 3 | 0.117 | 0.827 | 0.921 | 0.915 | 0.361 | 0.620 | 0.185 |
|   ablation: R3 retriever alone | this work | 3 | 0.101 | 0.809 | 0.913 | 0.906 | 0.352 | 0.608 | 0.111 |
|   ablation: R3 without cell x month nodes | this work | 1 (incomplete) | 0.097 | 0.800 | 0.909 | 0.905 | 0.355 | 0.580 | 0.111 |
|   ablation: R3 with month-collapsed cells | this work | 1 (incomplete) | 0.099 | 0.812 | 0.914 | 0.909 | 0.366 | 0.613 | 0.111 |
| System v1: R-GCN (plant-side leave-own-edges-out) + identity re-ranker | this work | 3 | 0.008 | 0.524 | 0.760 | 0.803 | 0.193 | 0.417 | 0.074 |
|   ablation: frozen R-GCN retriever alone | this work | 3 | 0.006 | 0.452 | 0.704 | 0.761 | 0.166 | 0.350 | 0.037 |
|   ablation: embedding-model retriever alone | this work | 1 (incomplete) | 0.010 | 0.501 | 0.731 | 0.727 | 0.083 | 0.217 | 0.222 |

*AUPR at network prevalence is primary; AUPR 1:3 and 1:1 re-weight negatives from the full ranking (the population version of sampled-negative evaluation). Deterministic baselines are single runs; learned models are averaged over seeds {42, 0, 1}. Bold = column best.*
