## Table 2 — cold-plant validation (cold_plant/val), 663 plants x 13124 candidates, chance AUPR 0.00132

| Method | Reference | seeds | AUPR | AUPR 1:3 | AUPR 1:1 | AUROC | nR@10 | nR@50 | unseen-genus nR@10 |
|---|---|:---:|---:|---:|---:|---:|---:|---:|---:|
| *Nulls* | | |  |  |  |  |  |  |  |
| Pollinator popularity | Aiyappa et al. 2025, ICML | 1 | 0.053 | 0.837 | 0.929 | 0.926 | 0.244 | 0.373 | 0.234 |
| Co-occurrence count N | Blanchet, Cazelles & Gravel 2020, Ecol Lett | 1 | 0.026 | 0.591 | 0.760 | 0.709 | 0.105 | 0.203 | 0.064 |
| Abundance neutral model | Vázquez, Chacoff & Cagnolo 2009, Ecology | 1 | 0.024 | 0.678 | 0.837 | 0.816 | 0.140 | 0.239 | 0.098 |
| *Ecological baselines* | | |  |  |  |  |  |  |  |
| Phenology x abundance | Vizentin-Bugoni et al. 2014, Proc R Soc B | 1 | 0.024 | 0.692 | 0.844 | 0.822 | 0.158 | 0.237 | 0.107 |
| Congeneric transfer | cf. Strydom et al. 2022, MEE | 1 | 0.057 | 0.812 | 0.904 | 0.876 | 0.293 | 0.444 | 0.213 |
| Truncated SVD + taxonomic imputation | Strydom et al. 2022, MEE | 1 | 0.095 | 0.850 | 0.924 | 0.895 | 0.291 | 0.449 | 0.256 |
| Co-occurrence PCA-15 + N (roadmap proof-of-concept; ANTHEIA v1) | Strydom et al. 2021, Phil Trans R Soc B; Li, Cher & Jacobs 2026 | 1 | 0.048 | 0.741 | 0.868 | 0.839 | 0.117 | 0.256 | 0.087 |
| Co-occurrence PCA-15 + N + Delta (ANTHEIA v1 scalar) | Li, Cher & Jacobs 2026 | 1 | 0.052 | 0.751 | 0.875 | 0.849 | 0.118 | 0.260 | 0.089 |
| Gradient boosting on pair features | Pichler et al. 2020, MEE | 1 | 0.079 | 0.857 | 0.936 | 0.930 | 0.290 | 0.442 | **0.276** |
| Spatial x phenological overlap product | Baiotto et al. 2026 (bioRxiv), Eq. 1 | 1 | 0.012 | 0.669 | 0.843 | 0.847 | 0.008 | 0.051 | 0.003 |
| NECTAR-style plausibility (genus constraint x overlap product) | Baiotto et al. 2026 (bioRxiv) | 1 | 0.081 | 0.803 | 0.905 | 0.890 | 0.161 | 0.297 | 0.003 |
| *Published architectures, re-implemented* | | |  |  |  |  |  |  |  |
| Two-tower retrieval (sampled softmax, logQ) | Yi et al. 2019, RecSys | 1 | 0.099 | 0.721 | 0.846 | 0.797 | 0.277 | 0.374 | 0.064 |
| Pair MLP (NCF-style) | He et al. 2017, WWW | 1 | 0.116 | 0.711 | 0.843 | 0.801 | 0.281 | 0.384 | 0.068 |
| Wide & Deep | Cheng et al. 2016 | 1 | 0.144 | 0.867 | 0.937 | 0.922 | 0.306 | 0.444 | 0.164 |
| DCN-V2 (2 cross layers) | Wang et al. 2021, WWW | 1 | 0.138 | 0.880 | 0.948 | 0.945 | 0.263 | 0.384 | 0.107 |
| TabICL (in-context tabular) | Qu et al. 2025 | 1 | 0.110 | 0.826 | 0.915 | 0.897 | 0.313 | 0.437 | 0.168 |
| *Hand-engineered features (our earlier system)* | | |  |  |  |  |  |  |  |
| Boosted ranker: taxonomy + spatial + per-cell features | this work, v2 | 1 | 0.102 | 0.806 | 0.904 | 0.881 | 0.320 | 0.426 | 0.122 |
| Genus-routed experts (booster / SVD by genus) | this work, v2 | 1 | 0.103 | 0.810 | 0.906 | 0.883 | 0.336 | 0.447 | 0.256 |
| *Ours: graph retriever + re-ranker* | | |  |  |  |  |  |  |  |
| **System v2: R-GCN (symmetric leave-own-edges-out, R3) + identity re-ranker** | this work | 3 | **0.209** | **0.931** | **0.972** | **0.972** | 0.383 | 0.549 | 0.245 |
|   ablation: R3 retriever alone | this work | 3 | 0.170 | 0.916 | 0.966 | 0.966 | 0.375 | 0.541 | 0.236 |
| System v1: R-GCN (plant-side leave-own-edges-out) + identity re-ranker | this work | 3 | 0.188 | 0.926 | 0.970 | 0.970 | 0.383 | 0.550 | 0.260 |
|   ablation: frozen R-GCN retriever alone | this work | 3 | 0.150 | 0.908 | 0.962 | 0.962 | 0.368 | 0.532 | 0.244 |
|   ablation: re-ranker on the embedding-model retriever | this work | 3 | 0.191 | 0.905 | 0.959 | 0.955 | 0.323 | 0.466 | 0.159 |
|   ablation: embedding-model retriever alone | this work | 3 | 0.159 | 0.899 | 0.957 | 0.954 | 0.282 | 0.416 | 0.119 |
|   ablation: re-ranker + joint field tokens | this work | 3 | 0.191 | 0.905 | 0.959 | 0.955 | 0.321 | 0.472 | 0.155 |

*AUPR at network prevalence is primary; AUPR 1:3 and 1:1 re-weight negatives from the full ranking (the population version of sampled-negative evaluation). Deterministic baselines are single runs; learned models are averaged over seeds {42, 0, 1}. Bold = column best.*
