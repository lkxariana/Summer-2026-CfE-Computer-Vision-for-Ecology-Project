## Table 2 — warm validation (warm/val), 1923 plants x 9843 candidates, chance AUPR 0.00029

| Method | Reference | seeds | AUPR | AUPR 1:3 | AUPR 1:1 | AUROC | nR@10 | nR@50 | unseen-genus nR@10 |
|---|---|:---:|---:|---:|---:|---:|---:|---:|---:|
| *Nulls* | | |  |  |  |  |  |  |  |
| Pollinator popularity | Aiyappa et al. 2025, ICML | 1 | 0.019 | 0.861 | 0.939 | 0.936 | 0.183 | 0.385 | 0.455 |
| Co-occurrence count N | Blanchet, Cazelles & Gravel 2020, Ecol Lett | 1 | 0.004 | 0.593 | 0.774 | 0.719 | 0.093 | 0.210 | 0.091 |
| Abundance neutral model | Vázquez, Chacoff & Cagnolo 2009, Ecology | 1 | 0.009 | 0.698 | 0.846 | 0.819 | 0.123 | 0.238 | **0.545** |
| *Ecological baselines* | | |  |  |  |  |  |  |  |
| Phenology x abundance | Vizentin-Bugoni et al. 2014, Proc R Soc B | 1 | 0.009 | 0.709 | 0.852 | 0.824 | 0.126 | 0.243 | 0.455 |
| Congeneric transfer | cf. Strydom et al. 2022, MEE | 1 | 0.016 | 0.791 | 0.891 | 0.857 | 0.235 | 0.429 | 0.364 |
| Truncated SVD + taxonomic imputation | Strydom et al. 2022, MEE | 1 | **0.105** | 0.897 | 0.949 | 0.928 | 0.250 | 0.484 | 0.455 |
| Co-occurrence PCA-15 + N (roadmap proof-of-concept; ANTHEIA v1) | Strydom et al. 2021, Phil Trans R Soc B; Li, Cher & Jacobs 2026 | 1 | 0.009 | 0.734 | 0.867 | 0.844 | 0.112 | 0.275 | 0.273 |
| Co-occurrence PCA-15 + N + Delta (ANTHEIA v1 scalar) | Li, Cher & Jacobs 2026 | 1 | 0.010 | 0.744 | 0.874 | 0.853 | 0.112 | 0.294 | 0.273 |
| Gradient boosting on pair features | Pichler et al. 2020, MEE | 1 | 0.029 | 0.881 | 0.948 | 0.942 | 0.241 | 0.461 | 0.455 |
| Spatial x phenological overlap product | Baiotto et al. 2026 (bioRxiv), Eq. 1 | 1 | 0.001 | 0.585 | 0.795 | 0.808 | 0.004 | 0.049 | 0.000 |
| NECTAR-style plausibility (genus constraint x overlap product) | Baiotto et al. 2026 (bioRxiv) | 1 | 0.015 | 0.759 | 0.880 | 0.860 | 0.156 | 0.319 | 0.000 |
| *Published architectures, re-implemented* | | |  |  |  |  |  |  |  |
| Two-tower retrieval (sampled softmax, logQ) | Yi et al. 2019, RecSys | 1 | 0.019 | 0.729 | 0.859 | 0.830 | 0.233 | 0.376 | 0.182 |
| Pair MLP (NCF-style) | He et al. 2017, WWW | 1 | 0.017 | 0.683 | 0.827 | 0.788 | 0.220 | 0.368 | 0.000 |
| Wide & Deep | Cheng et al. 2016 | 1 | 0.016 | 0.857 | 0.935 | 0.927 | 0.208 | 0.442 | 0.182 |
| DCN-V2 (2 cross layers) | Wang et al. 2021, WWW | 1 | 0.012 | 0.882 | 0.953 | 0.960 | 0.158 | 0.386 | 0.182 |
| *Hand-engineered features (our earlier system)* | | |  |  |  |  |  |  |  |
| Boosted ranker: taxonomy + spatial + per-cell features | this work, v2 | 1 | 0.026 | 0.793 | 0.897 | 0.873 | 0.253 | 0.405 | 0.182 |
| Genus-routed experts (booster / SVD by genus) | this work, v2 | 1 | 0.026 | 0.801 | 0.902 | 0.880 | 0.258 | 0.407 | 0.455 |
| *Ours: graph retriever + re-ranker* | | |  |  |  |  |  |  |  |
| **System v3: species-node R-GCN (text + interaction edges, symmetric rehearsal, no taxon nodes) + identity re-ranker** | this work | 3 | 0.051 | 0.941 | 0.976 | 0.974 | **0.304** | **0.568** | 0.455 |
|   ablation: v3 retriever alone | this work | 3 | 0.032 | 0.934 | 0.973 | 0.974 | 0.280 | 0.538 | 0.455 |
|   factorised retriever + identity re-ranker | this work | 1 (incomplete) | 0.050 | 0.941 | 0.976 | 0.975 | 0.291 | 0.556 | 0.455 |
|   variant: presence-embedding retriever + re-ranker | this work | 1 (incomplete) | 0.048 | 0.939 | 0.975 | 0.974 | 0.286 | 0.551 | **0.545** |
|   ablation: v3 system without cell x month nodes | this work | 1 (incomplete) | 0.046 | 0.937 | 0.974 | 0.973 | 0.273 | 0.542 | 0.455 |
| System v2: R-GCN (symmetric leave-own-edges-out, R3) + identity re-ranker | this work | 3 | 0.050 | **0.942** | **0.976** | **0.976** | 0.298 | 0.550 | 0.455 |
|   ablation: R3 retriever alone | this work | 3 | 0.034 | 0.935 | 0.974 | 0.975 | 0.276 | 0.526 | 0.424 |
| System v1: R-GCN (plant-side leave-own-edges-out) + identity re-ranker | this work | 3 | 0.043 | 0.936 | 0.973 | 0.971 | 0.267 | 0.538 | 0.455 |
|   ablation: frozen R-GCN retriever alone | this work | 3 | 0.032 | 0.929 | 0.970 | 0.969 | 0.257 | 0.522 | 0.455 |
|   ablation: embedding-model retriever alone | this work | 1 (incomplete) | 0.015 | 0.897 | 0.959 | 0.962 | 0.171 | 0.404 | 0.273 |

*AUPR at network prevalence is primary; AUPR 1:3 and 1:1 re-weight negatives from the full ranking (the population version of sampled-negative evaluation). Deterministic baselines are single runs; learned models are averaged over seeds {42, 0, 1}. Bold = column best.*
