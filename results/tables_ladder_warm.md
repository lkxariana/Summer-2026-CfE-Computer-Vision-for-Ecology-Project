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
| Co-occurrence PCA-15 + N (roadmap proof-of-concept; ANTHEIA v1) | Strydom et al. 2021, Phil Trans R Soc B; Li, Cher & Jacobs 2026 | 1 (pre-fix) | 0.010 | 0.735 | 0.867 | 0.844 | 0.112 | 0.283 | 0.273 |
| Co-occurrence PCA-15 + N + Delta (ANTHEIA v1 scalar) | Li, Cher & Jacobs 2026 | 1 (pre-fix) | 0.010 | 0.745 | 0.874 | 0.853 | 0.113 | 0.300 | 0.273 |
| Gradient boosting on pair features | Pichler et al. 2020, MEE | 1 (pre-fix) | 0.027 | 0.882 | 0.948 | 0.943 | 0.224 | 0.455 | 0.455 |
| Spatial x phenological overlap product | Baiotto et al. 2026 (bioRxiv), Eq. 1 | 1 | 0.001 | 0.585 | 0.795 | 0.808 | 0.004 | 0.049 | 0.000 |
| NECTAR-style plausibility (genus constraint x overlap product) | Baiotto et al. 2026 (bioRxiv) | 1 | 0.015 | 0.759 | 0.880 | 0.860 | 0.156 | 0.319 | 0.000 |
| *Published architectures, re-implemented* | | |  |  |  |  |  |  |  |
| Two-tower retrieval (sampled softmax, logQ) | Yi et al. 2019, RecSys | 1 (pre-fix) | 0.021 | 0.728 | 0.858 | 0.827 | 0.233 | 0.373 | 0.182 |
| Pair MLP (NCF-style) | He et al. 2017, WWW | 1 (pre-fix) | 0.017 | 0.678 | 0.824 | 0.783 | 0.225 | 0.371 | 0.000 |
| Wide & Deep | Cheng et al. 2016 | 1 (pre-fix) | 0.016 | 0.857 | 0.935 | 0.927 | 0.208 | 0.442 | 0.182 |
| DCN-V2 (2 cross layers) | Wang et al. 2021, WWW | 1 (pre-fix) | 0.012 | 0.882 | 0.953 | 0.960 | 0.158 | 0.386 | 0.182 |
| *Hand-engineered features (our earlier system)* | | |  |  |  |  |  |  |  |
| Boosted ranker: taxonomy + spatial + per-cell features | this work, v2 | 1 (pre-fix) | 0.027 | 0.797 | 0.898 | 0.871 | 0.246 | 0.401 | 0.364 |
| Genus-routed experts (booster / SVD by genus) | this work, v2 | 1 (pre-fix) | 0.027 | 0.797 | 0.898 | 0.871 | 0.247 | 0.402 | 0.455 |
| *Ours: graph retriever + re-ranker* | | |  |  |  |  |  |  |  |
| **System v2: R-GCN (symmetric leave-own-edges-out, R3) + identity re-ranker** | this work | 1 (incomplete) | 0.048 | **0.942** | **0.976** | **0.976** | **0.300** | **0.553** | 0.455 |
|   ablation: R3 retriever alone | this work | 1 (incomplete) (pre-fix) | 0.033 | 0.936 | 0.974 | 0.975 | 0.270 | 0.524 | 0.455 |
| System v1: R-GCN (plant-side leave-own-edges-out) + identity re-ranker | this work | 2 (incomplete) (pre-fix) | 0.031 | 0.928 | 0.970 | 0.970 | 0.199 | 0.487 | 0.455 |
|   ablation: frozen R-GCN retriever alone | this work | 1 (incomplete) | 0.032 | 0.929 | 0.971 | 0.969 | 0.257 | 0.528 | **0.545** |
|   ablation: embedding-model retriever alone | this work | 1 (incomplete) (pre-fix) | 0.015 | 0.897 | 0.959 | 0.962 | 0.171 | 0.404 | 0.273 |

*AUPR at network prevalence is primary; AUPR 1:3 and 1:1 re-weight negatives from the full ranking (the population version of sampled-negative evaluation). Deterministic baselines are single runs; learned models are averaged over seeds {42, 0, 1}. Bold = column best.*
