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
| Truncated SVD + taxonomic imputation | Strydom et al. 2022, MEE | 1 | 0.095 | 0.850 | 0.924 | 0.895 | 0.291 | **0.449** | 0.256 |
| ANTHEIA v1 spatial (PCA-15 + N, logistic) | Li, Cher & Jacobs 2026 | 1 | 0.048 | 0.741 | 0.868 | 0.839 | 0.117 | 0.256 | 0.087 |
| ANTHEIA v1 scalar (+ Delta) | Li, Cher & Jacobs 2026 | 1 | 0.052 | 0.751 | 0.875 | 0.849 | 0.118 | 0.260 | 0.089 |
| Gradient boosting on pair features | Pichler et al. 2020, MEE | 1 | 0.079 | 0.857 | 0.936 | 0.930 | 0.290 | 0.442 | **0.276** |
| *Published architectures, re-implemented* | | |  |  |  |  |  |  |  |
| TabICL (in-context tabular) | Qu et al. 2025 | 1 | 0.110 | 0.826 | 0.915 | 0.897 | 0.313 | 0.437 | 0.168 |
| *Ours* | | |  |  |  |  |  |  |  |
| Boosted ranker: taxonomy + spatial + per-cell | this work | 1 | 0.102 | 0.806 | 0.904 | 0.881 | 0.320 | 0.426 | 0.122 |
| Genus-routed experts | this work | 1 | 0.103 | 0.810 | 0.906 | 0.883 | **0.336** | 0.447 | 0.256 |
| Embedding model (identity + field blocks) | this work | 3 | 0.159 | 0.899 | 0.957 | 0.954 | 0.282 | 0.416 | 0.119 |

*AUPR at network prevalence is primary; AUPR 1:3 and 1:1 re-weight negatives from the full ranking (the population version of sampled-negative evaluation). Deterministic baselines are single runs; learned models are averaged over seeds {42, 0, 1}. Bold = column best.*
