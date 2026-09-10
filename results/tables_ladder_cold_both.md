## Table 2 — cold-both validation (cold_both/val), 249 plants x 1312 candidates, chance AUPR 0.00290

| Method | Reference | seeds | AUPR | AUPR 1:3 | AUPR 1:1 | AUROC | nR@10 | nR@50 | unseen-genus nR@10 |
|---|---|:---:|---:|---:|---:|---:|---:|---:|---:|
| *Nulls* | | |  |  |  |  |  |  |  |
| Pollinator popularity | Aiyappa et al. 2025, ICML | 1 | 0.003 | 0.300 | 0.541 | 0.500 | 0.011 | 0.061 | 0.000 |
| Co-occurrence count N | Blanchet, Cazelles & Gravel 2020, Ecol Lett | 1 | 0.020 | 0.539 | 0.735 | 0.691 | 0.218 | 0.408 | 0.157 |
| Abundance neutral model | Vázquez, Chacoff & Cagnolo 2009, Ecology | 1 | 0.028 | 0.633 | 0.814 | 0.798 | 0.190 | 0.426 | 0.151 |
| *Ecological baselines* | | |  |  |  |  |  |  |  |
| Phenology x abundance | Vizentin-Bugoni et al. 2014, Proc R Soc B | 1 | 0.029 | 0.653 | 0.824 | 0.805 | 0.243 | 0.440 | 0.294 |
| Congeneric transfer | cf. Strydom et al. 2022, MEE | 1 | 0.003 | 0.300 | 0.541 | 0.500 | 0.011 | 0.061 | 0.000 |
| Truncated SVD + taxonomic imputation | Strydom et al. 2022, MEE | 1 | 0.004 | 0.319 | 0.557 | 0.508 | 0.035 | 0.048 | 0.018 |
| Co-occurrence PCA-15 + N (roadmap proof-of-concept; ANTHEIA v1) | Strydom et al. 2021, Phil Trans R Soc B; Li, Cher & Jacobs 2026 | 1 | 0.041 | 0.664 | 0.827 | 0.799 | 0.315 | 0.540 | 0.415 |
| Co-occurrence PCA-15 + N + Delta (ANTHEIA v1 scalar) | Li, Cher & Jacobs 2026 | 1 | 0.044 | 0.677 | 0.835 | 0.810 | 0.324 | 0.561 | **0.460** |
| Gradient boosting on pair features | Pichler et al. 2020, MEE | 1 | 0.037 | 0.715 | 0.863 | 0.853 | 0.279 | 0.534 | 0.274 |
| Spatial x phenological overlap product | Baiotto et al. 2026 (bioRxiv), Eq. 1 | 1 | 0.016 | 0.601 | 0.806 | 0.822 | 0.038 | 0.280 | 0.026 |
| NECTAR-style plausibility (genus constraint x overlap product) | Baiotto et al. 2026 (bioRxiv) | 1 | 0.016 | 0.601 | 0.806 | 0.822 | 0.038 | 0.280 | 0.026 |
| *Published architectures, re-implemented* | | |  |  |  |  |  |  |  |
| Two-tower retrieval (sampled softmax, logQ) | Yi et al. 2019, RecSys | 1 | 0.007 | 0.332 | 0.582 | 0.543 | 0.029 | 0.090 | 0.071 |
| Pair MLP (NCF-style) | He et al. 2017, WWW | 1 | 0.004 | 0.292 | 0.551 | 0.558 | 0.045 | 0.159 | 0.036 |
| Wide & Deep | Cheng et al. 2016 | 1 | 0.026 | 0.631 | 0.818 | 0.817 | 0.165 | 0.376 | 0.133 |
| DCN-V2 (2 cross layers) | Wang et al. 2021, WWW | 1 | 0.008 | 0.415 | 0.667 | 0.674 | 0.047 | 0.146 | 0.000 |
| *Hand-engineered features (our earlier system)* | | |  |  |  |  |  |  |  |
| Boosted ranker: taxonomy + spatial + per-cell features | this work, v2 | 1 | 0.003 | 0.270 | 0.525 | 0.524 | 0.044 | 0.088 | 0.050 |
| Genus-routed experts (booster / SVD by genus) | this work, v2 | 1 | 0.003 | 0.267 | 0.521 | 0.507 | 0.017 | 0.091 | 0.000 |
| *Ours: graph retriever + re-ranker* | | |  |  |  |  |  |  |  |
| **System v3: species-node R-GCN (text + interaction edges, symmetric rehearsal, no taxon nodes) + identity re-ranker** | this work | 3 | **0.105** | **0.778** | **0.896** | **0.888** | **0.341** | **0.573** | 0.234 |
|   ablation: v3 retriever alone | this work | 3 | 0.095 | 0.748 | 0.879 | 0.868 | 0.336 | 0.567 | 0.208 |
| System v2: R-GCN (symmetric leave-own-edges-out, R3) + identity re-ranker | this work | 3 | 0.094 | 0.766 | 0.891 | 0.886 | 0.288 | 0.535 | 0.226 |
|   ablation: R3 retriever alone | this work | 3 | 0.073 | 0.730 | 0.873 | 0.872 | 0.291 | 0.506 | 0.151 |
| System v1: R-GCN (plant-side leave-own-edges-out) + identity re-ranker | this work | 3 | 0.029 | 0.664 | 0.839 | 0.841 | 0.212 | 0.456 | 0.184 |
|   ablation: frozen R-GCN retriever alone | this work | 3 | 0.010 | 0.487 | 0.732 | 0.766 | 0.158 | 0.339 | 0.104 |
|   ablation: embedding-model retriever alone | this work | 1 (incomplete) | 0.011 | 0.484 | 0.722 | 0.726 | 0.068 | 0.189 | 0.036 |

*AUPR at network prevalence is primary; AUPR 1:3 and 1:1 re-weight negatives from the full ranking (the population version of sampled-negative evaluation). Deterministic baselines are single runs; learned models are averaged over seeds {42, 0, 1}. Bold = column best.*
