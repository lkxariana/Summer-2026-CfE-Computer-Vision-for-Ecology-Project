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
| Co-occurrence PCA-15 + N (roadmap proof-of-concept; ANTHEIA v1) | Strydom et al. 2021, Phil Trans R Soc B; Li, Cher & Jacobs 2026 | 1 (pre-fix) | 0.042 | 0.664 | 0.826 | 0.798 | 0.312 | 0.545 | 0.486 |
| Co-occurrence PCA-15 + N + Delta (ANTHEIA v1 scalar) | Li, Cher & Jacobs 2026 | 1 (pre-fix) | 0.045 | 0.676 | 0.835 | 0.809 | **0.316** | **0.555** | **0.496** |
| Gradient boosting on pair features | Pichler et al. 2020, MEE | 1 (pre-fix) | 0.028 | **0.688** | **0.851** | **0.847** | 0.200 | 0.455 | 0.157 |
| Spatial x phenological overlap product | Baiotto et al. 2026 (bioRxiv), Eq. 1 | 1 | 0.016 | 0.601 | 0.806 | 0.822 | 0.038 | 0.280 | 0.026 |
| NECTAR-style plausibility (genus constraint x overlap product) | Baiotto et al. 2026 (bioRxiv) | 1 | 0.016 | 0.601 | 0.806 | 0.822 | 0.038 | 0.280 | 0.026 |
| *Published architectures, re-implemented* | | |  |  |  |  |  |  |  |
| Two-tower retrieval (sampled softmax, logQ) | Yi et al. 2019, RecSys | 1 (pre-fix) | 0.009 | 0.344 | 0.590 | 0.540 | 0.033 | 0.086 | 0.107 |
| Pair MLP (NCF-style) | He et al. 2017, WWW | 1 (pre-fix) | 0.004 | 0.312 | 0.572 | 0.569 | 0.030 | 0.147 | 0.036 |
| Wide & Deep | Cheng et al. 2016 | 1 (pre-fix) | 0.026 | 0.631 | 0.818 | 0.817 | 0.165 | 0.376 | 0.133 |
| DCN-V2 (2 cross layers) | Wang et al. 2021, WWW | 1 (pre-fix) | 0.008 | 0.415 | 0.667 | 0.674 | 0.047 | 0.146 | 0.000 |
| *Hand-engineered features (our earlier system)* | | |  |  |  |  |  |  |  |
| Boosted ranker: taxonomy + spatial + per-cell features | this work, v2 | 1 (pre-fix) | 0.004 | 0.289 | 0.548 | 0.555 | 0.038 | 0.155 | 0.071 |
| Genus-routed experts (booster / SVD by genus) | this work, v2 | 1 (pre-fix) | 0.004 | 0.289 | 0.548 | 0.553 | 0.035 | 0.142 | 0.018 |
| *Ours: graph retriever + re-ranker* | | |  |  |  |  |  |  |  |
| **System v2: R-GCN (symmetric leave-own-edges-out, R3) + identity re-ranker** | this work | 1 (incomplete) | **0.061** | 0.588 | 0.782 | 0.778 | 0.233 | 0.321 | 0.285 |
|   ablation: R3 retriever alone | this work | 1 (incomplete) (pre-fix) | 0.019 | 0.611 | 0.810 | 0.819 | 0.146 | 0.373 | 0.125 |
| System v1: R-GCN (plant-side leave-own-edges-out) + identity re-ranker | this work | 2 (incomplete) (pre-fix) | 0.030 | 0.619 | 0.806 | 0.800 | 0.204 | 0.372 | 0.169 |
|   ablation: frozen R-GCN retriever alone | this work | 2 (incomplete) (pre-fix) | 0.006 | 0.411 | 0.673 | 0.722 | 0.080 | 0.213 | 0.080 |
|   ablation: embedding-model retriever alone | this work | 1 (incomplete) (pre-fix) | 0.011 | 0.484 | 0.722 | 0.726 | 0.068 | 0.189 | 0.036 |

*AUPR at network prevalence is primary; AUPR 1:3 and 1:1 re-weight negatives from the full ranking (the population version of sampled-negative evaluation). Deterministic baselines are single runs; learned models are averaged over seeds {42, 0, 1}. Bold = column best.*
