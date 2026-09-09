## Table 1 — link prediction across regimes (validation). AUPR at network prevalence, AUROC, normalised recall@10.

| Method | cold plant: AUPR | cold plant: AUROC | cold plant: nR@10 | cold pollinator: AUPR | cold pollinator: AUROC | cold pollinator: nR@10 | cold both: AUPR | cold both: AUROC | cold both: nR@10 | warm: AUPR | warm: AUROC | warm: nR@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| *Nulls* | | | | | | | | | | | | |
| Pollinator popularity | 0.053 | 0.926 | 0.244 | 0.002 | 0.500 | 0.009 | 0.003 | 0.500 | 0.011 | 0.019 | 0.936 | 0.183 |
| Co-occurrence N | 0.026 | 0.709 | 0.105 | 0.021 | 0.728 | 0.199 | 0.020 | 0.691 | 0.218 | 0.004 | 0.719 | 0.093 |
| *Ecological baselines* | | | | | | | | | | | | |
| Phenology x abundance | 0.024 | 0.822 | 0.158 | 0.027 | 0.812 | 0.231 | 0.029 | 0.805 | 0.243 | 0.009 | 0.824 | 0.126 |
| Congeneric transfer | 0.057 | 0.876 | 0.293 | 0.002 | 0.500 | 0.009 | 0.003 | 0.500 | 0.011 | 0.016 | 0.857 | 0.235 |
| SVD + taxonomic imputation | 0.095 | 0.895 | 0.291 | 0.005 | 0.505 | 0.018 | 0.004 | 0.508 | 0.035 | **0.105** | 0.928 | 0.250 |
| Pair-feature GBM | 0.079 | 0.930 | 0.290 | 0.044 | 0.858 | 0.251 | 0.037 | 0.853 | 0.279 | 0.029 | 0.942 | 0.241 |
| NECTAR-style plausibility | 0.081 | 0.890 | 0.161 | 0.011 | 0.807 | 0.062 | 0.016 | 0.822 | 0.038 | 0.015 | 0.860 | 0.156 |
| *ANTHEIA v1 (prior work, this group)* | | | | | | | | | | | | |
| Co-occurrence PCA + N | 0.048 | 0.839 | 0.117 | 0.033 | 0.833 | 0.284 | 0.041 | 0.799 | 0.315 | 0.009 | 0.844 | 0.112 |
| Co-occurrence PCA + N + phenology overlap | 0.052 | 0.849 | 0.118 | 0.036 | 0.838 | 0.305 | 0.044 | 0.810 | **0.324** | 0.010 | 0.853 | 0.112 |
| *Published architectures* | | | | | | | | | | | | |
| Two-tower retrieval | 0.099 | 0.797 | 0.277 | 0.006 | 0.596 | 0.031 | 0.007 | 0.543 | 0.029 | 0.019 | 0.830 | 0.233 |
| Wide & Deep | 0.144 | 0.922 | 0.306 | 0.029 | 0.817 | 0.188 | 0.026 | 0.817 | 0.165 | 0.016 | 0.927 | 0.208 |
| DCN-V2 | 0.138 | 0.945 | 0.263 | 0.007 | 0.711 | 0.075 | 0.008 | 0.674 | 0.047 | 0.012 | 0.960 | 0.158 |
| *Ours* | | | | | | | | | | | | |
| R-GCN (plant-side leave-own-edges-out) | 0.150 | 0.962 | 0.368 | 0.006 | 0.761 | 0.166 | 0.010 | 0.766 | 0.158 | 0.032 | 0.969 | 0.257 |
|   + identity re-ranker | 0.188 | 0.970 | 0.383 | 0.008 | 0.803 | 0.193 | 0.029 | 0.841 | 0.212 | 0.043 | 0.971 | 0.267 |
| R-GCN (symmetric leave-own-edges-out) | 0.170 | 0.966 | 0.375 | 0.101 | 0.906 | 0.352 | 0.073 | 0.872 | 0.291 | 0.034 | 0.975 | 0.276 |
|   + identity re-ranker (**final**) | **0.209** | **0.972** | **0.383** | **0.117** | **0.915** | **0.361** | **0.094** | **0.886** | 0.288 | 0.050 | **0.976** | **0.298** |

Chance AUPR: cold plant 0.0013, cold pollinator 0.0023, cold both 0.0029, warm 0.0003. Learned models: mean of seeds {42, 0, 1} where available; deterministic baselines single run. † = run predates the pollinator-side protocol fix (being re-run). Bold = column best. Re-expressed AUPR (1:3, 1:1), nR@50 and strata: appendix.

## Table 2 — within-site network completion (91 surveyed networks; every local pair removed from training; chance AUPR = connectance 0.135)

| Method | mean AUPR | mean AUROC | precision@L | NODF (obs 36) | AUPR warm plants | AUPR cold plants |
|---|---:|---:|---:|---:|---:|---:|
| *Nulls* | | | | | | |
| Pollinator popularity | 0.192 [0.179, 0.207] | 0.582 | 0.209 | 92 | 0.194 | 0.205 |
| Co-occurrence N | 0.167 [0.153, 0.181] | 0.540 | 0.181 | 93 | 0.169 | 0.102 |
| *Ecological baselines* | | | | | | |
| Phenology x abundance | 0.184 [0.169, 0.200] | 0.563 | 0.180 | 95 | 0.186 | 0.166 |
| Congeneric transfer | **0.222** [0.205, 0.242] | 0.624 | **0.239** | 70 | **0.223** | **0.260** |
| SVD + taxonomic imputation | 0.179 [0.166, 0.192] | 0.586 | 0.199 | 76 | 0.180 | 0.244 |
| Pair-feature GBM | 0.203 [0.187, 0.219] | 0.596 | 0.218 | 83 | 0.205 | 0.209 |
| NECTAR-style plausibility | 0.190 [0.176, 0.204] | 0.591 | 0.215 | 73 | 0.191 | 0.242 |
| *ANTHEIA v1 (prior work, this group)* | | | | | | |
| Co-occurrence PCA + N | 0.178 [0.164, 0.193] | 0.555 | 0.184 | 100 | 0.179 | 0.164 |
| Co-occurrence PCA + N + phenology overlap | 0.179 [0.165, 0.194] | 0.561 | 0.187 | 100 | 0.181 | 0.169 |
| *Published architectures* | | | | | | |
| Wide & Deep | 0.153 | 0.533 | 0.146 | 44 | — | — |
| DCN-V2 | 0.166 | 0.576 | 0.167 | **43** | — | — |
| *Ours* | | | | | | |
| R-GCN (plant-side leave-own-edges-out) | 0.204 | 0.620 | 0.231 | 49 | 0.206 | 0.241 |
|   + identity re-ranker | 0.204 | 0.620 | 0.232 | 51 | — | — |
| R-GCN (symmetric leave-own-edges-out) | 0.215 | **0.629** | 0.234 | 52 | 0.217 | 0.226 |
|   + identity re-ranker (**final**) | 0.214 | 0.629 | 0.235 | 52 | — | — |

precision@L and NODF from the top-L pairs per network (L = observed links). Warm = plant keeps at least one edge outside the site; cold = none. Seeds averaged where several exist; single-seed rows show the bootstrap CI over networks.
