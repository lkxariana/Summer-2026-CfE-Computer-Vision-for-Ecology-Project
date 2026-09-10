## Table 3 — local-network completion (91 surveyed networks; mean connectance 0.135 = chance AUPR; observed NODF 36.0; warm-plant connectance 0.136, cold-plant connectance 0.077 over 45 networks with cold plants)

| Method | Reference | seeds | mean AUPR | mean AUROC | precision@L | deg rho plants | NODF pred | AUPR warm plants | AUPR cold plants |
|---|---|:---:|---:|---:|---:|---:|---:|---:|---:|
| *Nulls* | | |  |  |  |  |  |  |  |
| Pollinator popularity | Aiyappa et al. 2025, ICML | 1 | 0.192 [0.179,0.207] | 0.582 | 0.209 | -0.020 | 92.3 | 0.194 | 0.205 |
| Co-occurrence count N | Blanchet, Cazelles & Gravel 2020, Ecol Lett | 1 | 0.167 [0.153,0.181] | 0.540 | 0.181 | 0.019 | 92.9 | 0.169 | 0.102 |
| Abundance neutral model | Vázquez, Chacoff & Cagnolo 2009, Ecology | 1 | 0.176 [0.162,0.191] | 0.559 | 0.176 | 0.033 | 92.2 | 0.178 | 0.164 |
| *Ecological baselines* | | |  |  |  |  |  |  |  |
| Phenology x abundance | Vizentin-Bugoni et al. 2014, Proc R Soc B | 1 | 0.184 [0.169,0.200] | 0.563 | 0.180 | 0.008 | 94.9 | 0.186 | 0.166 |
| Congeneric transfer | cf. Strydom et al. 2022, MEE | 1 | **0.222** [0.205,0.242] | 0.624 | 0.239 | 0.100 | 69.7 | **0.223** | 0.260 |
| Truncated SVD + taxonomic imputation | Strydom et al. 2022, MEE | 1 | 0.179 [0.166,0.192] | 0.586 | 0.199 | 0.133 | 76.0 | 0.180 | 0.244 |
| Co-occurrence PCA-15 + N (roadmap proof-of-concept; ANTHEIA v1) | Strydom et al. 2021, Phil Trans R Soc B; Li, Cher & Jacobs 2026 | 1 | 0.178 [0.164,0.193] | 0.555 | 0.184 | 0.004 | 99.9 | 0.179 | 0.164 |
| Co-occurrence PCA-15 + N + Delta (ANTHEIA v1 scalar) | Li, Cher & Jacobs 2026 | 1 | 0.179 [0.165,0.194] | 0.561 | 0.187 | 0.012 | 99.6 | 0.181 | 0.169 |
| Gradient boosting on pair features | Pichler et al. 2020, MEE | 1 | 0.203 [0.187,0.219] | 0.596 | 0.218 | 0.073 | 83.3 | 0.205 | 0.209 |
| Spatial x phenological overlap product | Baiotto et al. 2026 (bioRxiv), Eq. 1 | 1 | 0.167 [0.153,0.180] | 0.552 | 0.173 | 0.022 | 95.3 | 0.168 | 0.166 |
| NECTAR-style plausibility (genus constraint x overlap product) | Baiotto et al. 2026 (bioRxiv) | 1 | 0.190 [0.176,0.204] | 0.591 | 0.215 | 0.113 | 72.7 | 0.191 | 0.242 |
| *Published architectures, re-implemented* | | |  |  |  |  |  |  |  |
| Wide & Deep | Cheng et al. 2016 | 3 | 0.153 | 0.533 | 0.146 | 0.029 | 44.0 | — | — |
| DCN-V2 (2 cross layers) | Wang et al. 2021, WWW | 3 | 0.166 | 0.576 | 0.167 | 0.036 | **42.5** | — | — |
| *Hand-engineered features (our earlier system)* | | |  |  |  |  |  |  |  |
| Boosted ranker: taxonomy + spatial + per-cell features | this work, v2 | 1 | 0.222 [0.205,0.239] | 0.627 | 0.252 | **0.197** | 62.4 | 0.223 | **0.268** |
| Genus-routed experts (booster / SVD by genus) | this work, v2 | 1 | 0.222 [0.205,0.239] | 0.627 | **0.252** | 0.197 | 62.5 | 0.223 | 0.268 |
| *Ours: graph retriever + re-ranker* | | |  |  |  |  |  |  |  |
| **System v3: species-node R-GCN (text + interaction edges, symmetric rehearsal, no taxon nodes) + identity re-ranker** | this work | 3 | 0.216 | 0.631 | 0.241 | 0.085 | 52.4 | — | — |
|   ablation: v3 retriever alone | this work | 3 | 0.218 | **0.633** | 0.241 | 0.071 | 51.1 | 0.220 | 0.233 |
| System v2: R-GCN (symmetric leave-own-edges-out, R3) + identity re-ranker | this work | 3 | 0.214 | 0.629 | 0.235 | 0.108 | 51.8 | — | — |
|   ablation: R3 retriever alone | this work | 3 | 0.215 | 0.629 | 0.234 | 0.052 | 51.6 | 0.217 | 0.226 |
|   ablation: R3 without cell x month nodes | this work | 1 | 0.212 [0.196,0.229] | 0.629 | 0.240 | 0.050 | 49.3 | 0.213 | 0.247 |
|   ablation: R3 with month-collapsed cells | this work | 1 | 0.210 [0.194,0.226] | 0.626 | 0.233 | 0.065 | 51.6 | 0.211 | 0.249 |
| System v1: R-GCN (plant-side leave-own-edges-out) + identity re-ranker | this work | 3 | 0.204 | 0.620 | 0.232 | 0.128 | 51.1 | — | — |
|   ablation: frozen R-GCN retriever alone | this work | 3 | 0.204 | 0.620 | 0.231 | 0.113 | 48.8 | 0.206 | 0.241 |
|   ablation: re-ranker on the embedding-model retriever | this work | 1 | 0.164 [0.151,0.177] | 0.572 | 0.159 | 0.009 | 45.6 | — | — |
|   ablation: embedding-model retriever alone | this work | 3 | 0.166 | 0.570 | 0.168 | 0.029 | 42.5 | 0.165 | 0.257 |
|   ablation: re-ranker + joint field tokens | this work | 1 | 0.165 [0.152,0.178] | 0.574 | 0.160 | 0.008 | 45.5 | — | — |

*Each network's plants x pollinators block is scored with every local-network pair removed from training. An absent pair among surveyed species is an observed non-interaction. precision@L and NODF use the top-L pairs, L = observed links; NODF closest to observed is bold. A plant is warm if any of its edges survives the removal, cold otherwise. Bootstrap CI over networks. Pooled metrics, lift and pollinator-degree correlation: appendix.*
