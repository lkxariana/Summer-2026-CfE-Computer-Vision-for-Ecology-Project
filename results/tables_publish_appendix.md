## Table A1 — cold-plant validation by stratum (nR@10 unless noted; seeds averaged)

| Method | low-degree plants (<= 2 partners): nR@10 / AUPR | other plants: nR@10 | unseen-genus: nR@10 / AUPR | zero-shot (text-imputed features): nR@10 |
|---|---:|---:|---:|---:|
| Pollinator popularity | 0.233 / 0.022 | 0.251 | 0.234 / 0.038 | 0.278 |
| Co-occurrence N | 0.063 / 0.001 | 0.130 | 0.064 / 0.005 | 0.000 |
| Phenology x abundance | 0.123 / 0.008 | 0.178 | 0.107 / 0.015 | 0.164 |
| Congeneric transfer | 0.265 / 0.016 | 0.310 | 0.213 / 0.020 | 0.321 |
| SVD + taxonomic imputation | 0.249 / 0.019 | 0.317 | 0.256 / 0.057 | 0.319 |
| Pair-feature GBM | 0.267 / 0.030 | 0.303 | 0.276 / 0.061 | 0.284 |
| NECTAR-style plausibility | 0.119 / 0.005 | 0.185 | 0.003 / 0.002 | 0.143 |
| Co-occurrence PCA + N | 0.091 / 0.007 | 0.133 | 0.087 / 0.015 | 0.123 |
| Co-occurrence PCA + N + phenology overlap | 0.091 / 0.007 | 0.134 | 0.089 / 0.016 | 0.127 |
| Two-tower retrieval | 0.235 / 0.012 | 0.302 | 0.064 / 0.001 | 0.276 |
| Wide & Deep | 0.243 / 0.028 | 0.343 | 0.164 / 0.039 | 0.258 |
| DCN-V2 | 0.196 / 0.017 | 0.302 | 0.107 / 0.026 | 0.199 |
| R-GCN, taxon + cell nodes (plant-side leave-own-edges-out) | 0.338 / 0.037 | 0.386 | 0.244 / 0.056 | 0.338 |
| + identity re-ranker | 0.356 / 0.037 | 0.398 | 0.260 / 0.069 | 0.369 |
| R-GCN, taxon + cell nodes (symmetric leave-own-edges-out) | 0.334 / 0.042 | 0.399 | 0.236 / 0.044 | 0.353 |
| + identity re-ranker | 0.350 / 0.041 | 0.402 | 0.245 / 0.051 | 0.361 |
| R-GCN, species + cell nodes (symmetric leave-own-edges-out) | 0.349 / 0.045 | 0.393 | 0.242 / 0.076 | 0.361 |
| + identity re-ranker (**final**) | 0.344 / 0.043 | 0.396 | 0.262 / 0.078 | 0.366 |
| variant: retriever + explicit co-presence (opportunity) term | 0.368 / 0.059 | 0.403 | 0.304 / 0.099 | 0.376 |
| variant: presence-embedding retriever + re-ranker | 0.306 / 0.041 | 0.388 | 0.203 / 0.043 | 0.342 |

## Table A2 — within-site mean AUPR by survey source (seeds averaged)

| Method | Guzman et al. 2022 (British Columbia) (n=71) | LaManna et al. (Missouri) (n=1) | web-of-life (n=19) |
|---|---:|---:|---:|
| Pollinator popularity | 0.186 | 0.056 | 0.222 |
| Co-occurrence N | 0.163 | 0.058 | 0.189 |
| Phenology x abundance | 0.176 | 0.048 | 0.222 |
| Congeneric transfer | 0.221 | 0.074 | 0.236 |
| SVD + taxonomic imputation | 0.174 | 0.066 | 0.202 |
| Pair-feature GBM | 0.196 | 0.063 | 0.236 |
| NECTAR-style plausibility | 0.185 | 0.071 | 0.215 |
| Co-occurrence PCA + N | 0.171 | 0.061 | 0.210 |
| Co-occurrence PCA + N + phenology overlap | 0.172 | 0.063 | 0.213 |
| Wide & Deep | 0.146 | 0.055 | 0.185 |
| DCN-V2 | 0.161 | 0.064 | 0.191 |
| R-GCN, taxon + cell nodes (plant-side leave-own-edges-out) | 0.203 | 0.078 | 0.215 |
| + identity re-ranker | 0.202 | 0.081 | 0.216 |
| R-GCN, taxon + cell nodes (symmetric leave-own-edges-out) | 0.212 | 0.082 | 0.233 |
| + identity re-ranker | 0.209 | 0.086 | 0.236 |
| R-GCN, species + cell nodes (symmetric leave-own-edges-out) | 0.216 | 0.082 | 0.235 |
| + identity re-ranker (**final**) | 0.212 | 0.086 | 0.235 |
| variant: retriever + explicit co-presence (opportunity) term | 0.211 | 0.082 | 0.236 |
| variant: same, opportunity term off within sites | 0.222 | 0.087 | 0.240 |
| variant: presence-embedding retriever + re-ranker | 0.189 | 0.080 | 0.224 |
