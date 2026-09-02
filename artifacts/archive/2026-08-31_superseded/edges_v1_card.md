# Edge list edges_v1

Built 2026-08-31 from `/scratch/ariana.l/Stage 4 Link Prediction Model/stage4_globi_conus_broad.csv`.
Coverage universe: 6348 plants x 24939 pollinators (f_curves source: `f_curves_corrected.csv`).
Orientation corrected by species-set membership with binomial-name fallback; see `src/antheia/globi.py`.

| stat | value |
|---|---|
| records_in | 663,374 |
| records_as_is | 168 |
| records_swapped | 329,106 |
| records_dropped | 334,100 |
| dropped_genus_only_tgt | 208,492 |
| unique_edges | 62,832 |
| n_plants | 3,689 |
| n_pollinators | 4,302 |
| connectance | 0.3959% |
| plant degree median / mean / max | 6 / 17.0 / 516 |
| edges with n_records >= 2 | 27,010 |

Top pollinators: Apis mellifera, Bombus impatiens, Bombus griseocollis, Bombus vosnesenskii, Danaus plexippus

Top plants: Achillea millefolium, Asclepias syriaca, Daucus carota, Pycnanthemum muticum, Leucanthemum vulgare
