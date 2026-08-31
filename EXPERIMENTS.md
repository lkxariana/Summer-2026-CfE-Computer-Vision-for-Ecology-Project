# Experiment Log

All experiments run on the frozen protocol unless noted: `edges_v1` (62,832 orientation-corrected GloBI pairs), degree-stratified plant split 75/10/15 (`artifacts/split_v1.json`, 553 test plants / 9,210 test positives), leave-plant-out, ranking over all 24,939 pollinators, bootstrap-over-plants CIs. Reference numbers to beat: **raw N recall@10 = 0.117, hit@10 = 0.505; best pooled PR-AUC 0.657**.

| # | Date | Experiment | Question | Result | Artifact |
|---|---|---|---|---|---|
| 1 | 08-31 | Split-stability verification | Need k-fold, or is one frozen plant split enough? | Between-holdout std (PR ±0.009) < within-split bootstrap std (±0.012); 3-fold ≈ 5-fold ≈ holdouts. Single frozen split + bootstrap justified. | `verify_splits.csv` |
| 2 | 08-31 | Corrected baselines v1 (classifiers) | How do the five ANTHEIA feature sets fare on the honest benchmark? | scalar best PR 0.652 [0.624–0.672] > spatial 0.636 > V_δ/PMf ≈ 0.633 (add nothing) >> nulls 0.50–0.53. BUT N-only wins ranking (R@10 0.117 vs ~0.074). Old 0.95–0.96 numbers were leakage/shortcut artifacts. | `baselines_v1.csv` |
| 3 | 08-31 | Stage A pairwise rankers | Does matching the training objective (within-plant ranking loss) close the ranking gap? | No. rank_n reproduces N-only exactly (sanity ✓); all learned combos still lose top-10 (0.076–0.081 vs 0.117). Features, not loss, are the bottleneck. | `ranker_v1.csv` |
| 4 | 08-31 | Co-occurrence slice | Is N's win just the co-occurrence gate? | No. Within N>0 candidates (median 5,060/plant; 96% of true partners co-occur), raw N intensity still wins (0.114 vs 0.079–0.085). | in log |
| 5 | 08-31 | Local per-bin Δ | Does spatially-local phenological overlap beat the range-averaged Δ? | Genuinely new signal (corr 0.837, best pooled PR 0.657 vs 0.652) and less top-k damage next to N (0.088 vs 0.081) — but N alone still wins top-10. Plant-side timing is not the binding constraint. | `local_delta_v1.csv` |
| 6 | 08-31 | GloBI provenance scan | How iNat-dependent are the labels? | 71.3% of records iNat; 23.2% of edges have ≥1 curated record (USGS bees, Guzmán 2022, Web of Life, CropPol); 9.1% multi-record + curated (premium tier). | `edge_provenance_v1.csv` |
| 7 | 08-31 | Evidence-tier 2×2 | Does N's dominance survive on curated (non-iNat) labels? Train-all→eval-Tier1 = independence test. | RUNNING | `tiers_v1.csv` |
| 8 | 08-31 | Taxonomy affinity + hybrids + segments | Does "who" (pollinator × plant-genus/family affinity from the training matrix) crack top-10? Do N-first hybrids beat pure N? Where does N fail? | QUEUED | `taxonomy_v1.csv` |
| 9 | 08-31 | GBM contextual ranker | Is the linear probe the ceiling? Trees over all features (incl. taxonomy) with feature interactions. | QUEUED | `gbm_v1.csv` |

Backlog: SDM a_curves swap (blocked on full SDM, Sept); temporal-holdout validation (GloBI eventDate); paired-bootstrap significance for scalar_local vs spatial; TaxaBind/SINR species embeddings as tower inputs (2× RTX 4090 available); two-tower Stage B.
