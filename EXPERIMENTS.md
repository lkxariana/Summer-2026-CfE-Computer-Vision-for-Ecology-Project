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
| 7 | 08-31 | Evidence-tier 2×2 | Does N's dominance survive on curated (non-iNat) labels? Train-all→eval-Tier1 = independence test. | **N is process-invariant: R@10 0.1162 on curated labels vs 0.1171 on all** (222 test plants w/ T1 partners). Learned models collapse out-of-process (train-all→T1: 0.011–0.013) but recover when trained on T1 (0.075–0.079) — the learned pollinator-side signal is largely *documentation-process* signal, not transferable ecology. N still beats everything within every cell. | `tiers_v1.csv` |
| 8 | 08-31 | Taxonomy affinity + hybrids + segments | Does "who" (pollinator × plant-genus/family affinity from the training matrix) crack top-10? | **YES — first features to beat N.** tax alone 0.179; N+tax 0.199 [0.179–0.221]; **N+tax+localΔ 0.209, hit@10 0.66** (vs N 0.117/0.50). Wins across all degree quartiles; works even for unseen genera via family (0.178 vs N's 0.124); once "who" is present, local Δ finally adds top-k value. Adding Vp still hurts (0.190). Lexicographic hybrids beat N (0.13–0.14) but lose to trained blends. | `taxonomy_v1.csv` |
| 9 | 08-31 | GBM contextual ranker | Is the linear probe the ceiling? | **YES — trees on geometry/phenology alone: 0.236 [0.212–0.259], R@50 0.49, hit@10 0.69** — best model so far, no taxonomy needed. Oddity: adding raw tax features *hurts* GBM (0.192) — likely overfit on high-cardinality counts; needs smoothing/tuning. | `gbm_v1.csv` |

| 10 | 08-31 | BioCLIP text embeddings | Build learned species representations from names (imageomics/bioclip text tower, GPU). | Built: 6,348 plants + 24,939 pollinators × 512D unit-norm (13s on RTX 4090). | `cache/bioclip_text_*.npy` |
| 11 | 08-31 | BioCLIP embedding features | Does plant↔pollinator name-embedding similarity (or PCA'd embeddings) add ranking signal? | **No — redundant with explicit taxonomy affinity.** embsim doesn't add to N (0.109) or to N+tax (0.196 vs 0.199); PCA'd embeddings hurt like Vp (0.179); GBM+emb 0.185 < gbm_geo 0.236. Text towers ≈ soft taxonomy; count-based affinity beats them. Non-name modalities (location/image encoders) remain the open representation angle. | `embeddings_v1.csv` |

| 12 | 08-31 | Tier-1 transfer of the winners | Do gbm_geo/taxonomy survive curated-label eval or collapse like Vp? | **They double: gbm_geo 0.470 [0.42–0.52] on Tier-1 (vs 0.236 all); N+tax 0.304 (vs 0.199); rank_n flat 0.116.** Taxonomy + geo/pheno interactions are ecology, not process; much of the all-label "error" is label noise. Eval reporting becomes two-row: all (conservative) + Tier-1 (skill). GBM val-tuning still running. | `transfer_v1.csv` |
| 12b | 08-31 | Val-tuned GBM | Select GBM hypers on val (12 configs, 2 specs). | Best: geo, 800 iters / lr .05 / leaf 100 / L2 5 (val 0.270) → **test 0.2454 [0.221–0.270]** (+0.010 over untuned). Smoothed taxonomy shares still hurt trees (val ~0.20). | `gbm_tune_v1.csv` |
| 15 | 08-31 | Two-tower + BioCLIP towers | Do foundation-model embeddings help as *tower inputs* (vs failed cosine features)? | **Yes: BioCLIP-1 towers 0.2612 [0.238–0.287], hit@10 0.736, T1 0.475 — new best.** BioCLIP-2 0.2512/0.457 (parity with v1-claim; the stronger model is not better here). Both beat no-emb tower (0.225) and tuned GBM (0.245). | `twotower_bioclip*.csv` |
| 13 | 08-31 | BioCLIP-2 text embeddings | Stronger encoder (TreeOfLife-200M) as NN tower input. | Built: 768-D for all 31,287 species (22s, GPU 0). | `cache/bioclip2_text_*.npy` |
| 14 | 08-31 | Two-tower wide&deep NN (v1, no emb) | Can a neural ranker (genus/family embeddings + curve towers + wide N/Δ/tax features, sampled softmax, val-selected) beat gbm_geo 0.236? | **Parity on first config: test 0.225 [0.202–0.250], hit@10 0.675, Tier-1 0.469 [0.42–0.52]** — matches GBM on both evals, transfers cleanly. Overfits by epoch ~2 (val peak 0.271) → regularization/negatives headroom. BioCLIP-2 and BioCLIP-1 tower variants queued next. | `twotower_none.csv` |

| 16 | 08-31 | **Final roster + paired significance** | One reproducible table: all models refit, per-plant scores saved, paired plant-level bootstrap. | See table below. **twotower_bioclip best (0.2612 / R@50 0.546 / hit@10 0.736 / T1 0.475)**; sig. > two-tower-none (+0.036, p=.001) and > N+tax+localΔ (+0.053, p<.001); **statistically tied with tuned GBM** (Δ0.016, p=.11). All learned models >> N (p<.001). BioCLIP-2 tied with BioCLIP-1. | `final_table_v1.csv`, `final_significance_v1.csv`, `final_scores/` |

### Final table (frozen split, 553 test plants; T1 = 222 curated-label plants)

| model | R@10 all [95% CI] | R@50 | hit@10 | R@10 Tier-1 |
|---|---|---|---|---|
| degree null | 0.080 [0.067–0.093] | 0.233 | 0.458 | 0.029 |
| N only | 0.115 [0.098–0.132] | 0.343 | 0.508 | 0.117 |
| N + taxonomy + local Δ (linear) | 0.209 [0.188–0.233] | 0.431 | 0.658 | 0.303 |
| GBM geo (val-tuned) | 0.245 [0.221–0.270] | 0.491 | 0.700 | 0.471 |
| two-tower (no emb) | 0.225 [0.202–0.250] | 0.483 | 0.675 | 0.469 |
| **two-tower + BioCLIP** | **0.261 [0.238–0.287]** | **0.546** | **0.736** | **0.475** |
| two-tower + BioCLIP-2 | 0.251 [0.229–0.275] | 0.545 | 0.727 | 0.457 |

| 17 | 08-31 | **Temporal holdout (prospective discovery)** | Train only on interactions documented ≤2020; rank the pairs *first documented 2021+* for known plants, with already-known partners masked. | **Works — the ordering holds prospectively on 2,179 plants:** N 0.104, N+tax+localΔ 0.184, **GBM 0.2146 [0.204–0.226], hit@10 0.70**, two-tower+BioCLIP 0.2074 (under-trained: 2 epochs, no early stopping — rerun with val carve-out in progress). Models trained on 2020 knowledge put a later-documented partner in the top 10 for 70% of plants. | `temporal_2020_v1.csv` |

| 18 | 08-31 | Error / segment / complementarity analysis | Where do models win and lose; is there ensemble headroom? | **(a) Generalization is fine: unseen-genus plants score as well as seen (0.265 vs 0.260)** — taxonomy features aren't memorizing. (b) Performance falls with plant degree (Q1 0.327 → Q4 0.129), largely a recall@10 ceiling artifact (a 40-partner plant caps at 0.25) → report degree-normalized recall. (c) Narrow-range plants easiest (0.314) vs wide (0.174). (d) **Ensemble headroom: GBM and two-tower disagree usefully (41 plants only-NN, 21 only-GBM); oracle hit@10 0.774 vs 0.736 best single.** (e) 88/553 plants missed by all models; 217 hit by all. | `error_analysis_v1.csv` |
| 19 | 08-31 | Rank-average ensemble + normalized metrics | Can combining GBM + two-tower realize the oracle headroom? | RUNNING | `ensemble_v1.csv` |

Backlog: SDM a_curves swap (blocked on full SDM, Sept); temporal-holdout validation (GloBI eventDate); paired-bootstrap significance table; TaxaBind encoders (verify HF model ids first).

## Paper story (working sketch, updated 08-31 evening)

1. **Data**: orientation-corrected GloBI CONUS plant-pollinator network — 62,832 edges (450× the prior extraction) with per-edge provenance tiers. Dataset contribution.
2. **Benchmark**: cold-start (leave-plant-out) full-universe ranking + nulls; demonstration that prior-style evaluation (random pair split, uniform negatives) is solved by shortcuts (N-only 0.99 ROC) and cannot rank architectures.
3. **Finding A — co-occurrence intensity is a documentation-process-invariant super-baseline** (R@10 0.117 on all labels, 0.116 on curated). The field should be required to report it.
4. **Finding B — process leakage in learned features**: dense pollinator representations collapse 6× when evaluated across documentation processes (all→Tier-1). First feature-level quantification of the iNat-circularity concern.
5. **Finding C — what beats the baseline**: "who" (taxonomic affinity, +78%), feature interactions (GBM on geometry/phenology, 2×), timing (local Δ) once "who" is present; embeddings-of-names redundant with taxonomy. Pending: tuned GBM, two-tower NN, tier transfer of the winners; September: SDM a_curves as the pollinator-side upgrade.
6. **Finding D — prospective validation**: models trained on pre-2021 documentation rank later-documented interactions (hit@10 0.70 on 2,179 plants), so top-ranked "false positives" behave like discovery candidates rather than errors.
7. **Generalization**: performance holds on plants whose *genus* never appeared in training (0.265 vs 0.260), and is strongest for narrow-range, low-degree plants — the specialists that matter most for conservation.
8. **Application**: ranked candidate pollinators for unseen plants (hit@10 ≈ 0.74), tier- and time-validated; downstream brittleness scoring.

Ops note (08-31): tier 2×2 initially never launched — its wait-loop pgrep pattern matched the wrapper's own cmdline (self-match). Killed, relaunched directly (~1h lost). Lesson: chain on artifact files, not process names.
