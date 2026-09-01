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
| **rank-average ensemble (GBM + BioCLIP tower)** | **0.277 [0.251–0.302]** | 0.552 | 0.741 | **0.504** |

Prospective (train ≤2020, rank pairs first documented 2021+, 2,179 plants): N 0.104 · linear 0.184 · GBM 0.215 · **two-tower+BioCLIP 0.244, hit@10 0.729**.

| 17 | 08-31 | **Temporal holdout (prospective discovery)** | Train only on interactions documented ≤2020; rank the pairs *first documented 2021+* for known plants, with already-known partners masked. | **Works — the ordering holds prospectively on 2,179 plants:** N 0.104, N+tax+localΔ 0.184, **GBM 0.2146 [0.204–0.226], hit@10 0.70**, two-tower+BioCLIP 0.2074 (under-trained: 2 epochs, no early stopping — rerun with val carve-out in progress). Models trained on 2020 knowledge put a later-documented partner in the top 10 for 70% of plants. | `temporal_2020_v1.csv` |

| 18 | 08-31 | Error / segment / complementarity analysis | Where do models win and lose; is there ensemble headroom? | **(a) Generalization is fine: unseen-genus plants score as well as seen (0.265 vs 0.260)** — taxonomy features aren't memorizing. (b) Performance falls with plant degree (Q1 0.327 → Q4 0.129), largely a recall@10 ceiling artifact (a 40-partner plant caps at 0.25) → report degree-normalized recall. (c) Narrow-range plants easiest (0.314) vs wide (0.174). (d) **Ensemble headroom: GBM and two-tower disagree usefully (41 plants only-NN, 21 only-GBM); oracle hit@10 0.774 vs 0.736 best single.** (e) 88/553 plants missed by all models; 217 hit by all. | `error_analysis_v1.csv` |
| 19 | 08-31 | Rank-average ensemble + normalized metrics | Can combining GBM + two-tower realize the oracle headroom? | **Yes, partially: rank-average 0.2771 [0.251–0.302], hit@10 0.741, normalized R@10 0.394, and Tier-1 0.504 (first model past 0.50).** Δ vs best single +0.016, p=0.054 — suggestive, not significant. Score-z fusion is worse (0.253): rank-space fusion matters. | `ensemble_v1.csv` |
| 17b | 08-31 | Temporal holdout, NN retrained | Fair prospective comparison (early stopping on held-out pre-2020 plants instead of 2 fixed epochs). | **Ordering reverses: two-tower+BioCLIP 0.2435 [0.232–0.256], hit@10 0.729 > GBM 0.2146.** The neural model is the best prospective discoverer; the earlier 0.207 was purely under-training. | `temporal_2020_v1.csv` |

| 20 | 09-01 | **Provenance probe (mechanism for the leakage finding)** | Among edges that are *all real interactions*, can each feature family predict WHO documented it (iNat vs curated)? 0.50 = process-blind. | **Yes, strongly — and inversely to transfer ability.** BioCLIP pollinator emb **0.914**, all-dense 0.925, Vp (pure occupancy, no taxonomy) **0.819**, N 0.681, Δ+localΔ 0.658, range 0.655, **taxonomy affinity 0.621 (lowest)**. Control: pollinator order alone 0.735, family 0.824 — so composition is a big part (Hymenoptera 38.2% curated vs Lepidoptera 2.4%, a 16× gap), but BioCLIP exceeds family, and Vp reaches 0.819 *containing no taxonomy at all* → geography itself is process-diagnostic (museum survey sites vs population centers). Feature families rank by process-predictiveness in the **reverse** order of their Tier-1 transfer. | `provenance_probe_v1.csv` |

| 21 | 09-01 | **Axis ablation — is provenance distinct from taxonomy?** (decisive control) | Compare curated vs iNat-only strata *within a single pollinator order*, so taxon composition is held fixed. **First pass omitted the Vp-heavy models that actually collapsed — corrected below.** | **The contrast survives the control, in refined form.** Within Hymenoptera only (curated vs iNat-only recall@10): **Vp-only 0.091 vs 0.180 = 0.51× (degrades 2×)**; rank_full_vp 0.95×; N 1.13×; N+tax+localΔ 1.07×; **gbm_geo 0.534 vs 0.402 = 1.33× (improves)**. So a **2.6× spread in process-response across feature families holding taxon fixed**: pure occupancy embeddings are genuinely process-dependent, structured/relational features are process-robust. Taxon composition inflates the raw gaps (gbm_geo 2.3× → 1.33×) but does **not** explain the direction or the spread. Consistent with the probe (Vp predicts source at 0.819 with zero taxonomic content). Secondary: all models far better on bees (0.53) than butterflies (0.30). | `axis_ablation_v1.csv` |

| 22 | 09-01 | Capture–recapture across documentation processes | Treat iNaturalist and curated sources as two "capture occasions": how complete is the documented network, and are the processes independent? | **Only 4.3% of edges (2,672/62,625) are documented by BOTH process families** — iNat-only 48,092, curated-only 11,861. Dependence ratio **0.23 (negative)**: the processes document largely *disjoint* edges because they specialise by taxon. Lincoln–Petersen/Chapman therefore **over**-estimates and is not identified here (pooled 276k vs stratified 354k, unstable — Lepidoptera overlap is 49 edges). Robust takeaway: the label set is a process-structured minority sample, and the two sources are near-disjoint *views*, which is why one works as a held-out test of the other. | `eval/capture_recapture.py` |

| 23 | 09-01 | **Identifiability graph** (Chen et al. ICML 2024 condition) | Unbiased-LTR theory says relevance is identifiable up to scale iff the bias-factor graph is connected. Nodes = GloBI source datasets, edge = ≥1 co-documented interaction. | 25 sources; 7 with ≥100 edges. Graph over the majors is **technically connected (1 component, 18 edges)** — but **practically vacuous**: only 4.8% of edges have >1 source, and the bridges are 4–13 shared edges (weakest: guzman2022↔web-of-life = **4 edges**). Combined with pair-dependent detection (exp 21 proves detection varies within taxon, breaking the rank-1 factorisation ULTR requires) and selection-on-the-outcome, **two-tower/ULTR-style correction is not identified on this data — and we can demonstrate it with three cheap diagnostics.** | `eval/identifiability_graph.py` |


| 24 | 09-01 | **Family-stratified provenance test** (falsification attempt on S1) | If the curated/iNat gap vanishes WITHIN pollinator families, detection is rank-1 conditional on taxon and debiasing IS identifiable — killing S1. | **S1 survives, but the direction FLIPS.** Within all 8 families with ≥25 test plants per stratum, curated-documented partners are *harder*, not easier: median ratio rank_n 0.37, vp 0.35, N+tax+lΔ 0.53, gbm_geo 0.77 (all <1). Models differ strongly on the same boundary within the same family (0.37 vs 0.77) ⇒ detection is pair-dependent even conditional on family ⇒ rank-1 violated, S1 holds. **BUT this is a Simpson's paradox vs exp 21** (order-level: curated *easier*, 1.33×). Consequence: the headline Tier-1 numbers (gbm_geo 0.471 vs 0.245 all-label) are substantially a **taxon-composition artifact** — the curated subset is Hymenoptera-heavy and bees are easier (0.53) than Lepidoptera (0.30). Honest statement: within-taxon, curated interactions are rarer/more specialised and *harder*; the provenance effect is real and model-dependent but its sign depends on conditioning. | `family_strata_v1.csv` |

| 25 | 09-01 | **S2 first test: species-level process signatures** (TOL-200M metadata, 666 shards) | Do per-species observation-process signatures, from a corpus independent of interactions, predict EDGE provenance? Is the species-level identifiability graph dense where the edge-level one was vacuous? | **Mixed, and the corpus is wrong.** 16,706/31,287 species profiled, 10.1M images. Composition (specimen share) predicts edge provenance at only **AUC 0.560**; magnitude 0.507 (nothing); all process features 0.654. Density is better than edge level (17.8% of species seen under both record types vs 4.8% of edges under >1 source) but far from dense. Two problems: plant coverage is only **25% of edges** (pollinators 80%), and **median specimen_share = 0** because TOL-200M only includes records that have IMAGES — museum specimens are structurally under-represented. Also corr(specimen_share, log_images)=0.40, so composition is not a clean abundance-free signal. **Fix: recompute the covariate from GBIF occurrence records (basisOfRecord, no image requirement, ~3B records) rather than an image corpus.** S2 not dead; the test was run on the wrong data. | `obs_profiles.parquet`, `eval/test_species_process.py` |
| 25b | 09-01 | **RETRACTION of exp 25** | Why did the obs pass find only 16,706 species when the embedding pass over the SAME corpus found 29,726 (95%)? | **Exp 25 is invalid.** ~93 of 111 shards per worker died with HfHubHTTPError — I ran 6 metadata workers concurrently with 8 embedding workers and got rate-limited; the script swallowed the errors and reported partial results as if complete. The S2 test therefore ran on ~15% of the corpus, and the "specimens are structurally under-represented / median specimen_share = 0" conclusion is also unsupported. Fixed: exponential-backoff retries + explicit failure accounting + lower concurrency. Rerunning. **Lesson: any silent `except: continue` in a network loop must count and report failures.** | `scripts/build_observation_profiles.py` |

| 26 | 09-01 | **Detection-factorization test** (predicted failure mode from integrated-model review) | Integrated interaction models (Kampe et al. 2025; Young et al. 2021; Anakok et al. 2024) assume detection factorizes as p_i·q_j. But one iNaturalist flower-visit photo documents BOTH partners in a single act, so their detections should be coupled. Test: fit species main effects to log(n_records) and measure residual pair structure. | **Confirmed, with a clean contrast.** Species main effects explain only **R²=0.187 for iNat-only edges** vs **0.398 for curated-supported edges** (multi-record: 0.116 vs 0.263). Documentation intensity is strongly pair-specific exactly where the data volume is. **The p_i·q_j factorization underpinning integrated/debiased interaction models is not supported for citizen-science data, and is ~2× better supported for specimen/survey data** — consistent with single-act coupling. | `eval/test_factorization.py` |
| 27 | 09-01 | **Anchor-dataset hunt** (research) | Is there a protocol-based NA plant-pollinator survey supplying true non-detections? | **No national one exists.** NEON has no pollinator/visitation product; USGS BIML host plants are opportunistic notes; the National Native Bee Monitoring Network is a coordination body with no data portal. **Best anchor: CaraDonna et al. 2017 RMBL** — weekly censuses 2013-15, 46 plants × 93 pollinators, ~30k interactions, Dryad 10.5061/dryad.s91p4 (open) + EDI edi.512.1 for effort metadata; closed plant list ⇒ non-detections legitimately derivable. Runners-up: Winfree lab Dryad (rates+effort, few plants), Oregon Bee Atlas (25k specimens w/ host plant genus). Closest large protocol dataset overall is European (EuPPollNet, 1.16M interactions with flower counts). | research report |
| 27b | 09-01 | RMBL acquisition attempt | Can we pull the anchor dataset programmatically? | **Blocked.** Dryad API needs a bearer token; `/downloads/file_stream/` returns 403 to this host; EDI `edi.512.1` is not publicly listable. Also note the Dryad deposit holds only 2 files (a 2 KB summary + a 16 MB simulation output) — it is the *turnover-analysis* deposit, likely NOT the raw plant×pollinator×week matrix. **ACTION FOR DAN: download via browser, and/or email Paul CaraDonna (Chicago Botanic Garden) for the raw weekly census data + post-2015 seasons.** | — |

| 28 | 09-01 | **Falsification of exp 26** (variance-artifact check) | Is the factorization R² contrast just an artifact of the strata having different n_records variance? Tests: (A) match strata on the n distribution; (C) permutation control. | **Survives.** Matched strata (identical y_SD 0.862): iNat R²=0.180 vs curated R²=0.355 — gap +0.175, barely reduced from the raw +0.211. Permutation control gives −0.052 ± 0.014, so the observed gap is ~16 SD from chance. **Detection is genuinely ~2× more factorizable for specimen/survey data than for citizen-science data; not a variance artifact.** | `eval/test_factorization_matched.py` |

| 29 | 09-01 | **Parametric null for exp 26/28 — S6 KILLED as stated** | Reviewer objection (from novelty check): low-mean overdispersed counts depress R² mechanically even under a TRUE multiplicative model; marginal matching doesn't fix mean-variance scaling. Correct null: simulate each stratum from its OWN fitted λ_i·μ_j (zero-truncated Poisson) and recompute. | **The source contrast does not survive.** Null R²: iNat 0.283, curated 0.675 → **null gap +0.392 EXCEEDS the observed gap +0.211**. Measured as shortfall from each stratum's own null, deviation from multiplicativity is **larger for curated (−0.277) than for iNat (−0.096)** — the opposite of the single-act-coupling mechanism I proposed. **What survives:** observed << null in BOTH strata, so detection is genuinely non-multiplicative for interaction data generally (contra Anakok 2024 / Kampe 2025 / Young 2021). **What dies:** the "citizen science is worse" contrast and its mechanism. Caveat: the generative fit is in-sample, which inflates the null more for the sparser curated stratum, so the reversal itself is not clean — the honest reading is that the source contrast is not evidential either way. | `eval/test_factorization_null.py` |

| 30 | 09-01 | **S4: BioCLIP-2 IMAGE embeddings as morphology proxy** (95% species coverage, 60.9M images) | Do image-derived species embeddings beat name-derived ones — i.e. is there trait/morphology signal beyond taxonomy? | **No.** Two-tower with image embeddings: **0.2553 [0.231–0.279], hit@10 0.725, Tier-1 0.443** — indistinguishable from BioCLIP-2 text (0.2512) and BioCLIP-1 text (0.2612). Mechanism diagnostic (balanced same-family pairs, 10k/10k): corr(text-sim, image-sim)=**0.844**; family-separation AUC text 0.920 vs image **0.943**; residual (image−text) alone still separates families at AUC 0.700. **Image centroids are a slightly sharper TAXONOMY encoder, not a distinct morphology axis** — redundant with the explicit taxonomy-affinity features we already use. (First cut used random pairs and had only 23 same-family positives; redone with stratified sampling.) | `twotower_bioclip2img.csv` |

| 31 | 09-01 | **S2 PROPER retest** (full TOL-200M sweep, 0 shard failures, 64.5M images / 29,743 species) | Redo of the invalid exp 25. Do species-level process signatures predict edge provenance? Is the species-level graph dense? Is composition abundance-free? | **S2 vindicated — Dan's idea works.** Coverage now **98.8% of edges (plants) / 99.7% (pollinators)**, 61,691 of 62,625 edges usable. **Composition alone (specimen share) predicts edge provenance at AUC 0.675**; magnitude 0.648; all process features 0.698. **Species-level identifiability is 14× denser than edge-level: 68.8% of species seen under BOTH record types vs 4.8% of edges under >1 source.** And **corr(specimen_share, log_images) = 0.055** — composition is essentially abundance-independent, so it is a legitimate exclusion-restriction candidate (the SAR-PU condition). Every number in exp 25 was an artifact of the rate-limited run, including the "specimens are under-represented" caveat (the corpus holds 10.2M specimen records). | `obs_profiles.parquet` |

| 32 | 09-01 | **Decisive S1 test: excluded observation-process bias head** (Dan's metadata-conditioning idea, implemented) | Shallow head fed ONLY species-level process descriptors, added to the score in training, discarded at inference. If it closes the within-Hymenoptera curated/iNat gap, debiasing IS achievable and S1 is wrong. | **Gap did NOT close; it widened slightly, and accuracy fell.** no-bias-head R@10 0.2612 (Hym curated 0.539 vs iNat 0.462, ratio 1.17); with bias head R@10 **0.2320** (0.528 vs 0.435, ratio **1.22**). Gap-to-parity change **+0.049** (wrong direction). Baseline reproduces twotower_bioclip 0.2612 exactly (sanity ✓). Consistent with S1's prediction that the standard remedy is not identified here — but a single configuration, and the head may simply be absorbing ecological signal or adding overfitting capacity. **Permuted-feature control required before this counts as evidence.** | `biashead_v1.csv` |
| 32b | 09-01 | **Permuted-feature control for exp 32** | Is the bias head's damage due to extra capacity/overfitting, or to the process features themselves? Same head, same scale, species↔feature association destroyed. | **Decisive: it is the features.** permuted head R@10 **0.2574** (vs baseline 0.2612 — costs 0.004, i.e. capacity is ~free) while the REAL head costs 0.029 (0.2320). Ratios: baseline 1.17, permuted **1.14**, real **1.22**. So conditioning on genuine process descriptors *removes ecologically useful signal* and moves the provenance ratio AWAY from parity, while random features of identical capacity do neither. **This is the "source is a descendant of the label" failure: curated collections are genuinely enriched in true bee-plant edges, so regressing out documentation process also regresses out ecology.** Strong support for S1: the standard remedy is not merely unidentified here, it is actively harmful — with a controlled demonstration. | `biashead_v1.csv` |

| 33 | 09-01 | **External robustness check** (Noori et al. 2026 Sci Data, 981,982 bee-plant records, 91 sources, global, CC BY) | Do the identifiability diagnostics replicate on an independently curated interaction database? (Caveat: GloBI-derived, so a differently-curated slice rather than a fully independent corpus.) | **Partial replication — thin bridges generalise, severity does not.** 165,418 edges; **18.8% documented by >1 source** (vs our 4.8% — 4× better, being global, bee-restricted and finer-grained in source labels). But the identifiability graph over 55 major sources is **still not connected** (2 components) and **61% of bridges rest on ≤13 shared edges** (median 7, min 1). So the "connected-but-vacuous" problem is a property of aggregated interaction databases generally, though our CONUS extraction is the more severe case. | `eval/replicate_noori.py` |

| 34 | 09-01 | **Second remedy family: importance reweighting toward the curated process** + ESS-matched control | Exp 32 tested only an additive bias head. Reweight training positives by P(curated)/P(iNat) from species-level process features. Control: 0/1 mask with identical effective sample size but no distribution distortion. | **Also fails, and the control rules out the obvious confound.** unweighted R@10 0.2612 (ratio 1.17); reweighted **0.1665 (ratio 1.41)**; **ESS-matched subsample 0.2388 (ratio 1.19)**. Weights are skewed (median 0.35, max 10) giving ESS = 8,685 = 18.5% of n — but an undistorted subsample of the *same* effective size keeps baseline behaviour, so reweighting costs **0.072 R@10 beyond sample-size loss** and pushes the ratio away from parity while the control stays at it. **Two structurally distinct remedy families now fail, each with its own control (permuted features; ESS-matched subsample).** | `reweight_v1.csv` |

| 35 | 09-01 | **SDM phase 1: GBIF vs SDM a_curves + first-ever bilateral per-bin Δ** (surfaces cached: 1,275 species × 3,162 cells × 52 wk from 62.4M rows, −0.25 offset, 5.2% out-of-grid) | Does model-predicted pollinator phenology beat observation-derived? Does making BOTH sides spatially local help? Restricted universe = 1,275 SDM-covered pollinators (79.6% of test edges); **numbers not comparable to full-universe results**. | **SDM does not beat GBIF, and no temporal feature beats N alone here.** N only **0.1688**; +GBIF Δ 0.1071; +GBIF local Δ 0.0975; +SDM Δ **0.0721** (worst); +SDM bilateral local Δ 0.0920; +both locals 0.0962. **But localisation helps SDM most:** SDM Δ 0.0721 → bilateral local 0.0920 (+0.020, the largest localisation gain of any arm), so per-bin matching does recover real signal within the SDM arm. Consistent with the old (invalid-benchmark) SDM-worse result and with Stage A, where Δ also hurt ranking *without* taxonomy present. **Caveat: this ladder has no taxonomy feature, and exp 8 showed temporal features only help once "who" context exists — so this is likely the wrong regime. Retest with taxonomy in all arms before concluding.** | `sdm_v1.csv` |

| 36 | 09-01 | **SDM phase 2: the taxonomy-present regime** (the correct regime per exp 8) | Does SDM phenology help once "who" context exists? Also answers plan step 4 (fitted-head vs zero-shot) via the per-arm head column. | **No. The September hypothesis is not supported.** N only 0.1688 → **N + tax 0.2569 (best)**; adding temporal features never improves it: +GBIF local 0.2535, +SDM Δ 0.2429, +SDM bilateral 0.2486, +both locals 0.2547 — all within overlapping CIs and all ≤ N+tax. Tier-1 same ordering (N+tax 0.3132 best). **Model-source stratification answers step 4 with no extra run: on the fitted-'head' subset (higher-quality SDM) N+tax is still best at 0.2688 vs SDM arms 0.2531–0.2659** — SDM fails even where the model is fitted rather than zero-shot. Note this also fails to replicate exp 8's small local-Δ gain (0.199→0.209) in the restricted universe, so that gain looks regime-specific/noise. Step 5 (permuted-SDM control) is moot: there is no gain to attribute. | `sdm_v1.csv` |

| 37 | 09-01 | **GBM fairness check for exp 36** | Can trees use phenology where the linear ranker could not (e.g. "Δ matters only at moderate N")? | **Model capacity WAS limiting — the sign flips, but the SDM verdict does not.** base (N+tax+ranges) 0.2622; **+GBIF temporal 0.2696; +SDM temporal 0.2673**; +both 0.2592. So with trees, temporal features are mildly *positive* (+0.007 GBIF, +0.005 SDM) rather than negative as in the linear arms — but all CIs overlap heavily and **GBIF ≈ SDM**. Corrects the implication of exp 35/36 that temporal features are simply harmful: they are weakly useful given a model that can express interactions. | `sdm_gbm_v1.csv` |

| 38 | 09-01 | **OOD test of Dan's hypothesis: does SDM win where GBIF is thin?** Stratify the same test set by GBIF record count per pollinator (median 12 records/species — most GBIF curves are ~noise). | Prediction: SDM−GBIF positive in the sparse tercile, shrinking toward dense. | **Prediction not borne out, and the test is structurally confounded.** SDM−GBIF by tercile: T0 (sparse) **−0.0051**, T1 −0.0089, T2 (dense) **+0.0031** — i.e. the opposite sign pattern, though all differences are tiny vs CIs. Absolute performance tracks density hard (T0 0.045, T1 0.107, T2 0.294), so there is a floor effect limiting power in T0. **The decisive confound: SDM quality is almost perfectly tied to GBIF density** — corr(log GBIF records, log SDM training obs) = **0.865**; T0 is **83.8% zero-shot LE-SINR with median 0 training observations**, T2 is 97.4% fitted heads with median 1,788. The SDM is trained on GBIF, so wherever GBIF is thin the SDM is *also* uninformed and falls back to its environmental-covariate pathway. | `sdm_ood_v1.csv` |
| 38b | 09-01 | **Implication: OOD claims are untestable with a pre-trained SDM** | Can any out-of-distribution variant (space or time) cleanly test "SDM generalises where observation is absent"? | **No, not with this build.** The zero-shot subset IS the sparse regime, and there SDM does not beat GBIF (−0.005, floor-limited). For an out-of-TIME test our GBIF curves are built from all years while the SDM's training cutoff is unknown — either direction of leakage is possible and we cannot control it. Same for out-of-space: the SDM was fitted across CONUS, so no region is genuinely held out. **A clean test requires an SDM retrained with explicit spatial/temporal holdouts — that is Dan's to produce, and it is the single change that would make the generalisation claim testable.** | — |

| 39 | 09-01 | **SDM on the ORIGINAL metric: pooled PR-AUC** (every prior SDM test used ranking; the project's original claim lives on PR-AUC) | Does SDM temporal beat GBIF temporal for pooled pair discrimination? Paired cluster bootstrap over test plants (2,000 resamples) since marginal CIs overlap ~95% but the arms are highly correlated. | **First positive SDM result — and it is metric-dependent.** Linear model: base 0.5181 → +GBIF 0.5439 → +SDM **0.5502** → +both 0.5547. Paired test: **SDM−GBIF +0.0063 [+0.0026,+0.0101], p=0.0005**; GBIF−base +0.0257, SDM−base +0.0320 (both p=0.0005). So on pooled discrimination, temporal features clearly help and **SDM significantly beats GBIF** — supporting the project's original hypothesis on its own metric. **Two caveats that bound the claim:** (a) the effect is small (+0.006 PR-AUC); (b) under a GBM the advantage disappears (SDM 0.4943 vs GBIF 0.4967), so SDM's benefit looks like its signal being more *linearly accessible* rather than containing more information. And it does not transfer to ranking, where SDM ≈ GBIF (exp 36/37). | `sdm_prauc_v1.csv`, `sdm_paired_v1.csv` |

| 40 | 09-01 | **Plan step 5: permuted-SDM control on the exp-39 gain** (now warranted — there is finally a gain to attribute) | Shuffle species → SDM-surface assignment, keeping dimensionality and geometry identical. If the permuted arm reproduces the gain, exp 39 is capacity not phenology. | **Exp 39 CONFIRMED.** real−base **+0.0320** [+0.0254,+0.0389]; **permuted−base −0.0009** [−0.0014,−0.0004] — i.e. essentially nothing; real−permuted +0.0329, p=0.0005. The entire SDM temporal gain is attributable to the *correct species'* phenology; random surfaces of identical shape contribute zero. This is the strongest positive result of the SDM phase. | `sdm_permuted_v1.csv` |

## SDM phase summary (2026-09-01) — all 5 plan steps complete

**Positive, with controls:** SDM phenology adds **+0.032 PR-AUC** over a base of Vf/Vp/N/taxonomy, and a permuted-species control contributes **−0.001**, so the gain is genuine phenological signal (exp 39/40). SDM significantly beats GBIF on this metric: **+0.0063, p=0.0005** paired (exp 39). The project's original hypothesis is supported *on its original metric*.

**Negative / bounded:** the advantage does not transfer to top-k ranking, where SDM ≈ GBIF and neither beats N+taxonomy (exp 36/37); it disappears under a GBM (0.4943 vs 0.4967), so SDM's benefit may be *linear accessibility* rather than extra information; bilateral per-bin Δ — computable for the first time — gave the largest within-arm localisation gain (0.0721→0.0920) but did not change the SDM-vs-GBIF verdict.

**Untestable with this build (exp 38/38b):** the generalisation claim. SDM training data and GBIF density correlate at **r=0.865** (sparse tercile = 83.8% zero-shot, median 0 training obs), so "SDM wins where observation is thin" cannot be separated. Out-of-time and out-of-space variants are equally confounded because the SDM's training cutoff and spatial coverage are not ours to control. **Actionable ask: an SDM retrained with explicit spatial/temporal holdouts would make this decisive.**

**Corrected during this phase:** exp 35/36 implied temporal features are harmful — that was a linear-model artifact; under trees they are mildly positive (exp 37).

| 41 | 09-01 | **Why does phenology help pooled PR-AUC but not recall@10?** Per-plant AUC of each feature against two negative sets: random species (filter-like) vs CO-OCCURRING non-partners (discriminator-like), 227 test plants. | Hypothesis: phenology is a feasibility *filter*, not a partner *discriminator*. | **Hypothesis too strong — refined.** Δ does NOT collapse against co-occurring candidates: AUC 0.866→**0.757** (drop 0.109); local Δ 0.919→**0.764** (drop 0.154); N 0.946→0.866 (drop 0.080). So phenology remains a genuine discriminator among co-occurring species, just a weaker one, and loses proportionally more power than N when co-occurrence is conditioned on. **The real reconciliation is AUC-vs-top-k**: Δ alone ranks at 0.757 AUC, i.e. it improves the *overall ordering*, but does not sharpen the *top of the list* — and once N and taxonomy are present its top-10 contribution is redundant. This explains the metric divergence (exp 36/37 vs 39) precisely. | `filter_vs_discriminator.csv` |

| 42 | 09-01 | **Cascade feasibility** (proposed method: retrieve with N+Δ where phenology pays, re-rank with taxonomy) | Exp 41 suggested Δ improves broad ordering, so it should raise recall at LARGE k even while failing at k=10 — the precondition for a cascade. | **Precondition fails; the proposed method is dead.** Δ hurts at *every* k and hurts MOST in the mid-range: N+localΔ minus N = −0.029 @10, **−0.107 @50, −0.130 @100**, −0.096 @500, −0.064 @1000. So Δ never produces a better candidate set. **Refines exp 41:** Δ alone has per-plant AUC 0.757, but it is redundant with N and *dilutes* it in a linear model — the pooled-PR gain (exp 39/40) arose in a much richer base (Vf,Vp,N,tax) where it is not redundant. Feature value here is strongly conditional on both the feature set and k: with taxonomy present, local Δ helps at k≤100 (R@100 0.543 vs 0.523) but hurts at k≥500 (R@1000 0.836 vs 0.861). | `cascade_feasibility.csv` |

| 42 | 09-01 | **Is recall@10 the right metric? MRR / nDCG / MAP comparison** (553 test plants, median degree 6, 36.3% have >10 partners) | Does the choice of ranking metric change which model wins? | **Four metrics agree; MRR is the one that disagrees — and it disagrees by rewarding the popularity shortcut.** recall@10, nrecall@10 (degree-normalised), nDCG@10 and MAP all give the identical ordering GBM > N+tax+localΔ > N-only > degree-null. **MRR alone promotes the degree null above N-only (0.2838 vs 0.2496)** because it scores only the FIRST hit: putting *Apis*/*Bombus* atop every list finds *a* partner fast while missing the rest. Degree cap is real — 36.3% of plants exceed 10 partners, so raw recall@10 understates (GBM 0.2454 raw vs **0.3505 degree-normalised**). Median rank of first true partner: GBM **3**, N-only 10, degree null 12. | `metric_comparison.csv` |

| 43 | 09-01 | **MAP vs nDCG, and is graded relevance safe?** nDCG's advantage over MAP is graded gains; our natural grade (`n_records`) is documentation intensity, which exp 6/19 showed is process-contaminated — so grading might reintroduce popularity bias. Three gain definitions tested. | Does the gain choice change model ordering or promote the popularity null? | **Grading is safe here, and the ordering is fully stable.** nDCG@10 — binary: GBM 0.3278 > N+tax+localΔ 0.2869 > N-only 0.1533 > degree null 0.1393; log n_records: 0.3016 / 0.2670 / 0.1278 / 0.1091; curated-weighted: 0.3112 / 0.2537 / 0.1252 / 0.1039. **Identical ordering under all three**, and the degree null stays last in every case (unlike under MRR, exp 42). Graded gains lower all absolute values (harder to match an ideal that concentrates weight on heavily-documented partners) but change no conclusion. (First run returned NaN: the provenance merge left ~200 edges unmatched; fixed by using the complete `n_records` column.) | `graded_ndcg.csv` |

## Candidate stories (kept deliberately plural; updated each loop tick)

| # | Framing | Evidence FOR | Evidence AGAINST / risk | Status |
|---|---|---|---|---|
| S1 | **Feasibility diagnostics for debiasing biodiversity data** — three cheap tests (provenance probe; within-stratum transfer gap; identifiability graph) that say whether bias correction is even identifiable on a given dataset. Applied to the largest interaction dataset, all three say no. | All three implemented and run (exp 19, 21, 23). IG connected but vacuous (4-13 edge bridges). Rank-1 violated. Selection-on-outcome non-recoverable. | Negative result; reviewers may demand a fix, not just a diagnosis. Adjacent to Yilmaz PNAS 2025 + DivShift AAAI 2025. | **Strongest and now with a controlled remedy-failure demonstration (32/32b) plus partial external replication (33).** |
| S2 | **Species-level observation model** (Dan, 09-01) — move the bias model from edge level (vacuous) to species level using TOL-200M process signatures; composition (specimen vs photo share), not magnitude, as the excluded covariate. | Edge-level IG is vacuous but species-level co-occurrence across sources should be dense. TOL-200M gives basisOfRecord + source_dataset free for ~all species. Satisfies SAR-PU's "propensity uses fewer attributes" condition by construction. | Untested. Abundance confounds raw counts. Still supplies no non-detections; absolute prevalence stays unidentified. | **Revived (exp 31): dense species-level graph (68.8%), composition predicts provenance at 0.675 and is abundance-free (r=0.055). Model form still overlaps Kampe 2025 — the contribution would be the covariate + the identifiability argument, not the structure.** |
| S3 | **Corrected benchmark + cold-start protocol** — 62,832-edge dataset (450x the prior extraction), nulls, prospective validation. | Reproduced published numbers exactly then broke them; N-only super-baseline; temporal holdout works (hit@10 0.73). | Dormann et al. 2025 already published "abundance baseline wins" for pollination networks. "We fixed a parsing bug" is not a story alone. | Supporting act |
| S4 | **Morphology-as-trait-proxy** — BioCLIP-2 *image* embeddings substitute for the trait databases that do not exist (corolla depth: 42 species worldwide). | Text embeddings already helped as tower inputs (+0.036, p=.001). Image centroids obtainable free from TOL-200M. | Untested; may be redundant with taxonomy affinity, exactly as the text embeddings were. | **Negative (exp 30): image ≈ text ≈ sharper taxonomy; no morphology axis.** Trait-matching still untested — needs real traits, which don't exist at scale. |
| S5 | **Modelled inputs beat observed inputs** — PPE (model-predicted plant phenology) helps; GBIF (observed pollinator phenology) hurts; SDM should help. Ties input provenance to generalisation. | Coherent with the process-dependence findings. | The original asymmetry was derived on the broken 139-pair benchmark and needs re-testing. Blocked on Dan's SDM. | **Metric-dependent (exp 35-39). On pooled PR-AUC the original claim HOLDS: temporal features help (+0.026/+0.032) and SDM significantly beats GBIF (+0.0063, p=0.0005). On top-k ranking it does NOT: SDM ≈ GBIF, and neither beats N+taxonomy. SDM's edge vanishes under a GBM, so it may be linear accessibility rather than extra information. Generalisation advantage remains untestable (exp 38b).** |

**Correction (exp 24):** Tier-1 gains reported in exp 12/16 are inflated by taxonomic composition; report within-taxon strata alongside them. Figures/tables citing Tier-1 need this caveat.

| S6 | **Detection does not factorize for citizen-science interaction data** — the p_i·q_j assumption behind every integrated/debiased interaction model (Kampe 2025, Young 2021, Anakok 2024) holds ~2× better for specimen data than for the photo-sourced data that dominates. Mechanism: one photo documents both partners in a single act. | Exp 26/28 raw contrast; **withdrawn by exp 29**. Residual: observed R² << parametric null in both strata (iNat 0.187 vs 0.283; curated 0.398 vs 0.675). | Needs an independent replication (e.g. on frugivory/EuPPollNet) to show it generalises beyond our dataset. | **KILLED as stated (exp 29).** Reduced claim survives: detection is non-multiplicative for interaction data in BOTH strata, contradicting the assumption in published integrated models. The source contrast — the novel part — is not evidential. |

Kill criteria (updated 09-01): **S1** dies if any structurally different debiasing remedy closes the provenance transfer gap — one form (additive excluded bias head) failed with a permuted control (exp 32/32b); a second form (importance reweighting) also failed with an ESS-matched control (exp 34). **Two distinct families with controls now support the class-level claim.** S1 would still die if a per-source-head / multi-annotator formulation, or a genuinely identified integrated model with an anchor dataset, closed the gap. **S2** survives as a measurement result (exp 31: composition predicts provenance at 0.675, abundance-free, species graph 14× denser) but its *remedial* use failed (exp 32) — so it is a good covariate and a bad correction. **S3** stands as a supporting contribution. **S4/S6** are dead (exp 30, 29). **S5** blocked on Dan's SDM.

Standing risk: every framing here rests on ONE dataset lineage (GloBI). Exp 33 is only a partial external check because Noori et al. is itself GloBI-derived. A genuinely independent corpus (designed survey, e.g. RMBL) remains the single highest-value acquisition.

## Session summary (2026-09-01) — what is established

**Established (with controls):**
1. The corrected dataset and cold-start benchmark (S3): 62,832 edges, nulls, prospective validation. Best model 0.277 R@10 / 0.741 hit@10 (rank-average ensemble).
2. Documentation process is strongly measurable: features predict the documenting source at 0.91 AUC; species-level composition predicts edge provenance at 0.675 and is abundance-free (r=0.055); species-level identifiability is 14× denser than edge-level (68.8% vs 4.8%).
3. Standard debiasing is not achievable here, shown three ways *before* fitting (rank-1 violated; identifiability graph connected-but-vacuous; detection non-multiplicative in both strata) and two ways *after* fitting (additive excluded bias head, and importance reweighting), each with its own control.
4. Mechanism for the failure: documentation source is a descendant of the label — curated collections are genuinely enriched in true bee-plant edges — so regressing out process also regresses out ecology.

**Retracted this session (kept for the record):** exp 25 (rate-limited run reported as complete); S6/exp 26 (source contrast in factorisation, killed by a parametric null in exp 29); the Tier-1 headline gains (composition-inflated, exp 24 Simpson's paradox); the first same-family embedding test (23 positives, redone in exp 30).

**Blocked:** Dan's full SDM (Sept); RMBL anchor dataset (needs manual download — Dryad/EDI refuse automated fetch).

**Highest-value next acquisition:** a designed-survey corpus supplying true non-detections. Everything here rests on one dataset lineage (GloBI); exp 33's external check is only partial because Noori et al. is GloBI-derived.

## Stopping point (2026-08-31, end of autonomous session)

Every unblocked rung is complete: data correction → protocol → benchmark → objective → features → representations → architectures → ensembling → transfer, temporal and error analysis. Headline arc: **recall@10 0.117 → 0.277 (2.4×), hit@10 0.51 → 0.74, Tier-1 0.117 → 0.504 (4.3×)**, with prospective validation on later-documented interactions.

**Blocked / needs Dan:** (a) SDM a_curves swap — the one remaining feature hypothesis, and now sharply testable: does SDM-derived pollinator timing beat GBIF-derived Δ inside the ranker? (b) Whether to promote `antheia-package` into `main` and treat `edges_v1` as the canonical dataset (the 139-pair results and everything downstream of them should be retired).

**Next rungs when work resumes:** SDM swap; recalibration of scores → probabilities for brittleness thresholds (Direction 4); NN architecture/regularization search (val peaks early, ~epoch 2 for the no-emb tower); TaxaBind/SINR non-name modalities; paper drafting from this log.

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

---

## Experiment 22 (09-01) — External ecological baselines · **changes the reference point**

`eval/run_eco_baselines.py` → `artifacts/eco_baselines_v1.csv`. Frozen split, same protocol.

| baseline | R@10 [95% CI] | R@50 | hit@10 | Tier-1 R@10 |
|---|---|---|---|---|
| abundance null (pollinator observation counts; Vázquez/Dormann neutral model) | 0.051 [0.042–0.062] | 0.242 | 0.389 | 0.058 |
| **congeneric transfer** (partners of same-genus training plants, family fallback) | **0.209 [0.188–0.233]** | 0.396 | 0.653 | 0.337 |
| **latent-trait SVD, k=32** (Strydom-style RDPG + taxonomic imputation) | **0.204 [0.181–0.225]** | 0.400 | 0.647 | 0.362 |

**Implications — the honest baseline is not N.**

1. **Congeneric transfer (0.209) exactly matches our engineered linear model** (N + taxonomy + local Δ, 0.209). "Look at what this plant's congeners are visited by" is free, needs no model, and equals a fitted feature-based ranker. Any claim must be made against *this*, not against N-only (0.117).
2. **Latent-trait SVD (0.204) is right behind it** — the ML-ecology incumbent is competitive out of the box, and has the best Tier-1 of the three (0.362).
3. **The abundance null is weak here (0.051)** — worse than the degree null. Dormann et al. 2025 found abundance dominant for interaction *frequency* within local networks; for continental cold-start *retrieval* it is not, because ranking within a plant cannot use the plant's abundance term at all. Worth stating explicitly, since a reviewer will expect abundance to win.
4. Our margin shrinks accordingly: best single model 0.261 and ensemble 0.277 vs **0.209** — a real but modest ~+0.07. The margin is much larger on curated labels (ensemble 0.504 vs SVD 0.362, +0.14), which is the more defensible headline.

75.6% of test plants have a training congener; the rest fall back to family, which is why the baseline degrades gracefully rather than failing.

---

## Experiment 23 (09-01) — Two-head model (shared towers, retrieval + compatibility)

`src/antheia/twohead.py`, `eval/run_twohead.py` → `artifacts/twohead_v1.csv`. Shared plant/pollinator
towers + `τ⟨p,q⟩`; **head R** adds only candidate-varying pair features; **head C** additionally gets
plant-only context (log range, flowering breadth) that is constant within a plant and therefore
invisible to any within-plant ranking. Joint loss = sampled-softmax (R) + BCE (C) on the same candidates.

| variant | head | R@10 | pooled PR | Tier-1 R@10 |
|---|---|---|---|---|
| joint | retrieval | 0.2574 | 0.8611 | 0.457 |
| joint | compatibility | 0.2598 | 0.8582 | 0.498 |
| rank_only (λ_C=0) | retrieval | **0.2612** | 0.8562 | 0.475 |
| rank_only | compatibility | 0.0597 | 0.6225 | 0.117 |
| comp_only (λ_R=0) | retrieval | 0.1053 | 0.7402 | 0.280 |
| comp_only | compatibility | 0.2180 | **0.8191** | 0.485 |

**1. Multi-task is worth it.** Joint training lifts the compatibility head substantially over
training it alone (pooled PR 0.8582 vs 0.8191; R@10 0.260 vs 0.218) while costing retrieval
nothing measurable (0.2574 vs 0.2612, overlapping CIs). One model serves both questions.

**2. The heads genuinely specialize.** Train one objective and the other head collapses —
rank_only's compatibility head falls to 0.060 R@10 / 0.62 PR; comp_only's retrieval head to 0.105.
The two objectives are not interchangeable views of one score.

**3. Δ scalars add nothing here — but this is NOT a test of temporal information.** Ablating the
Δ / local-Δ columns changed nothing (noDelta 0.2631 vs both 0.2574 vs global 0.2451 vs local
0.2438; pooled PR flat at 0.861–0.863, all CIs overlapping). **The towers already consume the raw
52-week flowering and activity curves**, so removing the hand-computed overlap scalar only tests
whether that scalar adds anything *beyond* what a neural encoder extracts from the curves itself.
It does not. The real temporal ablation must also remove the curves from the tower inputs
→ Experiment 24.

---

## Experiment 24 (09-01) — **True temporal ablation** (3 seeds, paired bootstrap) · the temporal result

`eval/run_temporal_ablation.py` → `artifacts/temporal_ablation_v2.csv`. Removes phenology from **both**
places it enters: the 52-week curves in the tower inputs *and* the Δ / local-Δ scalars in the wide path.
3 seeds, seed-averaged per-plant recall, paired bootstrap over the 553 test plants.

| variant | curves | Δ | retrieval R@10 | compatibility R@10 | pooled PR |
|---|---|---|---|---|---|
| curves + Δ | ✓ | ✓ | 0.2526 | 0.2558 | 0.8611 |
| curves only | ✓ | ✗ | 0.2539 | 0.2572 | 0.8609 |
| Δ only | ✗ | ✓ | 0.2437 | 0.2351 | 0.8549 |
| no temporal | ✗ | ✗ | 0.2436 | 0.2366 | 0.8578 |

**Paired bootstrap (seed-averaged, 10k resamples):**

| contrast | retrieval | compatibility |
|---|---|---|
| curves_only − no_temporal | +0.0103 [−0.003, +0.024] **p = 0.13** | **+0.0206 [+0.008, +0.033] p = 0.0014** |
| curves+Δ − no_temporal | +0.0090 p = 0.14 | **+0.0192 p = 0.0002** |
| curves+Δ − curves_only | −0.0013 p = 0.80 | −0.0014 p = 0.78 |
| Δ_only − no_temporal | +0.0001 p = 0.997 | −0.0015 p = 0.71 |

**Two clean findings.**

1. **Temporal information is objective-specific.** Raw phenological curves significantly improve the
   **compatibility** objective (+0.021, p = 0.001) but **not** retrieval (+0.010, p = 0.13). Ecologically
   coherent: phenological overlap tells you whether two species *can* meet — a feasibility / forbidden-link
   signal — but among candidates that already overlap in space and time it does not discriminate *which*
   one actually interacts. Phenology rules things out; it does not rule things in.
2. **The classical Δ carries essentially none of the usable signal.** `Σ min(f_t, a_t)` (Ridout & Linkie
   overlap — the construct ANTHEIA was built on) adds nothing on top of the curves (p ≈ 0.8) and,
   on its own, is indistinguishable from having no temporal information at all (p = 0.997 / 0.71).
   The signal lives in the raw weekly curves and a learned encoder recovers it; the hand-computed
   summary destroys it. This retroactively explains why Δ, V_δ (4-D/15-D) and PMf all hovered at
   no-effect — they are all elaborations of the same lossy statistic.

Note: single-seed numbers were noisier (retrieval gap 0.026); 3 seeds shrank the retrieval effect to
non-significance and confirmed the compatibility one. Multi-seed was necessary.

---

## Experiment 25 (09-01) — BioCLIP-2 **image** embeddings (morphology as trait proxy)

Built taxon-centroid image embeddings from `imageomics/TreeOfLife-200M-Embeddings` (CC0, precomputed;
no images downloaded): **60.9M images, 94.8% of plants / 95.1% of pollinators covered, median 86
images per species**, 768-D, streamed via 8 parallel workers over the taxonomically-sorted shards.
`scripts/build_image_embeddings.py` + `scripts/merge_image_embeddings.py`.

Two-head model, image embeddings substituted for text embeddings as tower inputs:

| tower input | retrieval R@10 | compatibility R@10 | pooled PR | Tier-1 R@10 |
|---|---|---|---|---|
| BioCLIP **text** (names) | 0.2574 | 0.2598 | 0.8611 | **0.498** |
| BioCLIP-2 **image** (morphology) | 0.2496 | 0.2565 | 0.8621 | 0.431 |

**Negative result: images do not beat names.** Retrieval and compatibility are within overlapping CIs;
Tier-1 is actually worse (0.431 vs 0.498). The morphology-as-trait-proxy hypothesis — "a long-tongued
bee *looks* like one, so vision should recover trait matching without a trait database" — is not
supported at species-centroid granularity.

**Likely why, and it is a tidy explanation:** BioCLIP is *trained to predict taxonomic labels*. Both its
text and image towers therefore converge on taxonomy-shaped representations, so neither supplies
information independent of the taxonomy features we already have. This unifies experiments 11, 15 and
25: name embeddings ≈ soft taxonomy, image embeddings ≈ soft taxonomy, count-based taxonomic affinity
beats both. A foundation model trained on taxonomy returns taxonomy.

Caveat: we used the mean embedding over ~86 images per species, which may wash out functional
morphology. A trait-supervised or part-level image representation is a different (untested) proposition.
