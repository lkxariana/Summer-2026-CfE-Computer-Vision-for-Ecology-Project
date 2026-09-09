# Experiment Log

All experiments run on the frozen protocol unless noted: `edges_v1` (62,832 orientation-corrected GloBI pairs), degree-stratified plant split 75/10/15 (`artifacts/split_v1.json`, 553 test plants / 9,210 test positives), leave-plant-out, ranking over all 24,939 pollinators, bootstrap-over-plants CIs. Reference numbers to beat: **raw N recall@10 = 0.117, hit@10 = 0.505; best pooled PR-AUC 0.657**.

**Metric protocol (settled exp 42-44):** report **recall@k, nDCG@k and MAP@k at k = 10 and 50**, plus hit@10 and median-rank-of-first-partner as plain-language deployment numbers. Use binary gains for nDCG (graded gains give identical ordering, exp 43). Do **not** headline k=1/5 or MRR — both reward the popularity shortcut (exp 42, 44). Degree-normalised recall (nrecall) accompanies recall because 36% of test plants have >10 partners.

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

| 44 | 09-01 | **Which cutoffs? k-sweep over recall / nrecall / nDCG / MAP at k = 1,5,10,20,50,100** | Is the model ordering stable in k, or does the cutoff choice change conclusions? | **The two real models never reorder at any k; only the baselines do — and only at small k.** At k=1 and k=5 the **degree (popularity) null beats N-only** on nrecall/nDCG/MAP (e.g. nDCG@5 0.1635 vs 0.1430); from **k≥10 onward N-only wins** (nDCG@10 0.1533 vs 0.1393). Same failure mode as MRR (exp 42): tiny cutoffs reward putting *Apis*/*Bombus* first. Degree cap context: share of test plants with degree>k = 78% (k=1), 50% (k=5), **36% (k=10)**, 21% (k=20), 7% (k=50). Metric shapes differ — recall rises with k by construction, MAP falls monotonically, nDCG is U-shaped (easy IDCG at small k). **Recommendation: report @10 and @50; never headline @1/@5.** | `k_sweep.csv` |

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
2. **The classical Δ carries essentially none of the usable signal.** `Σ min(f_t, a_t)` (the construct ANTHEIA was built on;
   the correct precedent is the phenology probability matrix of Vizentin-Bugoni et al. 2014, not
   Ridout & Linkie 2009, which is a diel camera-trap method) adds nothing on top of the curves (p ≈ 0.8) and,
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

---

## Overlap-summary family test (window 2, 09-01) — *does the Δ result generalise?*

`eval/run_overlap_summaries.py` → `artifacts/overlap_summaries_v1.csv`. Prompted by Dan's challenge:
"Σ min is our internal choice, not necessarily the field standard." So instead of one scalar, test
**seven** overlap statistics in common ecological use — coefficient of overlapping (`Σ min`, which for
normalised curves is algebraically identical to Schoener's D), Pianka cosine, Bhattacharyya, Pearson
correlation, joint active weeks (the "days of overlap" convention), circular peak offset, and
Jensen-Shannon similarity — against the raw 52-week curves. 4 arms × 3 seeds, paired bootstrap.

| arm | retrieval R@10 | compatibility R@10 |
|---|---|---|
| no temporal | 0.2340 | 0.2481 |
| **all 7 summaries** (no curves) | 0.2345 | 0.2321 |
| **curves** (no summaries) | 0.2440 | 0.2501 |
| curves + summaries | 0.2394 | 0.2396 |

| contrast | retrieval | compatibility |
|---|---|---|
| summaries − no_temporal | +0.0005 **p = 0.91** | **−0.0159 p = 0.004** |
| curves − no_temporal | **+0.0100 p = 0.012** | +0.0021 p = 0.61 |
| curves − summaries | +0.0095 p = 0.058 | **+0.0180 p = 0.002** |
| curves+summaries − curves | −0.0045 p = 0.28 | **−0.0105 p = 0.017** |

**The claim generalises.** No scalar overlap statistic in common ecological use recovers the signal:
all seven together are indistinguishable from having no temporal information (p = 0.91 on retrieval)
and actively *hurt* compatibility (−0.016, p = 0.004). Curves beat summaries on both objectives.
Adding summaries on top of curves is neutral-to-harmful. So the earlier finding was not an artifact
of ANTHEIA's particular Δ — **the scalar-summary encoding itself is the problem**, whichever of the
field's conventions you pick.

⚠️ Method caveat: this script trains a fixed 6 epochs with no val early-stopping (unlike exp 24), so
absolute values are not comparable across experiments; only the within-experiment contrasts are valid.
That is also why `curves − no_temporal` on compatibility is n.s. here (+0.002) while exp 24 measured
+0.021 with early stopping and a richer wide path.

---

## Phenology trajectory (window 2, 09-01) — **the encoding ladder has a peak, not a slope**

`scripts/build_phenology_trajectory.py` (6,348 × 260: per week, flowering mass + lat/lon centroid of that
mass + spatial spread + occupied extent, aligned across species by construction so it cannot repeat the
V_δ misalignment bug) → `eval/run_trajectory.py` → `artifacts/trajectory_v1.csv`. Plant-side temporal
representation varied; pollinator side held constant. 3 seeds, paired bootstrap.

| plant temporal input | dims | retrieval R@10 | compatibility R@10 |
|---|---|---|---|
| none | 0 | 0.2456 | 0.2534 |
| **52-week curve** | 52 | **0.2539** | **0.2572** |
| spatiotemporal trajectory | 260 | 0.2472 | 0.2329 |
| curve + trajectory | 312 | 0.2378 | 0.2485 |

| contrast | retrieval | compatibility |
|---|---|---|
| curve − none | **+0.0082 p = 0.013** | +0.0038 p = 0.29 |
| trajectory − none | +0.0015 p = 0.79 | **−0.0205 p = 0.0001** |
| trajectory − curve | −0.0067 p = 0.27 | **−0.0243 p = 0.0001** |
| curve+trajectory − curve | **−0.0161 p = 0.006** | −0.0087 p = 0.09 |

**Negative result: more spatial resolution does not help — it hurts.** The trajectory is no better than
no temporal information on retrieval (p = 0.79) and significantly *worse* than nothing on compatibility
(−0.021, p = 0.0001); adding it on top of the curve degrades retrieval (−0.016, p = 0.006).

**So the ladder is not monotone.** Scalar summaries (1–7 dims) carry nothing; the 52-week curve carries
a real, significant gain; 260-dim spatiotemporal structure adds noise. **The curve is the sweet spot.**

Two candidate explanations, and they are distinguishable in principle:
1. **Structural — the likely one.** The pollinator side has no spatial dimension (`a_curves` are pooled
   over all CONUS), so plant-side spatial detail has nothing to align against. A dot product cannot
   compute spatiotemporal matching when only one side carries space. This experiment therefore tests
   "does plant-side spatial detail help when the partner is aspatial" — and the answer is no. Consistent
   with exp 35, where making *both* sides local (SDM) was the only arm where localisation gained (+0.020).
2. **Capacity** — 260 mostly-redundant dims over 46,897 training positives.

---

## Unified encoding ladder (window 2, 09-01) — **⚠️ corrects exps 24 and the overlap-summary test**

`eval/run_encoding_ladder.py` → `artifacts/encoding_ladder_v1.csv`. All six arms under ONE protocol:
identical wide base `[log N, tax_genus, tax_family]`, identical towers, identical val early-stopping,
3 seeds, paired bootstrap vs `none`. Built because exps 24 / overlap-summaries / trajectory each used
a different training configuration, so their rows were never strictly comparable.

**Paired bootstrap vs `none` (seed-averaged per-plant recall@10):**

| arm | retrieval | compatibility |
|---|---|---|
| delta_scalar | +0.0040 p = 0.32 | −0.0002 p = 0.97 |
| summaries_7 | +0.0045 p = 0.30 | **−0.0143 p = 0.0008** |
| curves | +0.0041 p = 0.37 | +0.0011 p = 0.81 |
| curves + delta | +0.0077 p = 0.12 | **+0.0135 p = 0.004** |
| curves + trajectory | +0.0010 p = 0.85 | −0.0078 p = 0.16 |

**What survives, and what does not.**

- **Nothing significantly helps retrieval.** Every arm p ≥ 0.12. The +0.010 curve gain measured in the
  overlap-summaries run (p = 0.012) and the +0.008 in the trajectory run (p = 0.013) do **not** replicate
  under this protocol.
- **On compatibility, only `curves + delta` helps** (+0.0135, p = 0.004). Curves *alone* are null here
  (+0.001, p = 0.81) — directly contradicting exp 24, which measured +0.021 (p = 0.0014) for the same
  nominal contrast under a 5-column wide path.
- **Two negatives replicate cleanly**: scalar summaries never help retrieval and *hurt* compatibility
  (−0.014, p = 0.0008, matching the earlier −0.016); the trajectory never helps.

**Honest conclusion: the positive temporal effect is small and protocol-dependent.** Its sign and
significance move with the wide-path configuration. The negatives are robust; the positives are not.
Any claim of the form "phenology helps objective X by Y" needs many more seeds than 3 before it is
reportable. Superseding claim for the paper until then: *no encoding of phenology we tested — scalar,
multi-statistic, raw-curve, or spatiotemporal — produces a reliable retrieval gain, and only the
curve+scalar combination shows a compatibility gain that does not replicate across protocols.*

---

## Bilateral temporal ladder (window 2, 09-01) — **phenology helps when BOTH sides are model-derived and per-cell**

`src/antheia/localmatch.py`, `eval/run_bilateral_ladder.py` → `artifacts/bilateral_ladder_v1.csv`.
Restricted to the **1,275 SDM-covered pollinators** — the only universe where both sides have a
(location × week) surface. Same plant split; ranking over 1,275 candidates, so **numbers are NOT
comparable to full-universe tables**. Taxonomy present in all arms (the regime exp 8 showed is required).
2 seeds; uncertainty is the frozen bootstrap over test plants.

| arm | temporal representation | R@10 | vs spatial |
|---|---|---|---|
| spatial | none (N + taxonomy) | 0.3366 | — |
| curves | marginalized 52-wk curves, both towers | 0.3476 | +0.0110 **p = 0.044** |
| global_delta | + Σ min scalar | 0.3496 | +0.0130 **p = 0.028** |
| local_delta_hand | + per-cell Σ min, averaged over shared cells | 0.3543 | +0.0177 **p = 0.0056** |
| **local_learned** | **+ learned per-cell match (location-conditioned embedding)** | **0.3560** | **+0.0194 p = 0.0058** |
| N only | — | 0.1705 | −0.166 |
| congeneric transfer | — | 0.1716 | −0.165 |
| latent-trait SVD | — | 0.2538 | −0.083 |

**Every temporal arm significantly beats the spatial baseline** (p ≤ 0.044), and the ordering follows the
ladder: marginal < scalar < per-cell hand < per-cell learned. This is the **opposite** of the
full-universe result, where no encoding helped at all.

**The likely mechanism, and the paper's temporal claim:** in the full universe the pollinator side is a
GBIF weekly histogram pooled over all CONUS — observation-derived and aspatial. Here it is a model-derived
per-cell surface. Phenology appears to be usable only when both sides carry model-derived, spatially
resolved timing. That is the original ANTHEIA asymmetry hypothesis, tested properly for the first time.

**Two caveats that must not be dropped.**
1. ⚠️ **Source and locality are confounded in this ladder.** The `curves`/`global_delta` arms use GBIF
   `a_curves`; the two local arms use SDM surfaces. So `curves → local_*` changes *both* the data source
   and the granularity. An **SDM-marginalized-curves arm** is needed to separate them — queued next.
2. Differences *among* the temporal arms are within each other's CIs; only the comparisons against
   `spatial` are resolved. Do not claim `local_learned > curves` from this run.

---

## GloBI refuted-claims file (window 2, 09-02) — **DO NOT use as negatives**

Research recommended anti-joining GloBI's `refuted-interactions.tsv.gz` and using it as labelled
negatives — which would have solved our positive-unlabeled problem. **Verified against our data: the
file does not support that use.** (`eval/eda_refuted.py`; file at
`depot.globalbioticinteractions.org/snapshot/target/data/tsv/`, 525,879 rows, all `argumentTypeId=refute`.)

- 518,208 of 525,879 rows are `visitsFlowersOf` — our exact type. 248,059 distinct name pairs.
- **50.4% of our 62,832 positive edges also appear in the refuted set** (31,657 pairs). Identical result
  from the verbatim and interpreted files, so it is not an orientation artifact of our mapping.
- **The refuted rows are overwhelmingly correctly oriented animal→plant**: source/target kingdom is
  Animalia-or-Metazoa → Archaeplastida for 395,977 rows (76%). Only ~7,300 are genuine role violations
  (Metazoa→Metazoa or →Animalia).
- The stated reason for 517,498 of them is a single boilerplate string from a single 2020 EOL curation
  pass: *"Records of organisms other than plants having flower visitors are probably errors."* That
  reason does not describe the rows it is attached to.
- Contested pairs are ordinary plausible visits: *Bombus vosnesenskii* on *Lathyrus hirsutus*,
  *Coelioxys dolichos* on *Helenium autumnale*, a tachinid on *Daucus carota*. The research agent
  independently spotted *Trochilidae visitsFlowersOf Monarda fistulosa* (hummingbirds on bee balm).

**Conclusion:** this looks like a bulk flag — plausibly provider- or annotation-level rather than a
per-pair judgement — not a curated negative set. Treating it as negatives would label roughly half our
true positives as non-interactions. **Not adopted.** Worth one email to the GloBI maintainers to ask what
the flag actually denotes; until then the PU framing stands and negatives remain sampled.

Artifacts: `refuted_negatives_interp_v1.csv` (16,203 refuted pairs not among our positives) is retained
for inspection only, NOT for training or evaluation.

---

## Dataset rebuild (window 2, 09-02) — network constructed under the documented protocol

Built from the pinned GloBI snapshot (2026-08-26) following `docs/paper/dataset-construction.md`.
`scripts/build_edges.py` → `artifacts/v2/`, with `eval/eda_edges_v2.py` and `tests/test_edges_v2.py`
chained behind it (`scripts/run_build_pipeline.sh`).

**Network:** ~198k interactions, ~12.3k plant taxa, ~15.9k pollinator taxa, connectance ~0.11%.
Tier A (flower-visitation terms) is ~102k interactions with a clean profile — Hymenoptera 46%,
Lepidoptera 30%, Diptera 13%; Tier B adds the general-association records where Hemiptera enters.

**Yield ledger** (records in → out): 24.56M read → 5.61M after interaction-type selection → 2.89M in
region (1.31M coordinate-less, retained as metaweb) → 1.70M after role assignment and orientation
(33,177 orientation violations dropped) → 1.41M after identifier resolution → 1.36M after rank policy
→ **997k after deduplication (−26.8%)** → 958k after life-stage filter (38,853 immature dropped).

Genus-rank nodes contribute ~55k interactions (28% of the network). Source-holdout capacity improved
substantially: expert field networks 10,414 uniquely-sourced edges, specimen records 6,206.

**The test suite did its job.** Three failures on the first run, each a different kind of problem:
one bug (identifier collisions — `GBIF:1346141` denoted both *Calamagrostis* and the wasp genus
*Ammophila*, because GloBI identifier lists can carry several ids per record); one data-quality issue
(future-dated records); one incorrect test (tier semantics — `tier` is the best evidence supporting an
edge, so a Tier-A edge may legitimately also carry Tier-B records). After fixes, 11/12, with the two
remaining collisions being genuine cross-kingdom homonyms (*Lucilia* is both an Asteraceae genus and a
blowfly genus; likewise *Trifolium*). Those are distinct taxa sharing a name and are now keyed by
kingdom, which is the correct resolution rather than a suppression of the test.

Superseded results from the earlier name-matched network are archived under
`artifacts/archive/2026-08-31_superseded/` with a README stating why the numbers do not carry over.

---

## Feature-coverage analysis (window 2, 09-02) — **the modelled subgraph is 42% of the network**

`eval/eda_coverage.py` → `data/network/coverage_report.log`. Intersecting the constructed network with
the existing feature universes (6,466 plants from PhenoField/PPE; 24,939 pollinators from the GBIF
occurrence extract).

| | interactions | plants | pollinators |
|---|---|---|---|
| network | 198,327 | 11,031 | 15,012 |
| **both sides feature-covered** | **82,973 (41.8%)** | 4,243 | 6,217 |
| excluded at intersection | 115,354 | 9,839 | 12,874 |

Plant side covered 59.4%, pollinator side 68.3%.

**Three distinct causes, needing different responses.**

1. **Genus-rank taxa are 0% covered.** The feature files are keyed to species, so the 56,285
   interactions contributed by genus nodes cannot enter the modelled set as things stand. Either
   aggregate features over constituent species (a genus's occupancy is the union of its species'), or
   restrict modelling to species rank and report it as a stated scope limit.
2. **The pollinator pool has a genuine coverage gap, not just a scope boundary.** 44% of Hymenoptera
   taxa in the network (1,415 of 3,242) are absent from the feature file, along with 25% of Diptera and
   16% of Lepidoptera. These are flower visitors the pipeline found and the features do not cover —
   distinct from the taxa the pool deliberately excludes. Fully covering Hymenoptera alone would add
   **9,489 interactions**; Lepidoptera 3,475; Diptera 2,981.
3. **Orders outside the pool are entirely uncovered**, as designed: Hemiptera (2,934 taxa),
   Araneae (336), Orthoptera (238), Passeriformes (227), Odonata (120). Hemiptera would add 10,565
   interactions but are unlikely flower visitors — 20% of excluded interactions are Hemiptera against
   1.3% of the flower-visitation tier, so the tier system already separates them.

**Consequence for the plan.** Modelling cannot simply be re-indexed onto the new network: the feature
caches would need rebuilding from source occurrence data for roughly 7,700 additional pollinator taxa
and 6,400 additional plant taxa. Until then the modelled subgraph is 82,973 interactions over
4,243 plants — still larger than any previous version, but 42% of what was constructed.

---

## Feature-gap diagnosis (window 2, 09-02) — genus aggregation is worth ~35k interactions; feature rebuild is not

`eval/eda_feature_gap.py` → `data/network/feature_gap.log`. Two questions, both settled empirically.

**Q1 — can the missing pollinator features be rebuilt from the existing occurrence source? No.**
Of 6,361 species-rank pollinator taxa absent from the feature matrix, only **28** appear in the raw
occurrence file; the other 6,333 are absent from it entirely. The 28 recoverable taxa have a median of
2 occurrence records — too sparse to build a 52-week activity curve — and unlock just **203**
interactions. Closing the pollinator gap therefore requires a **new GBIF download** covering ~6,300
species, not a re-derivation from what we hold.

**Q2 — is genus-level feature aggregation feasible? Yes, and it is the largest available gain.**
64% of genus-rank plant nodes (998 of 1,548) and 46% of genus-rank pollinator nodes (607 of 1,313)
have at least one feature-covered congener, with a median of 2 and 6 covered species respectively.
Aggregating features over constituent species raises the modellable set from **82,973 to 117,880
interactions (+34,907, a 42% increase)**.

**Implication.** The two candidate remedies are very unequal. Genus aggregation is cheap, uses data
already held, and recovers 35k interactions; rebuilding features from the current source recovers 203.
The remaining gap — roughly 6,300 pollinator taxa the network documents but the occurrence extract
never covered — is a data-acquisition task, not a processing one.

---

## Ecological verification of the built network (window 2, 09-02) — passes every check

`eval/eda_selfcheck.py` → `data/network/selfcheck.log`. The checks specified in
`docs/paper/eda-questions.md`, run against the build.

**Wind-pollinated plants behave correctly, and the tier system is what makes them behave.**
In the flower-visitation tier, wind-pollinated families average degree 4.6 against 16.3 for
insect-pollinated ones (ratio 0.28); Poaceae is 0.27% of the tier with mean degree 3.9. Adding the
general-association tier raises the ratio to 0.64 and quadruples Poaceae's share to 1.07%, with oaks
(Fagaceae, degree 32.8) and birches rising sharply — these are insects recorded *on* trees, not
pollinating them. This is direct empirical support for tiering rather than merging the interaction
terms, and it gives the A-vs-A+B ablation a concrete interpretation.

**The fauna matches expectation.** Flower-visitation tier: Hymenoptera 46%, Lepidoptera 30%,
Diptera 13%, Coleoptera 7%, hummingbirds 1%. Adding tier B brings Hemiptera to 11%.

**Genus-only identifications sit where they should.** Concentrated in taxonomically difficult groups —
Asteraceae (229 genus-rank taxa), Poaceae (18.9% of its taxa), and among pollinators the mites
(Mesostigmata 88% genus-only, Trombidiformes 64%). Bees and butterflies are 8% and lower, consistent
with them being determinable from photographs.

**Known specialist mutualisms are present with the correct partners:** *Peponapis pruinosa* ×
*Cucurbita pepo* (squash bee), *Tegeticula synthetica* × *Yucca brevifolia* (Joshua tree yucca moth),
*Habropoda laboriosa* × *Vaccinium corymbosum* (southeastern blueberry bee), *Andrena vicina* ×
*Salix* — 305 interactions across the four checks. Real ecological structure survived the pipeline.

---

## Rebuilt pipeline end to end (09-04 → 09-06) — **93.5% coverage, and a leaderboard that survives its controls**

**Coverage.** The modelled universe is frozen at 11,031 plants × 13,124 pollinators, **180,349 of
192,945 interactions (93.5%)**, against 44.5% under the previous artifacts. Two changes did it:
Group B for the pollinator SDM was derived from the network rather than the legacy GloBI file
(572 → 7,770 zero-shot taxa), and 6,261 plants outside the e98 surface received text-conditioned
zero-shot curves.

**Node identity bug.** `build_edges.py` keyed nodes on the resolved identifier, so one taxon under
several checklist keys became several nodes (*Aceria* under six GBIF keys). Collapsing on
(label, rank): plants 12,347 → 11,031, pollinators 15,921 → 15,010, interactions 198,327 → 192,945.
**5,382 "interactions" were one interaction counted through different identifier pairs.**
`test_labels_unique_per_side` covers it.

**Curve marginalisation was wrong.** Averaging the per-cell `norm` weights a cell outside the range
as heavily as one at the range core. Marginalising absolute probability instead recovers the observed
peak week within two weeks for 60% of plants against 41%, reaching 0.583 histogram overlap — what
the supervised phenology head itself achieves (0.579). Phenology × abundance doubled, 0.079 → 0.158.

**Protocol.** Primary metric is normalised recall@10 (plain recall@10 is capped for the 29% of
plants with more than ten partners); scoring restricted to Tier A, training on both tiers. MAP added
alongside pooled PR-AUC — they evaluate different deliverables, a retrieval system queried one plant
at a time versus a metaweb thresholded once. Written up in `docs/paper/evaluation-protocol.md`.

### Leaderboard (validation, tier A, 663 plants)

| model | nR@10 | MAP | PR-AUC |
|---|---:|---:|---:|
| **taxonomy + spatial + per-cell (ours)** | **0.326** | 0.182 | **0.103** |
| congeneric transfer | 0.293 | 0.177 | 0.057 |
| truncated SVD + taxonomic | 0.292 | 0.185 | 0.095 |
| two-tower, multi-objective | 0.279 | 0.161 | 0.098 |
| pollinator popularity | 0.244 | 0.145 | 0.053 |
| phenology × abundance | 0.158 | 0.080 | 0.024 |
| co-occurrence count | 0.105 | 0.052 | 0.026 |

**Co-occurrence places sixth.** It scored 0.99 ROC-AUC on the old 139-pair benchmark; under cold
start on the corrected network the range-size shortcut does not survive.

### What the controls changed

**Species-permutation controls overturned the first Table 3.** Raw 52-week curves beat the scalar
overlap 0.261 vs 0.145 — but permuting the curves between species still scored 0.240. A 52-week
curve is near-unique, so extra columns act as a species identity code. Genuine margin: **+0.021, not
+0.115.** The per-week product f·a scores 0.145, *below* its own permuted control. Spatial survives:
PCA embedding 0.189 against 0.147 permuted, a real +0.042. **Adding scalars buys nothing on either
axis** — Jaccard over cell count +0.0000 (p=0.98), seven overlap statistics over one +0.0006 (p=0.88).

**Why the temporal axis is weak.** True pairs overlap 0.590, random pairs 0.495 (Cohen's d 0.48).
CONUS phenology is dominated by a shared summer peak — 51% of plants peak in weeks 18–30 — so
marginal phenological overlap barely discriminates between candidates *at any encoding*.

**Per-cell recovers it.** Exact per-cell co-activity (Σ over cells and weeks of the two surfaces
multiplied) added to the ranker: 0.318 → **0.326** nR@10 and 0.088 → **0.103** PR-AUC, against a
species-permuted control at 0.313/0.090. Real matching, and it helps calibration more than the top
of the ranking.

**Why the learned surface encoder failed.** A randomly initialised basis over 3,335 cells, fit from
132k interactions, scored 0.255 — worse than no encoder. A shared rank-256 basis from randomized SVD
captures 92% of surface variance and reproduces the exact co-activity at **Pearson 1.000, Spearman
0.946**, so the structure is there and cheap; the encoder simply could not find it from scratch.

**Robustness.** Species-rank-only evaluation (567 plants) preserves the ordering: ours 0.300,
svd 0.289, congeneric 0.277. Genus nodes are not carrying the result.

### Fixes that changed published numbers

- `congeneric` and `svd_taxonomic` resolved families through `store.cfg`, absent on `UniverseStore`,
  so both ran genus-only with every family `"UNK"`. congeneric 0.307 → 0.293, svd 0.289 → 0.292.
- Two-tower training stalled at loss 4.52 from epoch 2 because sampled negatives included the
  plant's own recorded partners. Masking accidental hits reaches 2.70.
- A binary term alongside the softmax took two-tower PR-AUC 0.054 → 0.098 with retrieval unchanged:
  the softmax orders within a plant and says nothing about comparability between plants.

## Identity panel (09-06) — **BioCLIP-2 text embeddings add nothing in a hand-built form**

Prompts are the bare binomial, no order or family string, so the embedding cannot smuggle in the
taxonomy the explicit row is built from. The implicit row scores a pair by the cosine between the
plant's text vector and the *prototype* of the plants a pollinator is recorded visiting — the
text-space analogue of congeneric transfer.

| | representation | nR@10 | MAP | PR-AUC |
|---|---|---:|---:|---:|
| explicit | taxonomic affinity (genus, family) | **0.254** | 0.134 | 0.061 |
| implicit | species-name embedding prototype | 0.136 | 0.081 | 0.055 |
| *control* | *same, species-permuted* | *0.141* | *0.064* | *0.023* |
| both | affinity + name embedding | 0.238 | 0.131 | 0.061 |
| full | ours + name embedding | 0.313 | 0.172 | 0.094 |
| full | ours (reference) | **0.320** | 0.181 | 0.100 |

**The prototype scores below its own permuted control** (margin −0.005, p=0.18), and adding it to the
full model *costs* 0.007. As a feature in this form it carries no species-specific signal.

**The embeddings themselves are fine.** Random plant pairs sit at cosine 0.427, same-genus pairs at
0.738 — they encode taxonomy well. The failure is the prototype: a pollinator recorded on fifty
plants across many families has a centroid near the global mean, so the cosine to it is nearly
constant. The affinity table, for all its brittleness, is an exact lookup.

This is the mirror of the surface result. There, the hand-computed per-cell feature worked and the
learned encoder failed; here the hand-built prototype fails and the remaining test is the learned
bilinear form — the embedding as a tower input, where the model chooses the projection.

**Exact vs projected per-cell overlap** costs about 0.006 nR@10 (0.326 exact against 0.320 via the
rank-256 basis), so the projection is the right choice inside a training loop and the exact feature
is worth keeping for the final model.

## Why the neural pair ranker trails the booster (09-07) — **it is the tabular inductive bias, not the implementation**

Seven complete training runs, each differing from the reference in exactly one design decision, same
seed, split, features and evaluation; differences by paired bootstrap over the same 663 plants.

| configuration | nR@10 | Δ vs reference | p |
|---|---:|---:|---:|
| reference (per-feature standardisation) | 0.2806 | — | |
| norm = LayerNorm | 0.2867 | +0.0061 | 0.37 |
| norm = none | 0.2762 | −0.0044 | 0.52 |
| **no logQ correction** | **0.1915** | **−0.0891** | **<0.001** |
| loss = softmax only | 0.2767 | −0.0039 | 0.45 |
| **loss = binary only** | **0.1468** | **−0.1338** | **<0.001** |
| no pollinator embedding | 0.2788 | −0.0018 | 0.73 |

**Four hypotheses, all refuted.** Feature normalisation does not matter, though the scales span
0.03 to 16,481 and skews reach 20.8 — LayerNorm, per-feature standardisation and none are within
noise of one another. The logQ correction is not discarding useful popularity signal; removing it
costs 0.089, because without it a popular pollinator appears as a sampled negative far more often
than a rare one and the model learns to penalise exactly the taxa most likely to be true partners.
The pollinator embedding neither helps nor overfits. The binary term adds nothing on either
objective for this architecture (softmax-only PR-AUC 0.1190 against 0.1163 with both), unlike the
two-tower where it took PR-AUC from 0.054 to 0.098.

**What remains is the model class.** Every implementation choice is neutral or already correct, and
the gap to the booster (0.281 against 0.326) persists. This is the documented behaviour of
gradient-boosted trees against neural networks on tabular features (Grinsztajn, Oyallon & Varoquaux
2022, NeurIPS Datasets and Benchmarks): neural networks are biased toward smooth functions, and the
strongest feature here is maximally non-smooth — taxonomic affinity is zero for almost every pair
and jumps sharply for a handful, which is exactly the irregular target trees fit natively.

**Consequence for the architecture.** A neural model will not win by fitting the same tabular
features better. It has to express something the booster structurally cannot. The candidate is a
learned metric over the shared niche basis: the booster sees one scalar, the inner product of two
256-dimensional surface projections, and cannot form a 256×256 bilinear map over them.

### The encounter term does not work as a backbone

A neutral model of the form P(interact) = P(encounter) × P(interact | encounter) is attractive
because the encounter term is computable analytically from two independently fitted distribution
models. It fails empirically: as a standalone ranker the per-cell encounter term reaches 0.0109
nR@10, against 0.1046 for a plain co-occurrence count, because it scales with range size and a
widespread pollinator co-occurs with everything. Every prevalence normalisation makes it worse —
lift 0.0053, PMI 0.0053, cosine 0.0176, square-root normalisation 0.0309.

Prevalence is not a nuisance to be divided out here; it is a large and genuinely predictive part of
the signal, which the logQ result says independently. The encounter term earns its place as a
correction alongside taxonomy and prevalence, worth +0.013 nR@10 and +0.013 PR-AUC against a
species-permuted control, not as a multiplicative backbone.

## Architecture search on the embedding model (09-07) — **one positive finding, six negatives**

Every row below is a complete 25-epoch run on the 663 validation plants, tier A, with a paired
bootstrap against the stated reference. Smoke runs on 200 plants proved unreliable and were dropped
as a basis for decisions after one over-claimed by 0.10 nrecall@10.

| change | nR@10 | Δ vs reference | p |
|---|---:|---:|---:|
| embedding model, deep only (reference) | 0.2920 | — | |
| Linnaean hierarchy prompts | 0.2949 | +0.0029 | 0.80 |
| wide path, single cross + degree | 0.3055 | +0.0135 | 0.26 |
| wide path + hierarchy prompts | 0.3108 | +0.0188 | 0.12 |
| **hierarchical back-off crosses** | **0.2699** | **−0.0221** | **0.036** |
| back-off + degree | 0.2865 | −0.0054 | 0.72 |
| back-off + single cross + degree | 0.2904 | −0.0016 | 0.92 |
| **DCN-V2, 2 cross layers** | **0.2628** | **−0.0292** | **0.003** |
| explicit affinity through the MLP | 0.2911 | −0.0008 | 0.99 |
| affinity through MLP + genus context | 0.2953 | +0.0033 | 0.78 |

**The wide path does not reproduce.** A 200-plant four-epoch smoke gave 0.4048; the full run gives
+0.0135 at p=0.26, and pooled PR-AUC falls 0.165 → 0.144. Routing the affinity *through* the MLP
instead leaves retrieval untouched and collapses PR-AUC to 0.073 — the Wide & Deep prediction holds
qualitatively, but the magnitude does not.

**Back-off crosses fail, and the motivating measurement was sound.** The memorised cross has 96,651
occupied cells at median count 1 and is empty for 57% of pollinators; coarser crosses reach median
count 12. Adding all five granularities costs 0.022 nrecall@10 (p=0.036). Rare-partner recall@10
moves from 0.0000 to 0.0114 — the mechanism does what it was built to do — but the coarse crosses
add more noise than the rare partners are worth.

**DCN-V2 hurts.** Learned bounded-degree crossing over the encoded pair is 0.029 worse than no
crossing (p=0.003), so on this data the failure of hand-chosen crosses is not a feature-engineering
problem that automation solves.

### The positive finding: the spatio-temporal signal is conditional, not marginal

Held-out (fit on half the validation plants, evaluated on the other half), incremental AUC of
per-cell co-activity over popularity and range overlap:

| candidate head | pop + N | + per-cell | Δ |
|---|---:|---:|---:|
| all 13,124 | 0.823 | 0.879 | +0.057 |
| top 1,000 | 0.682 | 0.805 | +0.123 |
| top 200 | 0.630 | 0.756 | **+0.125** |

The increment **grows** as the head narrows. Ranked on its own the same feature looks useless —
AUC 0.665 inside its own top 200, and only 25.9% of a plant's partners in that top 200 — because its
own head fills with widespread, heavily recorded pollinators.

**This explains the run of null results.** The encounter backbone (0.011 as a ranker), the niche
bilinear at every rank, the marginal curves that failed their permutation control: all used the
signal marginally, where prevalence swamps it. An earlier version of this test was in-sample; redone
on disjoint plants the effect grew rather than shrank.

Residualising the spatial terms against prevalence is null on the boosted ranker (−0.0014, p=0.75),
as expected — trees condition natively by splitting on prevalence first. Prevalence-matched
negatives, which remove the route by construction, are the live test.

### Inputs settled

- **z_static is redundant**: 99.2% linearly predictable from the BioCLIP-2 text embedding on held-out
  species. e98's species conditioning *is* that embedding, which is also why its text-conditioned
  zero-shot head works. The LE-SINR `spec` head likewise maps text to the species vector.
- **BioCLIP-2 hierarchy prompts are its native format** — seven text sequences per image, one per
  Linnaean rank — but neutral here, because bare binomials already place congeners at cosine 0.716.
- **GBOTB.extended places 94.0% of the plant universe** (59.3% exact tips, 94.7% genus present), so
  plant-side phylogeny is feasible; there is no comparable branch-length tree for 13,124 insects.
- **Image centroids are impractical.** TreeOfLife-200M is ordered by kingdom — a first attempt on
  shards 0–31 hit BIOSCAN insects and returned 41/11,031 plants, a sampling artifact. Strided across
  all 933 shards, 48 shards give 1,620/11,031 plants and 736/13,124 pollinators (9.8% overall). Full
  coverage needs most of 455 GB, which is not worth a feature that would still be missing on the
  majority of taxa.
- **Post-hoc calibration cannot change PR-AUC**: Platt is monotone and average precision is
  rank-based, so it returns a bit-identical value. Pooled PR-AUC measures cross-plant *ordinal*
  comparability, not probability calibration, and the write-up should say so.

## Prevalence is signal, not confound to be removed (09-07) — **three independent confirmations**

Per-cell co-activity adds 0.125 held-out AUC at the top 200 *given* popularity and range overlap,
so the conditional signal is real. The inference drawn from it -- that the model should be forced onto
that conditioning -- is wrong, and three separate interventions say so:

| intervention | effect on nR@10 | p |
|---|---:|---:|
| remove the logQ correction | −0.0891 | <0.001 |
| normalise the encounter term by prevalence (lift / PMI / cosine) | 0.011 → 0.005 as a ranker | — |
| **50% degree-matched negatives** | **−0.0391** | **<0.001** |
| **100% degree-matched negatives** | **−0.0478** | **<0.001** |

Matching negatives to their positive's recorded degree removes the route by which the loss can be
satisfied with prevalence. It costs 0.048 nrecall@10. The conditional signal it exposes is worth less
than the prevalence signal it hides, and the models already carry both features, so the incremental
AUC was being realised without the intervention.

The general statement, now supported three ways: **on this data prevalence is a large and genuinely
predictive component, and any method whose selling point is removing abundance confounding loses
accuracy.** That is a constraint on what can be claimed as novel, and it is worth stating in the paper
because the ecological literature treats co-occurrence-driven abundance as a nuisance to be
controlled for.

### The architecture search is closed

| change | Δ nR@10 vs the embedding model | p |
|---|---:|---:|
| DCN-V2, 2 cross layers | −0.0292 | 0.003 |
| DCN-V2, 4 cross layers | −0.0390 | <0.001 |
| back-off crosses | −0.0221 | 0.036 |
| degree-matched negatives | −0.0478 | <0.001 |
| genus-context attention + tier head | −0.0079 | — |
| learned niche metric (bilinear, ranks 16/64/128) | +0.002 / −0.010 / +0.004 | 0.76 / 0.09 / 0.54 |
| text bilinear (ranks 32/64/128) | +0.001 / −0.009 / −0.009 | 0.82 / 0.17 / 0.20 |
| wide path, single cross + degree | +0.0135 | 0.26 |

Nothing tried beats the two standing results: the boosted ranker at 0.3199-0.3259 nrecall@10 and the
embedding model at 0.1647 PR-AUC. Further search on this feature set is not warranted.

## Error analysis of the boosted ranker (09-07) — **the failure mode is an unseen genus**

Ten architecture variants had failed before this was run, which was the wrong order.

**Misses are catastrophic, not near.** Of 11,489 held-out partners over 13,124 candidates:

| rank | share |
|---|---:|
| 1–10 | 10.9% |
| 11–50 | 14.4% |
| 51–200 | 17.3% |
| 201–1,000 | 16.9% |
| **1,001–13,124** | **40.5%** |

Median rank 333. The model is not almost right; 40% of partners are nowhere near the top.

**One stratum dominates the error.**

| stratum | nrecall@10 | n |
|---|---:|---:|
| plant genus seen in training | **0.351** | 584 |
| **plant genus unseen** | **0.086** | 79 |
| feature source direct | 0.323 | 399 |
| feature source zeroshot_text | 0.315 | 264 |
| degree 1–2 | 0.273 | 247 |
| degree 51+ | 0.755 | 42 |

A plant whose genus never appears in training scores **four times worse**. By contrast the feature
source barely matters — 0.323 direct against 0.315 zero-shot — so the modelled surfaces and text
embeddings serve unobserved taxa as well as observed ones. The weakness is not coverage of the
inputs; it is that the taxonomic affinity table, the strongest single signal, is empty for an unseen
genus and the model falls back to little more than popularity.

**The spatio-temporal signal does not separate the misses.** Missed partners sit at the 0.776
percentile of their plant's per-cell co-activity against 0.943 for found partners — lower, but far
from absent. The signal is present for the partners we miss and is not decisive between them, which
is consistent with every marginal use of it failing.

**Part of the apparent error is unrecorded rather than wrong.** Of 5,377 top-10 predictions that are
not recorded partners, 15.2% are congeneric with a recorded partner of that plant and 35.1% are
confamilial. Under positive-unlabelled data that is an upper bound on how much of the measured error
is a labelling gap rather than a modelling one.

**Direction.** The highest-leverage remaining target is the unseen-genus stratum: 12% of test plants
at a quarter of the accuracy. That is exactly where a smooth taxonomic representation should beat a
lookup table, and where plant-side phylogeny with branch lengths would apply. It is also testable
without new inputs, by asking whether the embedding model — which has no affinity table and reaches
unseen genera through BioCLIP-2 text space — already wins that stratum.

## Genus-aware routing (09-07) — **the first significant gain over the boosted ranker**

The error analysis localised the failure to plants whose genus never appears in training: 79 of 663
held-out plants, at 0.086 nrecall@10 against 0.351 for the rest. Comparing methods within that
stratum shows the boosted ranker is the *worst* of the four there, not the best:

| model | overall | genus seen (n=584) | genus unseen (n=79) |
|---|---:|---:|---:|
| boosted ranker | 0.320 | **0.351** | 0.086 |
| embedding model | 0.292 | 0.313 | 0.134 |
| congeneric transfer | 0.293 | 0.304 | 0.213 |
| truncated SVD + taxonomic | 0.292 | 0.296 | **0.256** |

The composite affinity feature weights family at 1e-3 -- enough to break ties, not enough to carry a
prediction -- so when the genus lookup is empty the model falls back to little more than popularity,
while methods that impute from taxonomy degrade gracefully.

**Whether a plant's genus appears in training is known at inference and uses no test label**, so
routing on it is a deployable rule rather than an oracle. Scoring with the boosted ranker where the
genus is seen and truncated SVD where it is not:

| | nR@10 | PR-AUC |
|---|---:|---:|
| boosted ranker | 0.3200 | 0.1015 |
| **routed** | **0.3360** | **0.1032** |

**+0.0160 [+0.0079, +0.0254], p = 0.0004**, and pooled PR-AUC improves rather than degrading.

Two implementation notes. Splicing raw scores from two models destroys cross-plant comparability and
collapsed PR-AUC to 0.0245; mapping the SVD scores onto the boosted ranker's global score
distribution by quantile, fit on the seen-genus plants where both models apply, fixes it. And the
in-model alternative -- carrying family affinity as its own column with a genus-seen indicator --
lifts the target stratum from 0.086 to 0.122 but costs slightly on the rest, netting +0.0001
(p=0.97). The gap to 0.256 is not a missing family count; it is that the SVD latent factors carry
co-visitation structure a raw count does not.

This is a mixture of experts with a deterministic, interpretable gate, not a score-averaging
ensemble: exactly one model scores each plant, chosen by a property of that plant.

### Single-model alternative to routing: matrix-factorisation features

Routing between two models is inelegant, so the same signal was tried as features inside the boosted
ranker: the interaction matrix factorised at rank 12, the plant's latent vector imputed by taxonomy,
carried as its elementwise product with the pollinator latent.

| configuration | nR@10 | PR-AUC | genus seen | genus unseen |
|---|---:|---:|---:|---:|
| boosted ranker | 0.3199 | 0.0998 | 0.3514 | 0.0864 |
| + family fallback | 0.3200 | 0.1015 | 0.3468 | 0.1221 |
| + MF features, own latent for training plants | 0.2485 | 0.0775 | 0.2711 | 0.0816 |
| + MF features, leave-one-out imputed | 0.3093 | **0.1065** | 0.3353 | 0.1167 |
| **routed (booster / SVD by genus)** | **0.3360** | 0.1032 | — | — |

The first MF attempt cost 0.071 nrecall@10 (p<0.001) through a train/test feature shift of my own
making: a training plant received its *own* latent vector, which encodes its own partners, while a
held-out plant necessarily receives a taxonomy-imputed one, so the model learned to trust a feature
that degrades at inference. Imputing for every plant leave-one-out recovers most of it (0.2485 ->
0.3093) and gives the best PR-AUC of the boosted variants, but still does not reach the reference
(-0.0106, p=0.16) or the routed model.

Routing remains the only configuration that beats the boosted ranker on retrieval, and the reason is
visible in the strata: no single feature set recovers the 0.256 that truncated SVD reaches on unseen
genera while keeping the 0.351 the booster reaches on seen ones.

## Tier A against Tier A+B (09-07) — **the method ranking reverses**

Scoring the same models on the flower-visitation tier and on both tiers together is a robustness
check, and several methods fail it.

| model | A nR@10 | A+B nR@10 | A PR | A+B PR |
|---|---:|---:|---:|---:|
| boosted ranker | **0.3197** | 0.2897 | 0.0998 | 0.0811 |
| congeneric transfer | 0.2930 | **0.3046** | 0.0568 | 0.0511 |
| truncated SVD + taxonomic | 0.2915 | 0.2163 | 0.0951 | 0.0719 |
| embedding model | 0.2920 | 0.2779 | **0.1586** | **0.1294** |
| neural pair ranker | 0.2806 | 0.2659 | 0.1163 | 0.1007 |
| two-tower | 0.2786 | 0.2666 | 0.0994 | 0.0817 |

**Congeneric transfer is the only method that improves when the general-association tier is added**,
and it takes first place there. Truncated SVD falls hardest, 0.2915 to 0.2163, which is consistent
with its mechanism: it factorises the interaction matrix, and Tier B adds edges recording that an
insect was found on a plant rather than visiting its flowers, so the matrix it factorises becomes
noisier. Pure taxonomy is indifferent to that.

The boosted ranker keeps first place on Tier A and loses it on A+B. The embedding model is the most
stable of the learned models and keeps the best PR-AUC under both tiers.

**Consequence for the paper.** A single headline model cannot be claimed without stating the tier.
The protocol already fixes Tier A as the scoring tier on evidence-quality grounds, decided before
these numbers existed, so the reported result stands -- but the reversal belongs in the supplement as
a robustness check, and it is a point in favour of reporting both tiers rather than one.

## TabICL against the boosted ranker (09-07) — **the tabular premise holds for retrieval, not for calibration**

The claim that trees beat neural networks on these features rested on Grinsztajn, Oyallon &
Varoquaux (2022), and tabular foundation models have moved since -- TabPFN v2 (*Nature*, 2025), v3
(2026), and TabICL, which scales in-context learning past 100k rows. Run on identical features, the
same split and the same metrics, with a 50,000-row labelled context and all 663 x 13,124 pairs scored:

| | nR@10 | MAP | PR-AUC |
|---|---:|---:|---:|
| boosted ranker | **0.3199** | **0.1806** | 0.0998 |
| TabICL | 0.3126 [0.2875, 0.3390] | 0.1763 | **0.1104** |

TabICL trails on retrieval and leads on pooled PR-AUC, which is the same split the learned
representations show against the booster. The framing survives: on this task, for ranking the head of
a candidate list, a boosted tree remains the stronger model on hand-built features even against a
2026 foundation model. Inference cost is not close -- three hours against seventy seconds.

## Plant phylogeny (09-07) — **smoothing the taxonomic signal destroys it, for the third time**

GBOTB.extended places 10,372 of 11,031 universe plants (5,620 exact tips, 4,752 bound to their
genus's most recent common ancestor). Affinity was then smoothed over patristic distance,
weighting each training plant exp(-d/tau), which is congeneric transfer made continuous.

| bandwidth | nR@10 | PR-AUC | genus seen | genus unseen | Δ vs reference | p |
|---|---:|---:|---:|---:|---:|---:|
| boosted ranker | 0.3199 | 0.0998 | 0.3514 | 0.0864 | — | |
| tau = 50 Myr | 0.2931 | 0.0817 | 0.3198 | 0.0956 | −0.0268 | 0.0004 |
| tau = 100 Myr | 0.3008 | 0.0865 | 0.3278 | 0.1011 | −0.0191 | 0.0068 |
| tau = 200 Myr | 0.3022 | 0.0912 | 0.3282 | 0.1095 | −0.0177 | 0.0088 |

Every bandwidth hurts. The target stratum improves slightly -- 0.086 to 0.110 at the widest
bandwidth -- and the cost falls on the 88% of plants whose genus *is* seen, where a sharp lookup was
working.

The diagnosis is in the density: the smoothed affinity has 119 million non-zero cells against roughly
97,000 for the genus cross. **The taxonomic signal's value is its sparsity.** This is now the third
independent confirmation, after hierarchical back-off crosses (−0.0221, p=0.036) and BioCLIP-2
nearest-neighbour smoothing of the same table (−0.019, and its species-permuted control did equal
damage). Three different smoothers, three losses.

That is a real finding rather than three failures: methods that generalise taxonomy by averaging
over relatives lose more on the well-covered majority than they recover on the sparse tail. The only
construction that helped was routing -- keeping the sharp model where it works and switching models
entirely where it does not.

---

## Field-embedding phase (09-07) — **species as learned spatio-temporal influence**

Direction change agreed with Dan: the boosted ranker is the ceiling of scalar features, not the
method. The pollinator SDM (`pipelines/sdm/build_deliverable.py`) is already a spatio-temporal
SINR -- Fourier(lat,lon) + Climplicit monthly climate + Fourier(week) -> shared encoder -> per-species
linear head. Its `cls.weight` rows [5,822 x 256] are each species' learned spatio-temporal influence
and had never been used; only the model's *output* surfaces were, via an SVD grid projection.

**Diagnostics that motivated this (validation, tier A, 663 plants):**
- SVD surface projection is 64% (plants) / 84% (pollinators) unexplained by prevalence + taxonomy +
  BioCLIP-2 text combined: a genuine third input axis.
- Per-cell co-activity is 87.5% independent of N and range size, but 39% is the product of the two
  surface masses (model confidence x extent), and separating that scale term raises its solo nR@10
  0.011 -> 0.029. Used alone it is a filter (AUC 0.851 > N's 0.782), not a ranker (popularity 0.244).
- **The SDM head vector is 4.7% explained by BioCLIP-2 text** (held-out ridge R^2, 5,354 species;
  mean cosine 0.45). It is not taxonomy. Consequence: text imputation for the 7,770 zero-shot
  pollinators is weak, so the "no impute" arm is the clean one.

**Step 1 (running):** `scripts/build_field_embeddings.py` -> `data/features/poll_field.npy`;
`eval/run_field_swap.py`, six arms x seeds {42, 0, 1}, per-plant nR@10 saved for pooling, split by
genus seen/unseen. Reference config = embedding model, no genus context / tier head, 25 epochs.
Plant side unchanged (no plant field model yet).

**Step 2 (planned):** one spatio-temporal SINR over both kingdoms (plants from
`phenofield_cache/.../inat_train.npz`, pollinators from `pollinator_occ_gbifv3.npz`), ablating only
the negative-sampling scheme: uniform background / target-group spatial / target-group
spatio-temporal / same-location-different-species. The current SDM is accidentally target-group
(negatives only at observed points, no random background). Readout = unseen-genus stratum.

**Step 3 (planned):** encounter integral by Monte Carlo (parameter-free), then the pair head.

### Step 2 result — sampling-scheme ablation of the joint field model (09-07)

`pipelines/sdm/train_joint_field.py`: one SINR encoder (Fourier lat/lon + Climplicit monthly climate of
the cell + Fourier week, 276-D) over both kingdoms, 12,268 species (6,446 plants, flowering records
only; 5,822 pollinators), 2.99M rows capped at 1,000 per species, 20 epochs, seed 0. Only the
all-negative background distribution differs between arms. Held-out = 5% of rows; ranking within
kingdom. `eval/run_field_encounter.py` scores the trained sub-universe (379 val plants, 5,354 candidate
pollinators, 5,750 partners) -- untrained taxa have the zero vector, i.e. presence 0.5 everywhere, so
they cannot be candidates; nulls are recomputed on the same subset. **Not comparable to full-universe
tables.**

| scheme | held-out top-10 | direction-only effort R^2 (plants / polls) | encounter alone nR@10 | encounter re-ranking popularity's top-200: nR@50 / PR | cosine re-rank: nR@10 / unseen |
|---|---:|---:|---:|---:|---:|
| popularity (same subset) | -- | -- | 0.2346 | 0.3528 / 0.0663 | 0.2346 / 0.2603 |
| N (same subset) | -- | -- | 0.1697 | -- | -- |
| uniform | 0.158 | 0.519 / 0.779 | 0.0629 | **0.4475 / 0.0980** | 0.1364 / 0.0536 |
| tg_spatial | 0.185 | 0.586 / 0.796 | **0.0779** | 0.4183 / 0.0952 | 0.1574 / 0.1122 |
| tg_spatiotemporal | **0.193** | 0.615 / 0.820 | 0.0613 | 0.4259 / 0.0903 | 0.1607 / 0.1013 |
| slds | 0.171 | 0.662 / 0.836 | 0.0715 | 0.4190 / 0.0897 | **0.1865 / 0.1403** |

Three readings, two of them negative.

1. **The sampling scheme changes the SDM, not the retrieval.** Effort-matched background improves
   held-out species discrimination monotonically (0.158 -> 0.185 -> 0.193), as the SINR literature
   predicts. Every retrieval column moves by less than the gap to N.
2. **The "effort cancels" hypothesis is refuted in its stated form.** The unit-normalised species
   vector predicts log record count *better* under target-group backgrounds (plants 0.52 -> 0.62,
   polls 0.78 -> 0.84), not worse. Against a background that already follows effort, a heavily
   recorded species is exactly the one that exceeds the effort surface, so effort-relative prevalence
   is what the direction encodes. `corr(|u|, log n_obs)` is +0.70 to +0.76 under every scheme and is
   not diagnostic.
3. **Filter, not ranker -- third time.** Alone, the exact encounter integral ranks at 0.061-0.078
   (N: 0.170). Re-ranking popularity's own top 200 by it lifts nR@50 from 0.353 to 0.42-0.45 and
   PR-AUC from 0.066 to 0.090-0.098 while costing at k=10. This is the same signature as the SVD
   grid surfaces (AUC 0.851, nR@10 0.011) and the SDM pollinator head (step 1): the spatio-temporal
   field orders the list below the head and cannot pick the head.

Step 1 (SDM pollinator head into the embedding model, three seeds, pooled below once seed 1 lands)
was neutral in the full model and worse than the SVD projection without text -- the two sides were
in different coordinate systems. The joint model removes that; the both-sides arms are running
(`results/joint_field_swap_*`).

### Step 1 result, pooled — SDM pollinator head into the embedding model (3 seeds, 09-07)

`eval/pool_field_swap.py` -> `results/field_swap_pooled.csv`. Seed-averaged per-plant nR@10, paired
bootstrap over 663 validation plants (tier A). Reference = embedding model, blocks text+surface+pca+scale.

| arm | nR@10 | genus seen | genus unseen | vs reference | unseen-genus contrast |
|---|---:|---:|---:|---:|---:|
| reference | 0.2820 | 0.3040 | 0.1193 | -- | -- |
| field replaces surface (pollinator) | 0.2727 | 0.2907 | 0.1393 | -0.0093 p=0.15 | +0.020 p=0.23 |
| field added | 0.2820 | 0.3018 | 0.1360 | +0.0001 p=0.99 | +0.017 p=0.26 |
| field, trained rows only | 0.2836 | 0.3031 | 0.1392 | +0.0016 p=0.77 | +0.020 p=0.13 |
| no-text surface | 0.2196 | 0.2296 | 0.1459 | -0.0624 p<0.0001 | +0.027 p=0.29 |
| no-text field | 0.1991 | 0.2069 | 0.1420 | vs no-text surface **-0.0205 p<0.0001** | -0.004 p=0.81 |

- **Neutral in the full model, worse without text.** The SDM head vector adds nothing on top of the
  SVD grid projection and, as the only spatio-temporal input, is significantly worse than it. The two
  sides were in different coordinate systems (plant: grid basis; pollinator: SDM encoder), so the pair
  head's product and difference features had nothing to compare. That is the incommensurability
  problem restated at the representation level; the joint model (step 2/3) is the test of that reading.
- **A consistent but underpowered unseen-genus sign.** Every field arm lifts the 79 unseen-genus
  plants by +0.017 to +0.020 (p 0.13-0.26). Three seeds cannot resolve an effect of that size on 79
  plants; the pooled joint-field runs add the same plants again but are not independent evidence.
- **Text hurts where the genus is unseen.** Removing the BioCLIP-2 block costs 0.062 overall yet
  *raises* the unseen-genus stratum (0.119 -> 0.146). The identity signal misleads exactly where it
  has nothing to say -- consistent with the error analysis and with the booster's collapse there.

### Marginalisation test (09-07) — **same field, same grid: collapsing space or time before the product destroys the signal**

`eval/run_marginalisation_test.py` -> `results/marginalisation_test_val_tierA.csv`. From one joint field
model, each species' presence over the (cell, week) grid, P[c,w] = sigma(u.h(c,w)). Four expected
co-presence statistics of the *same two surfaces*, differing only in what is summed out before the
product: joint (nothing), space (weeks first = range overlap), time (cells first = phenology overlap),
scalar (both = prevalence product). Trained sub-universe: 379 val plants, 5,354 candidates, 5,750 partners.
Paired bootstrap, joint against each marginal, 5k resamples.

| statistic (uniform scheme) | alone: nR@10 / nR@50 | re-ranking popularity's top 200: nR@10 / nR@50 / PR-AUC | joint minus this, @50 (re-rank) |
|---|---:|---:|---:|
| popularity | 0.2346 / 0.3528 | -- / -- / 0.0663 | -- |
| **joint** (per cell, per week) | **0.0629 / 0.2208** | **0.2025 / 0.4475 / 0.0980** | -- |
| space marginal (range overlap) | 0.0280 / 0.1483 | 0.1491 / 0.4037 / 0.0857 | +0.044 [+0.028, +0.062] p<0.001 |
| time marginal (phenology overlap) | 0.0200 / 0.0853 | 0.0878 / 0.2676 / 0.0866 | +0.180 [+0.147, +0.214] p<0.001 |
| scalar (prevalence product) | 0.0067 / 0.0624 | 0.0602 / 0.2518 / 0.0737 | +0.196 [+0.161, +0.231] p<0.001 |

Same ordering under tg_spatiotemporal (joint 0.4259 > space 0.4064 > time 0.2252 > scalar 0.2081 at
nR@50, all p <= 0.002), and every @10 contrast is significant except scalar-alone under
tg_spatiotemporal (tie).

**What it establishes, with no confound left.** Earlier evidence for "marginalisation destroys the
signal" compared different data sources (GBIF curves vs SDM surfaces) or different models. Here the
model, the training data, the grid and the two surfaces are identical; only the order of summation
differs. The strict ladder joint > space > time > scalar is the paper's mechanistic claim in its
cleanest form.

**The phenology finding is now precise.** The time marginal -- the phenology-overlap statistic
ANTHEIA and the field's Delta/Schoener's-D constructs compute -- retains almost none of the joint
statistic's value (0.268 vs 0.448 at nR@50), while the space marginal retains most of it (0.404).
Phenology adds over range overlap *only* when it is kept per cell: joint minus space is +0.044
(p<0.001). That is the original ANTHEIA asymmetry hypothesis, tested properly, and the reason a
decade of marginal-curve overlap statistics have found little.

Caveat unchanged: this is the conditional (re-ranking) regime; used alone the joint statistic ranks
at 0.063 against popularity's 0.235.

### Step 3 result, pooled — joint two-kingdom field into the embedding model (3 seeds, 09-07)

`eval/pool_field_swap.py --preset joint` -> `results/joint_field_swap_pooled.csv`. Both sides now share
one encoder space (field on plants and pollinators, trained rows only; 41.5% / 40.8% of the universe
have a trained vector, the rest are zero and LayerNorm maps them to zero).

| arm | nR@10 | genus seen | genus unseen | unseen/seen | vs reference |
|---|---:|---:|---:|---:|---:|
| reference (text+surface+pca+scale) | 0.2820 | 0.3040 | 0.1193 | 0.39 | -- |
| joint field (uniform) replaces surface | 0.2795 | 0.3004 | 0.1243 | 0.41 | -0.0025 p=0.70 |
| joint field (uniform) + surface | 0.2737 | 0.2948 | 0.1177 | 0.40 | -0.0083 p=0.18 |
| joint field (tg_spatiotemporal) replaces surface | 0.2715 | 0.2906 | 0.1308 | 0.45 | -0.0104 p=0.10 |
| joint field (tg_spatiotemporal) + surface | 0.2853 | 0.3065 | 0.1287 | 0.42 | +0.0033 p=0.58 |
| no-text surface | 0.2196 | 0.2296 | 0.1459 | 0.64 | -- |
| no-text joint field (uniform) | 0.2048 | 0.2071 | **0.1877** | **0.91** | vs no-text surface -0.0148 p=0.14; unseen +0.042 p=0.21 |
| no-text joint field (tg_spatiotemporal) | 0.2075 | 0.2126 | 0.1693 | 0.80 | vs no-text surface -0.0121 p=0.25; unseen +0.023 p=0.56 |

- **Commensurability was not the bottleneck.** Putting both sides in one learned field space changes
  nothing in the full model (all p >= 0.10) and does not beat the SVD grid projection as the sole
  spatio-temporal input. Step 1's deficit (-0.021) closes to a non-significant -0.012 to -0.015, so the
  coordinate mismatch cost something, but the field carries no *ranking* information the surfaces lack.
- **Graceful degradation, descriptively.** With no identity input, the field model scores unseen-genus
  plants at 91% of its seen-genus level (0.188 vs 0.207); the reference is at 39%. Every arm carrying
  text collapses across the genus boundary. The contrast against the surface control (+0.042) is not
  significant on 79 plants and is reported as a pattern, not a result. On that stratum SVD-taxonomic
  (0.256) and popularity (0.234) remain better than any field arm.
- **Closes the field-embedding phase for retrieval.** Three representations of the niche axis (grid
  surfaces, SDM head, joint field), two sampling schemes, three seeds: none moves nR@10. The axis is
  real (sections above), and what it does is re-order the list below the head (marginalisation test).

### Marginalisation test on the production surfaces, full universe (09-07)

`scripts/build_surface_marginals.py` + `eval/run_marginalisation_surfaces.py` ->
`results/marginalisation_surfaces_val_tierA.csv`. Same four statistics, now from the PPE plant
surfaces and SDM pollinator surfaces that cover every taxon: 663 val plants, 13,124 candidates,
11,489 tier-A partners. The two sides come from different model families (a caveat for absolute
values, not for the within-pair contrast).

| statistic | alone nR@10 / nR@50 | re-rank popularity top-200: nR@10 / nR@50 / PR-AUC | joint minus this @50 (re-rank) | unseen-genus @10 (re-rank) |
|---|---:|---:|---:|---:|
| popularity | 0.2442 / 0.3728 | -- / -- / 0.0526 | -- | 0.2338 |
| **joint** | **0.0109 / 0.0808** | **0.1739 / 0.3595 / 0.0770** | -- | **0.1023** |
| space | 0.0079 / 0.0507 | 0.1393 / 0.3442 / 0.0676 | +0.015 p=0.008 | 0.0782 (p=0.002) |
| time | 0.0042 / 0.0244 | 0.0844 / 0.2553 / 0.0702 | +0.104 p<0.001 | 0.0430 (p=0.020) |
| scalar | 0.0012 / 0.0126 | 0.0498 / 0.2000 / 0.0612 | +0.160 p<0.001 | 0.0259 (p=0.003) |

- **The ladder replicates on the full protocol universe:** joint > space > time > scalar on every
  recall column, all joint-vs-marginal contrasts p <= 0.008, including on the unseen-genus stratum.
  On pooled PR-AUC alone the two marginals swap (time 0.070, space 0.068); joint stays first.
- **PR-AUC gain survives, the nR@50 gain does not.** Re-ranking popularity's top 200 by the joint
  statistic raises pooled PR-AUC 0.053 -> 0.077 (+46%) but leaves nR@50 at 0.360 against popularity's
  0.373 -- on the production surfaces the niche term improves cross-plant ordering without improving
  within-plant recall at 50. The joint-field model did both (0.353 -> 0.448 on its sub-universe), which
  is the one place the learned, commensurable field beat the production surfaces.
- **The time marginal is the weakest again** (0.255 vs space 0.344 at nR@50): phenology summed over
  space carries less than range overlap and far less than the per-cell product.

### Block ablation of the embedding model for PR-AUC (3 seeds, 09-08) — **the spatio-temporal blocks carry a fifth of the PR-AUC and little of the ranking**

`eval/run_field_swap.py --preset ablate --save-scores`, `eval/pool_prauc.py` (per-seed metrics averaged;
paired plant bootstrap within seed, 1,000 resamples) -> `results/block_ablation_prauc_pooled.csv`,
`results/block_ablation_val_tierA_s*.csv`. Validation, tier A, 663 plants, all 13,124 candidates.
Chance PR-AUC = prevalence = 0.00132.

| arm | PR-AUC | AUROC | PR-AUC re-expressed at 1:3 / 1:1 | Delta PR-AUC vs full | nR@10 | Delta nR@10 vs full |
|---|---:|---:|---:|---:|---:|---:|
| **full** (text+surface+pca+scale) | **0.159** | 0.954 | 0.899 / 0.957 | -- | 0.282 | -- |
| -surface | 0.145 | 0.952 | 0.890 / 0.954 | **-0.014 p<0.001** | 0.272 | -0.010 p=0.14 |
| -pca | 0.152 | 0.951 | 0.893 / 0.954 | -0.007 p=0.036 | 0.276 | -0.006 p=0.27 |
| text + scale only | 0.129 | 0.942 | 0.872 / 0.945 | **-0.030 p<0.001** | 0.255 | -0.027 p<0.001 |
| -text | 0.092 | 0.918 | 0.826 / 0.922 | -0.068 p<0.001 | 0.220 | -0.062 p<0.001 |
| surface + scale only | 0.084 | 0.909 | 0.810 / 0.914 | -0.075 p<0.001 | 0.202 | -0.080 p<0.001 |

- The two spatial blocks together are worth 0.030 of 0.159 PR-AUC (19%), each significant on its own;
  for nR@10 they are individually non-significant and jointly worth 0.027. Identity carries 0.068 of the
  PR-AUC and all of the ranking gap. The two-task split is measurable inside one model.
- The prevalence re-expression (negatives re-weighted so positives are 25% / 50% of the data, computed
  from the full ranking rather than a sample) gives the numbers the ecological literature reports:
  0.90 / 0.96. Report all three with the prevalence stated.
- Note for pooling: averaging score matrices across seeds is an ensemble and inflated PR-AUC to 0.182;
  the table uses per-seed metrics averaged.

---

## Connectivity ladder — S0 and S1 log (09-08)

Plan: `docs/plan/connectivity-ladder-v1.md`. Harness: `eval/run_ladder.py` -> `runs/<model>/<cfg>/<split>/s<seed>/`
(scores, Y, per-query, metrics, config, git). Report: `eval/report_ladder.py`.

### S0 — shared encoders and caches

- **Field v2 (text-conditioned head) — gate FAILS on pollinators, passes on plants.** `pipelines/sdm/train_field_v2.py`,
  scheme tg_spatiotemporal, 20 epochs, 5% held-out rows + 5% held-out *species* (zero-shot).

  | variant | held-out top-10 (observed species) | zero-shot AUROC plants | zero-shot AUROC pollinators |
  |---|---:|---:|---:|
  | residual, drop 0.3 | 0.194 | 0.866 | 0.759 |
  | residual, drop 0.5 | 0.185 | 0.866 | 0.766 |
  | text only (no residual) | -- | -- | min 0.745 |

  Observed-species quality is unchanged from v1 (0.193), so the text head costs nothing there. The
  pollinator zero-shot ceiling is ~0.77 whatever the head: BioCLIP-2 text places an unseen insect's
  climate niche less well than a plant's. **DECISION FOR DAN (pre-registered gate 0.80):** (a) mixed
  sources -- field v2 for observed taxa, production PPE/SDM surfaces for zero-shot taxa, with a
  `field_source` stratum; (b) kingdom-specific -- field v2 for all plants (passes), production SDM for
  zero-shot pollinators; (c) lower the gate to 0.75 and use field v2 everywhere (one encoder, one scale,
  the stratum tracks the weaker zero-shot pollinators). I lean (c) for consistency with the shared-encoder
  premise, but it changes a pre-registered gate. Token caches (joint / space / time) are built for field v2
  (`scripts/build_field_tokens.py`), so M2 can run under any choice; (a)/(b) need a rebuild for zero-shot taxa.
- **Splits (frozen):** cold_plant (existing), cold_poll (existing `pollinators_75_10_15.json`, 9,843/1,312/1,969),
  cold_both (1,085 x 1,312 val, 947 tier-A positives), warm (75,777 / 10,103 / 15,155 edges among 8,123 x 9,843).
  `scripts/build_splits_battery.py`, `data/splits/battery_summary.json`.
- **Local networks:** `scripts/extract_local_networks.py` -> 91 (dataset, citation, locality) networks with >= 10 plants
  and >= 10 pollinators, all with coordinates; 71 inside the CONUS grid. Robertson 1929 (Carlinville IL, 421 x 682,
  9,772 pairs), LaManna (Montana), Clements 1923 (Colorado), 24 Guzman 2022 sites (BC). Within-network connectance
  0.096. **Design note:** these edges are in GloBI and hence in training, so the clean evaluation trains with the
  source held out (existing `--holdout-source` machinery) and scores its networks -- S4 work. Two historical
  networks (1923, 1929) are a temporal caveat.

### S1 — retriever loss sweep (in progress; bundles under runs/)

- Harness reproduces the earlier reference bit-for-bit (seed 42: AUPR 0.1647, nR@10 0.2920).
- **M1.1 (pooled BCE only, softmax weight 0) collapses: seed 42 AUPR 0.032, nR@10 0.108, AUROC 0.951.** The
  within-plant softmax is what gives the ranking its head; without it the binary term with 383 negatives per
  positive flattens the logits (the score-contrast failure already documented in `embednet.py`). This is a real
  result for the two-task story: a *purely* pooled objective is not the fix, the pooled metric still needs the
  within-plant term. **Queue re-ordered (plan §8.3):** phase A = loss sweep (softmax/bce weights), phase B =
  degree heads / PU / bilinear on the phase-A winner, since M1.3-M1.5 as first queued would have inherited the
  collapse.

**DECISION (Dan, 09-08): option (c).** The zero-shot gate is lowered to 0.75 and field v2 (text-conditioned
head, residual drop 0.3) is the presence source for every taxon in models 2 and 3; the `zeroshot` stratum
tracks the weaker zero-shot pollinators (AUROC 0.76). Noted as a possible later experiment: kingdom-specific
or mixed presence sources for zero-shot pollinators.

**DECISION (Dan, 09-08): connectivity metrics first.** Priority moves to the connectivity evaluation --
local-network completion with real non-interactions (per-network AUPR / AUROC), network-structure fidelity
(degree correlation, NODF at matched connectance) and, once probabilities are used, calibration. The
within-plant ranking task (nrecall@k) is secondary/appendix for this paper and no longer drives decisions.

### Concurrent work: NECTAR (Baiotto et al., bioRxiv 2026-04-01) — read, and turned into a baseline (09-08)

Summary in `docs/08-methods-literature.md`. Their inference rule is `p = (spatial overlap) x (phenological overlap)` gated by
a genus-level taxonomic constraint -- the marginal-product construction our ladder measures against -- evaluated by recall
of withheld pairs (53.8% vs 25.2% for the genus-constrained null), no negatives, no AUPR. Re-implemented on our inputs as
`nectar_like` (gated) and `nectar_ungated` (overlap product alone); both go through the cold-plant harness and the
local-network evaluation as comparative rows. **Possible test data (for Dan):** their Zenodo bundle is embargoed to
2026-10-01; after that, their species-level pairs from non-GloBI sources would be an external positive-only California test
set. The predicted metaweb is model output, not ground truth, and should not be used as labels.

### S1 phase A verdict (09-08) — **the reference weighting stays; no pooled-objective variant improves AUPR**

`eval/report_ladder.py`, 3 seeds, paired plant bootstrap (300 resamples) vs the reference (softmax 1.0, BCE 0.5):

| arm | AUPR | Delta AUPR vs reference | p | nR@10 |
|---|---:|---:|---:|---:|
| reference (softmax 1.0, BCE 0.5) | 0.159 | -- | -- | 0.282 |
| softmax 1.0, BCE 0.25 | 0.161 | +0.002 | 0.50 | 0.274 |
| softmax 1.0, BCE 2.0 | 0.151 | -0.009 | <0.001 | 0.276 |
| softmax 1.0, BCE 1.0 | 0.149 | -0.010 | <0.001 | 0.274 |
| softmax 0.25, BCE 1.0 | 0.147 | -0.013 | <0.001 | 0.289 |
| BCE only (softmax 0) | 0.033 | -0.126 | <0.001 | 0.110 |

Gate (Delta >= 0.01, p < 0.05) not met by any arm -> **retriever config = reference**. Reading: the within-plant
softmax is necessary even for the pooled metric (removing it collapses AUPR by 0.13); *more* binary weight
flattens logits and lowers AUPR; the reference sits at the optimum of this family. The plan's hypothesis "0.159 is
objective-limited" is rejected for this loss family. Phase B (degree heads, nnPU prior, concat+bilinear head) runs
on the reference base.

### Table 3 first pass — local-network completion (09-08) — **the ranking reverses on real non-interactions**

`eval/run_localnets.py` / `eval/make_localnet_table.py` -> `results/tables_localnet.md`. 91 surveyed networks (71 inside
the grid), every local-network pair removed from training, per-network AUPR (chance = connectance 0.135), predicted
network at matched connectance for structure. Learned rows are seed 42 only so far.

| method | mean AUPR [CI] | pooled AUROC | deg rho plants / polls | NODF pred (obs 36) | precision@L |
|---|---:|---:|---:|---:|---:|
| popularity | 0.192 [0.179, 0.207] | 0.621 | -0.02 / 0.27 | 92 | 0.209 |
| congeneric transfer | 0.222 [0.205, 0.242] | 0.610 | 0.10 / 0.32 | 70 | 0.239 |
| SVD + taxonomic | 0.179 | 0.607 | 0.13 / 0.28 | 76 | 0.199 |
| pair GBM (Pichler) | 0.203 | 0.605 | 0.07 / 0.34 | 83 | 0.218 |
| **boosted ranker / routed (ours)** | **0.222 [0.205, 0.239]** | 0.614 | **0.20 / 0.35** | 62 | **0.252** |
| Wide & Deep | 0.155 | 0.583 | 0.04 / 0.25 | 44 | 0.145 |
| embedding model (ours) | 0.163 [0.150, 0.177] | **0.632** | 0.00 / 0.26 | **43** | 0.161 |

**Reading.** Within a surveyed site, where every candidate pair co-occurs by construction, the trees and congeneric
transfer lead on per-network AUPR and on the head of the network (precision@L); the embedding model -- the AUPR leader on
the full universe -- falls *below popularity* on per-network AUPR while holding the best AUROC and the most realistic
nestedness. That is the filter-not-ranker signature again, now for the whole neural model: its ordering is good across
the block and weak at the top. Global prevalence/range signal, which powers pooled AUPR over 145M pairs, is worth much
less inside a site; sharp taxonomic affinity is worth more.

**LARGE DECISION FOR DAN (not taken):** if local-network completion is the connectivity headline, the boosted/routed model
is currently the best of ours on it, not the embedding model. The plan's model 2 (fusion re-ranker) uses the embedding
model as retriever; a booster-based retriever (or re-ranking the booster's candidates) may be the better connectivity
route. Pending: seeds 0/1 for the neural rows, the NECTAR-style rows, and the fusion / R-GCN rows before deciding.
Caveats: two historical networks (1923, 1929); 20 BC networks outside the grid; connectance 0.135 makes this a much
denser problem than the universe (0.0013).

**Diagnostic (seed 42, paired per network):** the boosted ranker beats the embedding model in **81 of 91 networks**
(median +0.048 AUPR), uniformly across datasets (Web of Life 0.235 vs 0.192; Guzman/LaManna 0.219 vs 0.156), network
sizes (small/medium/large terciles all show the gap) and in/out of the grid. It is not driven by the historical giants:
on Robertson 1929 (9,712 links, connectance 0.034) **every** method sits at chance (0.034-0.039), and Clements 1923
is only weakly predictable -- a century-old single-locality network is a caveat for the whole evaluation, not a
discriminator between models. Lift over connectance: booster 1.72x, popularity 1.52x, embedding model 1.24x. The
embedding model beats popularity in only 31 of 91 networks. Wide & Deep (affinity on a linear path) is worse still
(0.155), so simply exposing the affinity lookup to the neural model does not close the gap; the within-site deficit of
the neural pair models is systematic and unexplained for now.

### S2 first arm (09-08, seed 42, provisional) — **the identity-only re-ranker is the largest single gain so far**

`eval/run_fusion.py`, retriever = reference embedding model (recall@500 on val plants 0.684), fusion = CLS + two identity
tokens (no field tokens), 3 layers, residual re-ranking of the top-500, pooled BCE over positives + retriever hard
negatives + random negatives, 8 epochs.

| | AUPR | AUPR 1:3 / 1:1 | AUROC | nR@10 | nR@50 | unseen-genus nR@10 |
|---|---:|---:|---:|---:|---:|---:|
| retriever (reference embedding model), seed 42 | 0.165 | 0.901 / 0.958 | 0.955 | 0.292 | 0.432 | 0.134 |
| **M2.1 identity-only fusion re-ranker**, seed 42 | **0.197** | 0.907 / 0.960 | 0.956 | **0.329** | **0.481** | **0.176** |

The arm designed as the control ("attention over identity should be a wash") is +0.032 AUPR and +0.037 nR@10 over the
retriever. Since it carries no field tokens, the gain is the **re-ranking mechanism**: training on the retriever's own
top-500 confusers (hard negatives) with a residual head, under a pooled objective. This is the standard retrieve-then-
re-rank benefit, and it is what every other arm must now beat. Leakage check: retriever and fusion fit on train plants
only; val plants and their pairs never enter training; the residual head starts at the retriever's scores. Caveats:
one seed (0 and 1 queued); retriever recall@500 0.684 caps what re-ranking can recover (plan: K=1000 if < 0.85 -> to
be tested after the core arms).

### Table 3, warm vs cold plants (09-08) — **the trees' within-site lead is not memorisation**

A plant in a surveyed network is *warm* if any of its edges survives the removal of local-network pairs from
training, *cold* otherwise (45 of 91 networks contain cold plants; cold connectance 0.077 vs warm 0.136).

| method | AUPR warm plants | AUPR cold plants |
|---|---:|---:|
| popularity | 0.194 | 0.205 |
| co-occurrence N | 0.169 | 0.102 |
| congeneric transfer | 0.223 | 0.260 |
| SVD + taxonomic | 0.180 | 0.244 |
| NECTAR-style (genus x overlap) | 0.191 | 0.242 |
| pair GBM (Pichler) | 0.205 | 0.209 |
| **boosted ranker / routed** | 0.223 | **0.268** |

Every taxonomy-carrying method scores *higher* on cold plants than on warm ones, at roughly half the connectance --
a 3.5x lift on cold plants for the booster against 1.6x on warm. The within-site lead of the trees and of
congeneric transfer therefore does not come from having seen the plant's other partners; it comes from
identity signal (genus-level co-visitation) that generalises to a plant with no edges at all. Co-occurrence N,
by contrast, collapses on cold plants (0.102 at 0.077 connectance). The embedding model's warm/cold values arrive
with its rerun; its fusion re-ranker (identity tokens) sits at 0.164 overall, i.e. the universe gain (+0.03 AUPR)
does not transfer to within-site prediction. Table 3 now carries: mean AUPR [CI], mean AUROC, precision@L, plant-
degree rho, NODF, warm/cold AUPR; pooled metrics and lift moved to the appendix.

### S2b pre-registration (09-08) — **can a neural system also win within sites?**

Dan's goal: one system, neural, leading both the universe metric and local-network completion. The within-site
deficit of every neural pair model (0.15-0.17 vs trees 0.22) has two candidate causes, and each gets an arm on
both tables (`scripts/queue_s2b.sh`, fusion re-ranker on the reference retriever, identity tokens):

| arm | what changes | hypothesis |
|---|---|---|
| M2.9 co-occurrence negatives | 16 extra negatives per plant drawn from pollinators with N > 0 | the models never trained on the within-site regime (uniform negatives are mostly non-co-occurring) |
| M2.10 genus-profile tokens | 32 tokens: pollinators recorded with the plant's genus (leave-one-out), with log count | the trees' strongest feature is an interaction-derived genus lookup that text cannot supply |
| M2.11 both | | |

Decision rule: a neural arm that reaches the trees' 0.22 (CI-overlapping) on local networks while holding >= 0.19
universe AUPR becomes the headline system (paper version A); otherwise M2.8 (re-ranker on the routed trees) is the
fallback (version B). M3.1 R-GCN is the third neural candidate (message passing over training edges via genus nodes).

### S1 phase B verdict (09-08) — **retriever_v1 frozen = reference embedding model**

3 seeds, paired plant bootstrap vs the reference (softmax 1.0, BCE 0.5, elementwise head):

| arm | AUPR | Delta AUPR | p | unseen-genus nR@10 Delta |
|---|---:|---:|---:|---:|
| reference | 0.159 | -- | -- | -- |
| + degree heads | 0.152 | -0.007 | 0.03 | +0.037 (p=0.03) |
| + degree heads + nnPU prior | 0.154 | -0.005 | 0.10 | +0.007 |
| + degree heads, concat+bilinear head | 0.138 | -0.021 | <0.001 | +0.018 |
| concat+bilinear head only | 0.147 | -0.012 | <0.001 | +0.004 |

Nothing clears the gate; the concat+bilinear head is significantly worse than the shared-space elementwise head
(the SBERT-style inductive bias earns its place), and the content-based degree heads buy nothing for pooled AUPR.
**retriever_v1 := M1.0_reference** (`configs`: blocks text+surface+pca+scale, softmax 1.0, BCE 0.5, elementwise
head, 25 epochs). The stage-2 re-ranker (S2) sits on this retriever.

### The neural within-site deficit is a warm-plant deficit (09-08)

Embedding model on local networks, warm/cold split (seed 42): **warm plants 0.165, cold plants 0.257** (cold
connectance 0.077). Against the trees' 0.223 / 0.268 and popularity's 0.194 / 0.205:

| | warm plants (own edges survive) | cold plants (no edges) |
|---|---:|---:|
| boosted ranker | 0.223 | 0.268 |
| embedding model | 0.165 | 0.257 |
| popularity | 0.194 | 0.205 |

On cold plants the neural model is within 0.01 of the trees and well above popularity; its whole deficit is on
plants whose other edges are in training. The trees' genus x pollinator affinity table includes a warm plant's own
surviving edges, so within a site it effectively knows the plant's other partners; the embedding model, built for
cold start, has no channel for that information. Consequences: (i) the earlier "neural models are weak within
sites" reading is too strong -- they are weak at *using warm information*; (ii) the fixes are the ones already
queued: genus-profile tokens in the re-ranker (M2.10; the lookup as a set), the R-GCN (message passing over the
plant's own edges), and a DropoutNet-style warm residual; (iii) M2.8 (mis-scaled base) is worse than the trees on
both tables (universe 0.114, local 0.214) and is superseded by M2.8b.

### S2 verdict (09-08, 3 seeds) — **the re-ranker is the gain; the field tokens are inert; the tree-retriever hybrid is weak**

`eval/report_ladder.py --ref M2.1_fusion_identity`, paired plant bootstrap (300), seed-averaged:

| arm | AUPR | AUROC | nR@10 | nR@50 | Delta AUPR vs identity-only | p |
|---|---:|---:|---:|---:|---:|---:|
| retriever_v1 alone (embedding model) | 0.159 | 0.954 | 0.282 | 0.416 | -0.032 | <0.001 |
| **M2.1 identity-only re-ranker** | **0.191** | 0.955 | 0.323 | 0.466 | -- | -- |
| M2.2 + joint field tokens (k=128) | 0.191 | 0.955 | 0.321 | 0.472 | -0.0005 | 0.49 |
| M2.7 space-marginal tokens | 0.194 | 0.955 | 0.324 | 0.472 | +0.003 | <0.001 |
| M2.7 time-marginal tokens | 0.194 | 0.955 | 0.321 | 0.468 | +0.003 | -- |
| M2.8 / M2.8b re-ranker on the routed trees (raw / affine base) | 0.114 / 0.114 | 0.887 / 0.881 | 0.289 / 0.297 | 0.443 / 0.453 | -- | -- |

- **Re-ranking the retriever's own top-500 with a pooled objective is worth +0.032 AUPR and +0.041 nR@10** over the
  retriever (p < 0.001), replicated on all three seeds (+0.032, +0.035, +0.030). It also lifts nR@50 by 0.05 and the
  unseen-genus stratum by 0.04. This is now the best universe-metric result of the project and the paper's system:
  stage 1 embedding-model retriever, stage 2 identity-token re-ranker.
- **Field tokens are inert inside the re-ranker.** Joint tokens -0.0005 (p = 0.49); the space and time *marginal*
  controls are +0.003 -- a difference smaller than any decision threshold, and in the wrong direction for the
  "joint beats marginal" mechanism. In the fusion model the per-cell field contributes nothing measurable on the
  universe metric; the marginalisation ladder remains an analysis result about the inputs, not the model's engine.
- **The tree-retriever hybrid does not work.** Re-ranking the routed trees' top-500 (recall@500 0.603) gives 0.114 AUPR
  -- above the trees alone (0.103) but far below the neural system -- and *lowers* the trees' nR@10 (0.336 -> 0.29).
  The learned affine base did not change this, so it is not a score-scale artefact. Version B (hybrid) is out unless
  its local-network run (pending) is exceptional.

**S2b / hybrid, local networks (09-08):** M2.8b (re-ranker on routed trees, affine base) 0.219 vs trees 0.222 -- no gain
on either table; the hybrid is closed. M2.9 (co-occurrence-conditioned negatives) 0.165 -- identical to the identity-only
re-ranker; the within-site deficit is not a negative-sampling artefact. Consistent with the warm-plant diagnosis: the
neural system lacks a channel for a plant's own surviving edges. M2.10 (genus-profile tokens) is the arm that tests
that directly; M3.1 R-GCN (corrected loss) is the second.

### M3.1 R-GCN, corrected loss (09-08, seed 42, provisional) — **the best cold-start ranker so far**

R-GCN over species / genus+family / cell x month nodes, two relation-typed layers, leave-own-edges-out, the frozen
retriever's objective (softmax 1.0, BCE 0.5, elementwise head):

| | AUPR | AUROC | nR@10 | nR@50 | unseen-genus nR@10 |
|---|---:|---:|---:|---:|---:|
| embedding model (seed 42) | 0.165 | 0.955 | 0.292 | 0.432 | 0.134 |
| routed trees | 0.103 | 0.883 | 0.336 | 0.447 | 0.256 |
| identity-only re-ranker (seed 42) | 0.197 | 0.956 | 0.329 | 0.481 | 0.176 |
| **R-GCN (seed 42)** | 0.154 | **0.963** | **0.369** | **0.528** | 0.235 |

The first run of this model (0.038) used the pooled-only loss that phase A showed collapses; with the retriever's
objective it is the strongest ranker of the project and within 0.01 of the embedding model on AUPR. Mechanism: the
genus nodes carry interaction-derived identity (what the trees' affinity table holds) and the cell x month nodes
carry the field, both reached by message passing; unseen-genus recall 0.235 says the family node backs off where the
genus is empty. Seeds 0/1 and the local-network run follow.

**Pre-registered next arm (M2.12):** fusion re-ranker (identity tokens) on the R-GCN as retriever. Hypothesis: the
+0.03 AUPR re-ranking gain transfers, giving a system that leads AUPR *and* recall; the local-network run of the
R-GCN decides whether message passing over a plant's own edges closes the warm-plant gap.

**M2.10 genus tokens, local networks: 0.165 -- no change.** Cause identified on inspection: the profile was built
leave-one-out at the *plant* level, so a warm plant's own surviving edges were excluded from its own tokens -- the very
information the trees use. Pre-registered fix **M2.10b**: profile includes the plant's own edges; only the scored
candidate is masked (pair-level leave-one-out), which is what the trees' affinity table effectively does. Local-network
run first (the readout), then the universe seeds.

**M2.11 (genus tokens + co-occurrence negatives), universe:** 0.196 / 0.144 / 0.123 across seeds 42 / 0 / 1 -- unstable,
while each component alone sits at ~0.19 on every seed. Likely conflict: co-occurring negatives include pollinators that
appear in the plant's genus profile, so the same token is presented as "visits this genus" and labelled negative for the
plant. Not pursued further; the pair-masked genus tokens (M2.10b) are the clean version.

**M2.10b (pair-masked genus tokens), local networks: 0.165 -- no change.** Second construction flaw: masking the candidate's
token entirely during training removes congener evidence too, so the model can never learn "candidate appears in the genus
profile -> partner", the trees' signal. **M2.10c** subtracts only the plant's own contribution to the candidate's count
(mask only if nothing from congeners remains). This matches the information the trees' affinity table has for a held-out
pair. Local-network run first; M2.10b's remaining universe seeds were stopped as uninformative.

### M3.1 R-GCN, complete (09-08) — **the neural model that approaches the trees within sites while beating them everywhere else**

Universe (3 seeds): AUPR 0.150, AUROC 0.962, nR@10 0.368, nR@50 0.532, unseen-genus nR@10 0.244.
Local networks (seed 42): **mean AUPR 0.204 [0.190, 0.219]**, pooled AUROC 0.643 (best), precision@L 0.231, NODF 48 (obs 36),
warm-plant AUPR **0.206**, cold-plant 0.238.

| | universe AUPR | nR@10 | local mean AUPR | local warm / cold | NODF pred |
|---|---:|---:|---:|---:|---:|
| boosted / routed trees | 0.103 | 0.336 | 0.222 | 0.223 / 0.268 | 62 |
| embedding model | 0.159 | 0.282 | 0.165 | 0.165 / 0.257 | 43 |
| identity re-ranker on embedding model | 0.191 | 0.323 | 0.164 | -- | 46 |
| **R-GCN** | 0.150 | **0.368** | **0.204** | **0.206** / 0.238 | 48 |

Message passing over the plant's own surviving edges (through genus and cell x month nodes) is the warm-information
channel the pair models lacked: warm-plant AUPR rises from 0.165 to 0.206 (trees 0.223), and the local mean lands
inside the trees' CI. It is the best ranker on the universe by a clear margin and third on universe AUPR. **Headline
candidate (for Dan): the R-GCN, with the identity re-ranker on top (M2.12, running) as the full system** -- if the
re-ranker's +0.03 AUPR transfers, the system leads or ties every column except within-site precision@L.
Remaining gap: precision@L 0.231 vs 0.252 and cold-plant 0.238 vs 0.268 within sites.

**M2.10c (own-count subtraction), local networks: 0.165.** Three masking schemes (plant-LOO, pair mask, own-count
subtraction) all leave the re-ranker at 0.165 within sites: the token-set re-ranker does not convert a genus profile
into within-site gains, whereas the R-GCN's message passing does (0.204). The genus-token route is closed; the neural
warm-information channel is the graph. Remaining M2.10c universe seeds stopped as uninformative.

### M2.12 — re-ranker on the R-GCN (09-08, seed 42, provisional) — **best on every universe column**

R-GCN retriever (recall@500 on val plants 0.745, vs 0.684 for the embedding model) + identity-token re-ranker:

| | AUPR | AUPR 1:3 / 1:1 | AUROC | nR@10 | nR@50 | unseen-genus nR@10 |
|---|---:|---:|---:|---:|---:|---:|
| R-GCN alone (seed 42) | 0.154 | 0.908 / 0.962 | 0.963 | 0.369 | 0.528 | 0.235 |
| re-ranker on embedding model (seed 42) | 0.197 | 0.907 / 0.960 | 0.956 | 0.329 | 0.481 | 0.176 |
| **re-ranker on R-GCN (seed 42)** | **0.191** | **0.926 / 0.970** | **0.970** | **0.384** | **0.552** | **0.253** |

The re-ranking gain (+0.037 AUPR here) transfers to the R-GCN retriever, and the R-GCN's ranking advantage is kept.
Pending: seeds 0/1 and the local-network run (the R-GCN alone is at 0.204 there).

**R-GCN local networks, 3 seeds:** mean AUPR 0.204 / 0.205 / 0.203; warm-plant 0.206 / 0.207 / 0.206; cold 0.238 / 0.243 /
0.241; pooled AUROC 0.643 on all; precision@L 0.231. Stable to the third decimal. **M2.12 universe, seed 0:** AUPR 0.186,
AUROC 0.970, nR@10 0.383 (seed 42: 0.191 / 0.970 / 0.384) -- replicates.

### THE SYSTEM (09-08): R-GCN retriever + identity-token re-ranker (M2.12) — **frozen; headline candidate for Dan**

| | universe AUPR (3 seeds) | AUPR 1:3 / 1:1 | AUROC | nR@10 | nR@50 | unseen-genus | local mean AUPR | local warm / cold | precision@L |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| trees (boosted / routed) | 0.103 | 0.81 / 0.91 | 0.883 | 0.336 | 0.447 | 0.256 | 0.222 [0.205, 0.239] | 0.223 / 0.268 | 0.252 |
| embedding retriever + re-ranker | 0.191 | 0.905 / 0.959 | 0.955 | 0.323 | 0.466 | 0.159 | 0.164 | -- | 0.159 |
| **R-GCN retriever + re-ranker** | **0.188** (0.191 / 0.186 / 0.186) | **0.926 / 0.970** | **0.970** | **0.382** | **0.550** | **0.260** | **0.205** [0.191, 0.221] (seed 0; seeds 42/1 running) | R-GCN alone: 0.206 / 0.241 | 0.233 |

- Leads every universe column: AUPR within noise of the embedding re-ranker (0.188 vs 0.191) and far above the trees;
  AUROC, 1:3, 1:1, nR@10, nR@50 and unseen-genus recall all best by a clear margin.
- Within sites it ties the trees (0.205 vs 0.222; each inside the other's CI) and beats every other neural model by 0.04;
  precision@L 0.233 vs 0.252 and cold-plant AUPR ~0.24 vs 0.27 are the residual gaps.
- Frozen config: retriever = `rgcn` with `{"field_dir": field_v2, "softmax_weight": 1.0, "bce_weight": 0.5,
  "head_type": "elementwise", "use_degree_heads": false}` (2 relation-typed layers, d=128, genus+family and cell x month
  nodes, leave-own-edges-out 0.3); re-ranker = identity tokens, 3 layers d=192, top-500, pooled BCE, 8 epochs.
- Ablations for the paper: R-GCN alone (0.150 / 0.368 / local 0.204); embedding retriever alone (0.159 / 0.282 / 0.166);
  re-ranker on the embedding retriever (0.191 / 0.323 / 0.164); re-ranker + field tokens (inert). Boosted / routed trees
  move to a "hand-engineered features (our earlier system)" baseline group. Tables regenerated.
- Pre-registered rule asked for ~0.22 locally; 0.205 is inside the trees' interval, not above it. Dan decides whether
  "ties the trees within sites, leads everywhere else" is the headline, or whether to push the within-site gap first.

**Pre-registered probe (09-08): R-GCN with 3 message-passing layers** (M3.2), one universe seed + local networks s42.
Hypothesis: a third hop reaches plant -> genus -> sibling plant -> pollinator paths the two-layer model only sees through
the pooled genus node, and may move the within-site residuals (precision@L 0.232 vs trees 0.252; cold plants ~0.24 vs
0.27). Adopt only if local mean AUPR rises by >= 0.01 without a universe AUPR loss > 0.01. M2.12 local seed 42: 0.203.

### Paired contrasts for the system (09-08, 3 seeds, 300 plant resamples, cold-plant validation)

vs **R-GCN retriever + identity re-ranker** (universe AUPR 0.188, AUROC 0.970, nR@10 0.383, nR@50 0.550, unseen-genus 0.260):

| comparison | Delta AUPR | p | Delta nR@10 | p | Delta unseen-genus nR@10 | p |
|---|---:|---:|---:|---:|---:|---:|
| R-GCN alone | -0.038 [-0.044, -0.033] | <0.001 | -0.015 | <0.001 | -0.018 | 0.13 |
| re-ranker on the embedding model | +0.003 [-0.010, +0.018] | 0.69 | -0.059 | <0.001 | -0.101 | <0.001 |
| embedding model alone | -0.029 | <0.001 | -0.101 | <0.001 | -0.140 | <0.001 |
| routed trees | -0.089 | <0.001 | -0.049 | <0.001 | +0.003 | 0.93 |
| boosted ranker | -0.090 | <0.001 | -0.064 | <0.001 | -0.131 | <0.001 |

The re-ranking stage is worth +0.038 AUPR on the R-GCN (p<0.001), the same size as on the embedding model. The two
re-ranked systems tie on AUPR (p=0.69) and the R-GCN-based one wins ranking by 0.059 nR@10 and the unseen-genus
stratum by 0.10 (p<0.001). Against the trees: +0.089 AUPR, +0.049 nR@10, tie on unseen genera. Local networks (2 seeds
so far): 0.205 / 0.203 vs trees 0.222 [0.205, 0.239].

**System on local networks, 3 seeds:** 0.203 / 0.205 / 0.203 (mean 0.204), pooled AUROC 0.642-0.643, precision@L 0.230-0.233,
NODF 50-53 (obs 36). Identical to the R-GCN alone (0.204): the re-ranker neither helps nor hurts within sites. Trees:
0.222 [0.205, 0.239]. GPU 1 idle by design (no new arms beyond the pre-registered M3.2 probe without Dan).

**M3.2 probe result (09-08, seed 42): 3-layer R-GCN does not help.** Universe AUPR 0.132 (2-layer: 0.150), nR@10 0.371
(0.369); local mean AUPR 0.205 (0.204), warm 0.207 / cold 0.259, precision@L 0.234. Fails the adoption rule (local gain
+0.001 < 0.01 with a universe loss of 0.018). The two-layer R-GCN stays as the retriever. No further arms are queued
without Dan; both GPUs idle.

## Refinement stage and split battery (09-08, evening) -- Dan: "use the system as the basis and refine; add warm start and the other regimes"

Pre-registered rule for every refinement: adopt only if local mean AUPR rises by >= 0.01 with no universe AUPR loss > 0.01
and every universe lead retained. Arms (R-GCN retriever, frozen objective): R1 warm residual (`warm_residual`, DropoutNet-style
per-species vector, zero for cold species by construction), R2 direct genus <-> partner relations (`genus_edges`, log1p training
counts, leave-own-edges-out), R1+R2, R3 symmetric leave-own-edges-out (`anchor_side="both"`), and an attention-aggregation
control (`aggregation="attention"`, per-relation GAT-style, SimpleHGN-like) requested by Dan to answer "why not a transformer".
Split battery (cold_poll, cold_both, warm; `scripts/queue_battery.sh`) for the system, its retriever and all comparison models.

**R2 genus edges, seed 42:** universe AUPR 0.151 (M3.1 s42 0.150), nR@10 0.366 (0.368), unseen-genus 0.225; local mean AUPR
0.190 [0.175, 0.205] (M3.1 0.204), warm 0.192 (0.206), cold 0.252 (0.241), precision@L 0.213 (0.233), NODF 47. The direct
genus lookup lifts cold plants within sites a little and costs warm plants more: fails the rule on seed 42.

**Battery, cold_poll seed 42 (8,123 plants x 1,312 held-out pollinators, 5,829 positives, prevalence 0.00055):** the frozen
system collapses. R-GCN alone AUPR 0.005 (AUROC 0.71), system 0.017 (0.76), while the co-occurrence models lead: ANTHEIA v1
scalar 0.037 (AUROC 0.84), ANTHEIA v1 spatial 0.033, pair GBM 0.031 (0.85), Wide & Deep 0.029, phenology x abundance 0.027,
abundance 0.026, N 0.021; identity-based baselines are at chance (congeneric = popularity = 0.002, SVD 0.005, boosted trees
0.003, pair MLP 0.002). Diagnosis: leave-own-edges-out was applied to plants only, so the model never rehearsed a pollinator
without edges; its pollinator side is pure identity. Field-based baselines win where identity is empty -- consistent with the
field-independence measurement (84% of pollinator field variance unexplained by taxonomy/text). R3 (symmetric anchors) is the
principled fix and is queued as `M3.7_rgcn_sym` (cold_plant + local + cold_poll, 3 seeds; smoke after 1 epoch on cold_poll:
AUPR 0.0375, AUROC 0.87 -- a smoke, not a result). Adopting R3 into the frozen system is Dan's call (it changes the retriever).

**R1 warm residual, seed 42:** universe AUPR 0.170 (M3.1 s42 0.150), AUROC 0.966, nR@10 0.374 (0.368), nR@50 0.542, unseen-genus
0.223; local mean AUPR 0.179 [0.166, 0.193] (0.204), warm 0.182 (0.206), cold 0.237 (0.241), precision@L 0.191 (0.233), plant
degree rho 0.03 (0.13). The universe gain arrives through the pollinator side (every pollinator is warm on this split and now
carries a memory vector); within sites the same memory over-weights universe-level generalists and the model loses 0.025.
Fails the adoption rule (local must rise). Second arm in a row where a universe gain and a within-site loss travel together.

**R3 symmetric anchors, cold-plant seed 42:** universe AUPR 0.169, AUROC 0.966, nR@10 0.371, nR@50 0.543, unseen-genus 0.221 --
also +0.019 over the frozen retriever on cold plants. Local networks and cold_poll pending.

**R3 symmetric anchors, local networks seed 42:** mean AUPR 0.215 [0.199, 0.233] (M3.1 0.204; trees 0.222 [0.205, 0.239]), warm
0.217 (0.206), cold 0.227 (0.241), precision@L 0.232 (0.233), pooled AUROC 0.631, NODF 51.5 (obs 36). With the universe +0.019,
R3 is the first arm that passes the adoption rule on seed 42 on both tables (local +0.011, universe +0.019). Rehearsing
pollinators without edges regularises the pollinator side, which is what the within-site task ranks over. Seeds 0/1 and the
cold_poll seeds are queued (`scripts/queue_refine2.sh`); if they hold, the system (re-ranker on the R3 retriever) is queued
behind a go-file (`logs/r3_system.go`, `scripts/queue_r3_system.sh`) -- adopting it as the frozen retriever is Dan's call.

**Site-time term at inference (R-site), on the R3 local bundle:** beta cross-fitted two-fold (0.25 selected in both folds);
mean AUPR 0.215 -> 0.218 (+0.002 [-0.000, +0.005], p = 0.09), precision@L 0.232 -> 0.237, AUROC unchanged. Within a site the
field's phenology at that cell adds essentially nothing over the graph model. Negative; closed (inference-only, no seeds needed).

**R1+R2 (warm residual + genus edges), seed 42:** universe AUPR 0.164, nR@10 0.373, nR@50 0.552, unseen-genus 0.240; local mean
AUPR 0.185 [0.171, 0.199] (0.204), warm 0.188, cold 0.250, precision@L 0.209. Fails the rule, as each part did alone. The
refinement queue on GPU 1 now runs seeds 0/1 of R2, R1 and R1+R2 for the record, then the attention control.

**R3 symmetric anchors, cold_poll seed 42:** AUPR 0.0255 (frozen R-GCN 0.0047, frozen system 0.0172), AUROC 0.835 (0.708),
nR@10 0.224 (0.119), nR@50 0.439 (0.265). The collapse is repaired (5x) but the retriever alone still trails the co-occurrence
models on this split (ANTHEIA v1 scalar 0.0367 / AUROC 0.837, pair GBM 0.0312 / 0.852). The re-ranker added +0.012 on the
frozen retriever here, so the system on R3 is expected near 0.035; the explicit pair co-presence statistic (R4) is the arm
aimed at the remainder and should also be run on cold_poll.

**Battery, cold_both seed 42 (1,085 held-out plants x 1,312 held-out pollinators, 947 positives, prevalence 0.00067):**

| model | AUPR | AUROC | nR@10 |
|---|---:|---:|---:|
| ANTHEIA v1 scalar (PCA-15 + N + Delta) | **0.045** | 0.809 | **0.316** |
| ANTHEIA v1 spatial (PCA-15 + N) | 0.042 | 0.798 | 0.313 |
| frozen system (R-GCN + re-ranker) | 0.031 | 0.762 | 0.203 |
| phenology x abundance | 0.029 | 0.805 | 0.243 |
| abundance neutral | 0.028 | 0.798 | 0.190 |
| pair GBM | 0.028 | **0.847** | 0.200 |
| Wide & Deep | 0.026 | 0.817 | 0.165 |
| co-occurrence N | 0.020 | 0.691 | 0.218 |
| NECTAR-style (gated = ungated) | 0.016 | 0.822 | 0.039 |
| embedding retriever | 0.011 | 0.726 | 0.068 |
| two-tower / DCN-V2 / frozen R-GCN alone | 0.009 / 0.008 / 0.006 | 0.54 / 0.67 / 0.72 | |
| pair MLP / SVD+tax / trees / routed / congeneric / popularity | 0.003-0.004 | 0.50-0.57 | |

Where both sides are unseen, every identity-driven model is at chance and the co-occurrence PCA models lead; the frozen
system is third on AUPR only because the re-ranker rescues its retriever (0.006 -> 0.031). Same diagnosis as cold_poll; R3
on this split is queued in refine2.

**R3 symmetric anchors, cold-plant 3 seeds:** AUPR 0.169 / 0.167 / 0.175 (mean 0.170; frozen retriever 0.150), nR@10 0.371-0.378,
unseen-genus 0.221-0.253; cold_poll seeds 42/0: 0.0255 / 0.0263. Universe criterion met on three seeds, local criterion met on
seed 42 (+0.011). Go-file created: the system on the R3 retriever (M4.0_system_sym) starts on GPU 1 -- cold_plant 3 seeds,
local networks, battery seed 42. Local seeds 0/1 for the R3 retriever still to run.

**Warm-split evaluation bug (09-08 21:00), fixed before any warm number is used.** The first warm run (frozen R-GCN, seed 42)
returned AUPR 0.015 with AUROC 0.967: the evaluated plants' *training* partners stayed in the candidate set as label-0
negatives, so a model that remembers its training edges is punished for it. Fix = filtered ranking (Bordes et al. 2013): an
evaluated plant's known pairs (training edges and the other part's held-out pairs) are removed from every pooled metric and
pushed below all candidates in the per-query rankings (`evaluate_scores(..., exclude=)`, `exclude.npy` saved with the bundle,
`n_excluded` in metrics). Applies to the warm split only; cold splits have no such pairs by construction. Warm bundles
written before the fix are deleted and re-run by `scripts/queue_warm_fix.sh` after the battery finishes.
