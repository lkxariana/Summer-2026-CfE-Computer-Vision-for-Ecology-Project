# 08 — Methods Literature (Redesign References)

Compiled 2026-08-31 to inform the post-August redesign (GloBI orientation fix, evaluation overhaul, richer models). Complements the reading path in `docs/07-reading-list.md`: 07 is the paper-positioning path; this file is the methods toolbox. All citations were verified against the literature; items that could not be verified are marked.

**Corrections to existing docs:** (1) The roadmap paper cited throughout as "Poisot et al. 2021" is **Strydom et al. 2021** (Poisot is senior author). (2) "Coelho et al. 2024" could not be located; the ~20%-of-co-occurrences finding, and the DOI given for it in `docs/07`, belong to **Galiana et al. 2024** (see §3 below); the "Netflix problem" framing originates in **Desjardins-Proulx et al. 2017**, not the roadmap.

---

## 1. Stronger problem formulations for ecological interaction prediction

**Strydom, Catchen, Banville, …, Gravel, Pollock & Poisot 2021.** *A roadmap towards predicting species interaction networks (across space and time).* Phil. Trans. R. Soc. B 376:20210063. https://doi.org/10.1098/rstb.2021.0063
Interaction prediction must be explicitly spatial/temporal and must separate the *metaweb* (which species can interact) from *local realizations* (which do, here and now). ANTHEIA currently conflates the two by folding spatial co-occurrence into the same feature vector that predicts the interaction itself.

**Desjardins-Proulx, Laigle, Poisot & Gravel 2017.** *Ecological interactions and the Netflix problem.* PeerJ 5:e3644. https://doi.org/10.7717/peerj.3644
Casts interaction inference as recommendation over positive-only data: rank candidate partners rather than binary-classify pairs. Reframing ANTHEIA as ranking (recall@k over candidate pollinators per plant) reports something ecologically actionable, unlike pooled ROC-AUC.

**Strydom, Dalla Riva & Poisot 2022.** *SVD entropy reveals the high complexity of ecological networks / phylogenetic transfer of low-rank network representations.* Methods Ecol. Evol. 13:2308–2319. https://doi.org/10.1111/2041-210X.13835
Truncated SVD of the adjacency matrix yields latent traits (an RDPG); phylogeny transfers those traits to species without interaction data. The direct competitor to Vf/Vp: latent factors learned from the *interaction* matrix, not the occupancy matrix.

**Strydom, Bouskila, Banville, …, Dalla Riva & Poisot 2023.** *Graph embedding and transfer learning can help predict potential species interaction networks despite data limitations.* Methods Ecol. Evol. 14:2917–2930. https://doi.org/10.1111/2041-210X.14228
Projects a European mammal metaweb into latent space and infers Canadian species' latent traits by phylogenetic proximity — the template for cross-region transfer.

**Pichler, Boreux, Klein, Schleuning & Hartig 2020.** *Machine learning algorithms to infer trait-matching and predict species interactions in ecological networks.* Methods Ecol. Evol. 11:281–293. https://doi.org/10.1111/2041-210X.13329
Boosted trees / RF / DNNs on plant × pollinator trait pairs recover interaction-mediating trait combinations that GLMs miss. ANTHEIA has no trait channel, so it cannot express the mechanisms (corolla depth × proboscis length) that determine feasibility.

**Bartomeus, Gravel, Tylianakis, Aizen, Dickie & Bernard-Verdier 2016.** *A common framework for identifying linkage rules across different types of interactions.* Funct. Ecol. 30:1894–1903. https://doi.org/10.1111/1365-2435.12666
Decomposes any observed link into trait-based *feasibility* (forbidden links) × neutral/abundance *encounter* probability. ANTHEIA's N and Δ are pure encounter terms; the model is currently Bartomeus's neutral null with no feasibility term.

**Morales-Castilla, Matias, Gravel & Araújo 2015.** *Inferring biotic interactions from proxies.* Trends Ecol. Evol. 30:347–356. https://doi.org/10.1016/j.tree.2015.03.014
Impute metaweb structure from functional groups, phylogeny, and geography — in that order. Geography should be applied *last* as a filter on a trait/phylogeny-derived candidate set, not first as the primary signal.

**Stock, Piot, Vanbesien, Meys, Smagghe & De Baets 2021.** *Pairwise learning for predicting pollination interactions based on traits and phylogeny.* Ecol. Modelling 451:109508. https://doi.org/10.1016/j.ecolmodel.2021.109508
Kronecker-kernel pairwise learning over plant and pollinator feature spaces, evaluated under explicit unseen-plant / unseen-pollinator settings. The closest published analogue to ANTHEIA's task and its natural published baseline.

**Anakok, Barbillon, Fontaine & Thébault 2024.** *Disentangling the structure of ecological bipartite networks from observation processes.* arXiv:2403.02011 (preprint). https://arxiv.org/abs/2403.02011
Bipartite graph VAE with an HSIC penalty forcing embeddings independent of sampling-effort covariates. Directly targets ANTHEIA's largest confound: observation effort as the strongest feature.

**Kampe, DeSisto & Dunson 2025.** *COIL+: latent-factor link prediction under extreme taxonomic bias.* arXiv:2506.23370 (preprint). https://arxiv.org/abs/2506.23370
Pools multiple biased datasets with traits and phylogeny in a latent-factor model; recovers unobserved links concentrated in understudied taxa. The correct response to a biased label set is a bias-aware generative model, not uniform negative sampling.

**Peralta, CaraDonna, Rakosy, …, Castillo & Vázquez 2024.** *Predicting plant–pollinator interactions: concepts, methods, and challenges.* Trends Ecol. Evol. 39:494–505. https://doi.org/10.1016/j.tree.2023.12.005
Consensus statement of concepts, methods and pitfalls for plant–pollinator interaction prediction — the yardstick for whether the formulation is defensible to the ecology audience.

---

## 2. Learned species-range and location representations (alternatives to PCA of occupancy grids)

**SINR — Cole, Van Horn, Lange, Shepard, Leary, Perona, Loarie & Mac Aodha 2023.** *Spatial implicit neural representations for global-scale species mapping.* ICML 2023 (PMLR v202). https://arxiv.org/abs/2306.02564
A coordinate-input network gives a continuous shared location embedding with per-species output heads (~47k species). Each species' learned head vector replaces its 15D PCA row; the continuous surface removes the 3,162-bin discretization.

**LE-SINR — Hamilton, Lange, Cole, Shepard, Heinrich, Van Horn & Maji 2024.** *Combining observational data and language for species range estimation.* NeurIPS 2024. https://arxiv.org/abs/2410.10931 — with follow-up **Lange et al. 2025**, *Feedforward few-shot species range estimation*, ICML 2025. https://arxiv.org/abs/2502.14977
Zero-/few-shot range estimation — precisely the regime where rare species have near-empty occupancy rows that PCA cannot represent.

**SatCLIP — Klemmer, Rolf, Robinson, Mackey & Rußwurm 2025.** AAAI 2025. https://arxiv.org/abs/2311.17179
Contrastive Sentinel-2/coordinate alignment yields general-purpose dense location embeddings; a species becomes a pooled embedding of *where* it occurs rather than a binary bin-indicator vector.

**GeoCLIP — Vivanco Cepeda, Nayak & Shah 2023.** NeurIPS 2023. https://arxiv.org/abs/2309.16020
CLIP-style image-to-GPS alignment with a random-Fourier-feature location encoder — smooth multi-scale coordinate features without hard-binning artifacts.

**TaxaBind — Sastry, Khanal, Dhakal, Ahmad & Jacobs 2025.** WACV 2025 (Oral); MVRL, WashU. https://arxiv.org/abs/2411.00683
Binds six modalities (species image, location, satellite image, text, audio, environmental features) with species images as the binding modality. In-house replacement for PCA-of-occupancy grounding each species in appearance, geography and environment simultaneously.

**CSP — Mai, Lao, He, Song & Ermon 2023.** *Self-supervised contrastive spatial pre-training for geospatial-visual representations.* ICML 2023. https://arxiv.org/abs/2305.01118
Dual-encoder contrastive pretraining on geo-tagged images; transferable location representations (10–34% relative gain on iNat2018).

**MOSAIKS — Rolf, Proctor, Carleton, Bolliger, Shankar, Ishihara, Recht & Hsiang 2021.** Nature Communications 12:4392. https://doi.org/10.1038/s41467-021-24638-z
Random convolutional features over satellite imagery: a fixed, task-agnostic per-location vector usable with linear models. A drop-in representation derived from *environment* rather than the occupancy matrix — avoiding the circularity of embedding co-occurrence to predict co-occurrence.

---

## 3. Evaluation protocols for link prediction with few positives

**Pahikkala, Airola, Pietilä, Shakyawar, Szwajda, Tang & Aittokallio 2015.** *Toward more realistic drug–target interaction predictions.* Brief. Bioinform. 16:325–337. https://doi.org/10.1093/bib/bbu010
Canonical demonstration that random pair splits collapse under unseen rows/columns; defines the four settings (both seen / new row / new column / both new). A random pair split is the easiest setting: the model can memorize per-species base rates through Vf/Vp.

**Biton, Puzis & Pilosof 2025.** *Inductive link prediction facilitates the discovery of missing links and enables cross-community inference in ecological networks.* Nature Ecol. Evol. 9:1214–1223. https://doi.org/10.1038/s41559-025-02715-6
Inductive (feature-based, species-generalizing) link prediction transfers across communities where transductive methods cannot — the ecological precedent for leave-species-out evaluation.

**Roberts, Bahn, Ciuti, …, Hartig & Dormann 2017.** *Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure.* Ecography 40:913–929. https://doi.org/10.1111/ecog.02881 — tooling: **Valavi et al. 2019**, blockCV, Methods Ecol. Evol. 10:225–232. https://doi.org/10.1111/2041-210X.13107
Random CV on spatially structured data seriously underestimates predictive error; ANTHEIA's features are built from spatial occupancy, so its error estimate is optimistic by construction.

**Aiyappa, Wang, Kim, Seckin, Yoon, Ahn & Kojaku 2024.** *Implicit degree bias in the link prediction task.* arXiv:2405.14985 (preprint). https://arxiv.org/abs/2405.14985
Standard link-prediction evaluation is biased toward high-degree nodes; a degree-only null can score near-optimally. Every results table must include a degree-only baseline or its AUC numbers are uninterpretable.

**Phillips, Dudík, Elith, Graham, Lehmann, Leathwick & Ferrier 2009.** *Sample selection bias and presence-only distribution models (target-group background).* Ecol. Appl. 19:181–197. https://doi.org/10.1890/07-2153.1
Backgrounds drawn uniformly rather than with the same sampling bias as presences produce spuriously good models; the fix is bias-matched background. Uniform 3:1 negatives let the classifier separate classes on sampling effort alone.

**Elkan & Noto 2008.** *Learning classifiers from only positive and unlabeled data.* KDD '08, 213–220. https://cseweb.ucsd.edu/~elkan/posonly.pdf
Under SCAR, positive-vs-unlabeled training recovers the true posterior up to a constant. Undocumented pairs are unlabeled, not negative — PU is the correct framing. Ecological precedent: **Ward, Hastie, Barry, Elith & Leathwick 2009**, Biometrics 65:554–563. https://doi.org/10.1111/j.1541-0420.2008.01116.x (prevalence not identifiable without external constraint; the 3:1 ratio silently fixes implied prevalence and calibration).

**Poisot 2023.** *Guidelines for the prediction of species interactions through binary classification.* Methods Ecol. Evol. 14:1333–1345. https://doi.org/10.1111/2041-210X.14071
Measure choice, thresholding, and training-set assembly for exactly this extreme-imbalance setting — the most citable single objection to ROC-AUC as headline metric.

**Saito & Rehmsmeier 2015.** PLoS ONE 10:e0118432. https://doi.org/10.1371/journal.pone.0118432 — and **Hanczar et al. 2010**, Bioinformatics 26:822–830. https://academic.oup.com/bioinformatics/article/26/6/822/244957
PR curves are more informative than ROC under imbalance; small-sample AUC estimates correlate weakly with true AUC. Report PR-AUC with bootstrap CIs; seeds over the same split quantify only split-independent noise.

**Becker, Albery, Sjodin, Poisot, Bergner et al. 2022.** *Optimising predictive models to prioritise viral discovery in zoonotic reservoirs.* Lancet Microbe 3:e625–e637. https://doi.org/10.1016/S2666-5247(21)00245-7
Prospective out-of-sample validation of a host–virus link-prediction ensemble; in-sample performance was a poor guide to real discovery. The model for validating ANTHEIA: held-out newly-documented GloBI records, not a random split.

**Galiana, Arnoldi, Mestre et al. 2024.** Nat. Ecol. Evol. 8:209–217. https://doi.org/10.1038/s41559-023-02254-y — and **Menares et al. 2024**, Ecol. Evol. 14:e70498. https://doi.org/10.1002/ece3.70498
~20% of co-occurrences correspond to real interactions; degree distributions differ between co-occurrence and interaction networks; the best co-occurrence method recovered only 25.2% of documented plant–butterfly mutualisms. Together these bound how much of a co-occurrence model's signal can be interaction rather than shared range. (This is the finding `docs/07` attributes to "Coelho et al. 2024".)

---

## 4. Interaction data sources beyond the current GloBI pull (North America)

**GloBI itself** — Poelen, Simons & Mungall 2014, Ecological Informatics 24:148–159. https://doi.org/10.1016/j.ecoinf.2014.08.005
GloBI ingests iNaturalist, USGS/BISON and 200+ sources. The curated bee subset of **Noori et al. 2026, Scientific Data** (https://doi.org/10.1038/s41597-026-06970-5) holds **981,982 unique bee–plant records across 5,537 bee and 12,699 plant taxa, 85.3% North American**. The 139-positive count reflects extraction/orientation/name-matching failures, not a data shortage — re-extracting GloBI correctly is the highest-value fix before adding any external source.

**Global Bee Interaction Data** — Seltmann, Poelen & the GloBI Community, v7.0, Zenodo 2025. https://zenodo.org/records/17957582
3,030,355 records (1,868,619 deduplicated); iNaturalist contributes 287,441, USGS BISON 258,683. Adds CONUS positives but overlaps a correct GloBI pull.

**Mangal** — Poisot et al. 2016, Ecography 39:384–390. https://doi.org/10.1111/ecog.00976
>1,300 networks, ~120,000 interactions, ~7,000 taxa (figures from Poisot et al. 2021, J. Biogeogr. 48:1552–1563, https://doi.org/10.1111/jbi.14127). Includes North American pollination networks; CONUS is a modest slice.

**Web of Life / IWDB** — https://www.web-of-life.es/ (no canonical data paper). 60 top-level pollination networks; IWDB appears unmaintained. Marginal: tens of new CONUS pairs from classic webs.

**DoPI** — Balfour, Castellanos, Goulson, Philippides & Johnson 2022, Ecology 103(11):e3801. https://doi.org/10.1002/ecy.3801
101,539 UK records, >17,000 unique pairs. No direct CONUS positives; transfer limited to shared Holarctic and introduced taxa.

**EuPPollNet** — Lanuza, Knight, …, Bartomeus (110 authors) 2025, Global Ecology and Biogeography 34(2). https://doi.org/10.1111/geb.70000
1,162,109 interactions, 1,864 networks, 23 European countries, 1,411 plants × 2,223 pollinators (2004–2021). No CONUS positives — pretraining/transfer data with strong domain shift.

**USGS BIML** — Droege et al. 2026, USGS sampling-event dataset v1.30. https://ipt.gbif.us/resource?r=usgs-pwrc-biml
329,917 North American bee/wasp records, many with free-text host-plant notes; CONUS-relevant but requires string parsing — no clean pre-extracted pair table (size of extractable pairs unverified).

**iNaturalist interaction fields** — Gazdic & Groom 2019, Biodiversity Information Science and Standards 3:e37303. https://doi.org/10.3897/biss.3.37303
Mining "Interaction→Visited flower of" observation fields; CONUS-weighted, but these records already flow into GloBI.
