# Comparison methods

Candidate benchmark methods for cold-start bipartite interaction prediction. Our task is
**Setting B** in the pairwise-prediction taxonomy (Stock et al. 2018): a left node (plant) held out
entirely, all right nodes (pollinators) seen. Every entry states whether it can score a plant with
zero training edges — this disqualifies most of the standard link-prediction toolkit and must be
reported rather than quietly avoided.



---

## A. Nulls and heuristics

**Popularity / right-degree null** — score each pollinator by its training degree, independent of the
plant. Cold-start: yes. **Mandatory.** Aiyappa, Wang, Kim, Seckin, Yoon, Ahn & Kojaku, "Implicit
degree bias in the link prediction task," ICML 2025, PMLR v267:874–908 ([arXiv:2405.14985](https://arxiv.org/abs/2405.14985))
show that under uniform negative sampling a degree-only null is near-optimal, and propose a
degree-corrected protocol. **This is our earlier N-only-scores-0.99 observation, already published** —
we cite it as the known failure mode we are avoiding, not as a finding.

**Neighbourhood heuristics** (common neighbours, Adamic–Adar, resource allocation; Liben-Nowell &
Kleinberg 2007, *JASIST* 58:1019–1031) require the 2-hop plant–plant projection and are **not
cold-start capable** — a held-out plant's projection row is empty.

**Feature-similarity k-NN** — find the training plants most similar in occupancy, phenology and
taxonomy; transfer their pollinators. Cold-start: yes. The non-learned cold-start baseline, and
frequently strong.

**Negative-sampling protocol.** Li et al., "Evaluating GNNs for Link Prediction: Current Pitfalls and
New Benchmarking," NeurIPS 2023 D&B ([arXiv:2306.10453](https://arxiv.org/abs/2306.10453)) introduce
**HeaRT**, per-positive hard negatives. Results should be reported under both standard and
degree-corrected negatives.

## B. Matrix factorisation — transductive, reported as a ceiling

All learn a free embedding per plant, so a held-out plant has no vector and **cannot be scored**.
Included as a greyed row establishing the warm-start ceiling.

- **BPR-MF** — Rendle et al., UAI 2009 ([arXiv:1205.2618](https://arxiv.org/abs/1205.2618)). Pairwise
  logistic loss over (plant, positive, sampled negative).
- **iALS / WRMF** — Hu, Koren & Volinsky, ICDM 2008, DOI 10.1109/ICDM.2008.22. Confidence-weighted
  squared loss over the full matrix, alternating least squares. Library: `implicit`.
- **eALS** — He et al., SIGIR 2016 ([arXiv:1708.05024](https://arxiv.org/abs/1708.05024)).
  Popularity-weighted missing data — directly relevant to positive-unlabeled labels.
- **PureSVD** — Cremonesi, Koren & Turrin, RecSys 2010. Truncated SVD of the binary matrix; three
  lines, and a strong top-N baseline.

## C. Hybrid models that handle cold start — the core competitors

- **LightFM** — Kula 2015 ([arXiv:1507.08439](https://arxiv.org/abs/1507.08439)). Node embedding is
  the sum of its content-feature embeddings; WARP loss optimises rank directly. Fully cold-start
  capable given features. Library: `lightfm`. Features must be sparse/categorical, so occupancy and
  phenology need bucketing.
- **Factorization Machines** — Rendle, ICDM 2010. Factorised second-order interactions over a joint
  feature vector; cold-start capable via feature-only representation.
- **Two-tower retrieval with sampled softmax** — Yi et al., RecSys 2019, DOI 10.1145/3298689.3346996.
  Two encoders, scaled dot product, in-batch softmax. **Two corrections we currently lack** (see below).
- **DropoutNet** — Volkovs, Yu & Poutanen, NeurIPS 2017. Distils a warm CF model into a content-only
  network by dropping the preference input during training. Author code available.
- **Heater** — Zhu et al., SIGIR 2020, DOI 10.1145/3397271.3401178. Same distillation idea with a
  mixture-of-experts map from content to CF space.

## D. Graph neural networks

**The honest framing:** with zero training edges, every message-passing method collapses to its
feature encoder. We report this rather than omitting GNNs.

- **GraphSAGE** — Hamilton, Ying & Leskovec, NeurIPS 2017 ([arXiv:1706.02216](https://arxiv.org/abs/1706.02216)).
  Inductive, but only for a node with at least one edge; at zero degree it reduces to the layer-0
  feature MLP.
- **LightGCN** — He et al., SIGIR 2020 ([arXiv:2002.02126](https://arxiv.org/abs/2002.02126)).
  Purely transductive: layer-0 is a free ID embedding, so an unseen plant is unscorable.
- **IGMC** ([arXiv:1904.12058](https://arxiv.org/abs/1904.12058)), **GraIL** (ICML 2020),
  **NBFNet** ([arXiv:2106.06935](https://arxiv.org/abs/2106.06935)) — subgraph and path-based methods,
  inductive with respect to identity but all require non-empty local structure.

## E. Pairwise learning from the drug–target literature

The closest technical analogue: same bipartite cold-start structure, more mature evaluation.

- **Setting taxonomy** — Pahikkala et al., *Briefings in Bioinformatics* 16(2):325–337, 2015;
  formalised as settings A/B/C/D by Stock, Pahikkala, Airola, De Baets & Waegeman, *Neural
  Computation* 30(8):2245–2283, **2018** ([arXiv:1803.01575](https://arxiv.org/abs/1803.01575)).
  Adopt this vocabulary — both communities accept it.
- **KronRLS** — van Laarhoven, Nabuurs & Marchiori, *Bioinformatics* 27(21):3036–3043, 2011,
  DOI 10.1093/bioinformatics/btr500. Kernel ridge regression with a Kronecker pairwise kernel,
  eigendecomposed per side. Cold-start capable **only with feature kernels** (occupancy Jaccard,
  phenology RBF, taxonomic kernel), not the edge-derived interaction-profile kernel.
- **Two-step kernel ridge regression** — Stock et al. 2018; often better in settings B/C, closed-form
  leave-one-out.
- **DeepDTA-style dual encoder** — Öztürk, Özgür & Ozkirimli, *Bioinformatics* 34(17):i821–i829, 2018,
  DOI 10.1093/bioinformatics/bty593. Encode each side, **concatenate, then MLP** — a strictly more
  expressive interaction head than a dot product. Worth including alongside the two-tower.

## F. Positive-unlabeled learning

- **Elkan & Noto**, KDD 2008. Under SCAR, a P-vs-U classifier estimates the true posterior up to a
  constant; estimate and rescale. Cheapest possible correction.
- **nnPU** — Kiryo, Niu, du Plessis & Sugiyama, NeurIPS 2017 ([arXiv:1703.00593](https://arxiv.org/abs/1703.00593)).
  Non-negative PU risk; requires the class prior. Report a **sensitivity sweep over the assumed prior**.
- **One-Class CF** — Pan et al., ICDM 2008. Weighted low-rank approximation and degree-proportional
  negative sampling.

**Reporting consequence:** recall@k is a **lower bound** under positive-unlabeled labels, since
undocumented true interactions count as misses. State this, and pair it with the independent-evidence
slice (curated non-iNaturalist sources) as the closest thing to a clean test.

---

## Two defects in our current two-tower, from this literature

1. **No logQ correction.** In-batch negatives are sampled in proportion to popularity, so
   frequently-occurring pollinators are systematically over-penalised. Yi et al. 2019 correct the
   logits by subtracting the log sampling probability. With pollinator degree ranging from 2 (median)
   to 2,581 (max), this is not optional. See also "Correcting the LogQ Correction," RecSys 2025
   ([arXiv:2507.09331](https://arxiv.org/abs/2507.09331)).
2. **No mixed negative sampling.** Yang et al., WWW 2020 Companion, DOI 10.1145/3366424.3386195, add
   uniformly-sampled negatives from the full catalogue to the in-batch set, so that pollinators with
   no training edges appear as negatives at all. With a median degree of 2, most of our catalogue is
   in that tail.

Both are small changes to the existing training loop and should be made before the model is used as
our contribution's representative.


---

# ECOLOGY-SIDE METHODS

## G. Neutral / abundance models

**Vázquez neutral null** — Vázquez, Chacoff & Cagnolo 2009, *Ecology* 90(8):2039–2046,
DOI [10.1890/08-1837.1](https://doi.org/10.1890/08-1837.1). Interaction probability proportional to
the product of species' relative abundances; no fitting. Cold-start: yes for the abundance version;
**no** for the degree-marginal version, since a held-out plant has no degree — occupancy or record
count must be substituted, which is itself worth reporting. `bipartite::vaznull` (R) or ~15 lines.

**Abundance cross-product benchmark** — Dormann et al. 2025 (EcoEvoRxiv
[10.32942/X2S63Z](https://doi.org/10.32942/X2S63Z), **preprint**). Across 14 networks, abundance
dominates and traits and phylogeny add little; best Spearman ρ ≈ 0.4. The citation that legitimises
reporting the parameter-free encounter-rate benchmark.

**tapnet** — Benadi, Dormann, Fründ, Stephan & Vázquez 2022, *American Naturalist* 199(6):841–854,
DOI [10.1086/714420](https://doi.org/10.1086/714420). Expected frequency = abundance product × a
trait-matching kernel, with latent traits from phylogeny; maximum likelihood. CRAN `tapnet`.
Cold-start: yes. We can run the abundance + taxonomy-surrogate variant.

**Phenology × abundance likelihood model** — Vizentin-Bugoni, Maruyama & Sazima 2014,
*Proc. R. Soc. B* 281:20132397, DOI [10.1098/rspb.2013.2397](https://doi.org/10.1098/rspb.2013.2397).
Candidate probability matrices (abundance product, morphological match, phenological overlap)
multiplied elementwise and ranked by likelihood; forbidden links beat abundance alone.
**This is the ecological ancestor of our phenological-overlap term** and must be cited as the
hand-specified model our learned encoding is compared against. Reimplement, ~40 lines.

## H. Trait matching

**Pichler et al. 2020**, *Methods Ecol. Evol.* 11:281–293,
DOI [10.1111/2041-210X.13329](https://doi.org/10.1111/2041-210X.13329). Binary classification over
the pair table with plant traits, pollinator traits and their interactions; compares GLM, RF, boosted
trees, SVM, DNN under species-blocked CV. The framework is feature-agnostic, so substituting
phenology, occupancy and taxonomy for morphological traits gives a legitimate Pichler-style baseline —
the cheapest reviewer-satisfying comparison we can run. R `TraitMatching`, or ~30 lines sklearn.

**Two-step Kronecker kernel ridge regression** — Stock, Piot, Vanbesien, Meys, Smagghe & De Baets
2021, *Ecological Modelling* 451:109508,
DOI [10.1016/j.ecolmodel.2021.109508](https://doi.org/10.1016/j.ecolmodel.2021.109508). Kernel ridge
regression on K_plant ⊗ K_pollinator, closed-form via per-side eigendecomposition. Accepts occupancy
cosine, phenology RBF and taxonomic kernels — **no trait data required** — and its evaluation
formalism (settings A–D) matches our retrieval objective exactly. `RLScore`, or ~60 lines.

**Morphological matching (proboscis × corolla)** — e.g. Klumpers, Stang & Klinkhamer 2012,
*Oecologia*, DOI [10.1007/s00442-012-2290-3](https://doi.org/10.1007/s00442-012-2290-3). Requires
proboscis length and corolla depth. **Report as a limitation; do not benchmark.**

## I. Latent-variable models for interaction networks

**t-SVD / RDPG with taxonomic transfer** — Strydom et al. 2022, *Methods Ecol. Evol.* 13:2308–2319,
DOI [10.1111/2041-210X.13835](https://doi.org/10.1111/2041-210X.13835). Truncated SVD at **rank 12**
(~60% variance), latent positions imputed for unseen species by ancestral-state estimation under
Brownian motion, threshold at p = 0.22 by Youden's J. **The canonical ecology cold-start metaweb
method**, and our closest analogue. ~80 lines in Python.

**Covariate-informed latent factors (COIL/COIL+)** — Kampe, DeSisto & Dunson 2026,
*Methods Ecol. Evol.*, DOI [10.1111/2041-210X.70368](https://doi.org/10.1111/2041-210X.70368).
Bayesian bipartite latent factors with adaptive rank shrinkage, covariates and phylogeny in the prior,
and explicit correction for study-level taxonomic sampling bias. Directly designed for our
positive-unlabeled, source-biased setting. Substantial to implement.

**Latent factors with implicit feedback** — Seo & Hutchinson, **AAAI-18**, pp. 808–815
([page](https://aaai.org/papers/11345-predicting-links-in-plant-pollinator-interaction-networks-using-latent-factor-models-with-implicit-feedback/),
code [Hutchinson-Lab/pollination-networks-implicit-feedback](https://github.com/Hutchinson-Lab/pollination-networks-implicit-feedback)).
Weighted matrix factorisation treating unobserved cells as weak negatives, evaluated with
recommender-system ranking metrics, on plant–pollinator data. **This is the closest existing work to
our framing** — the positive-unlabeled treatment and the ranking evaluation are both already there.
It is transductive, which is where our contribution differs.

**Bipartite VGAE with sampling-effort decorrelation** — Anakok et al. 2024,
[arXiv:2403.02011](https://arxiv.org/abs/2403.02011) (**preprint**). Two GCN encoders with an HSIC
penalty decorrelating embeddings from effort covariates. Relevant given our network is
iNaturalist-dominated.

**cassandRa structural suite** — Terry & Lewis 2020, *Ecology* 101(7):e03047,
DOI [10.1002/ecy.3047](https://doi.org/10.1002/ecy.3047). Fits SBM, matching-centrality, connectance
and coverage-deficit models in one call (CRAN `cassandRa`). Transductive; one hour of work for the
structural upper reference.

## J. Phylogenetic and taxonomic transfer

**Hierarchical Bayesian phylogenetic affinity** — Elmasri, Farrell, Davies & Stephens 2020,
*Annals of Applied Statistics* 14(1):221–240,
DOI [10.1214/19-AOAS1296](https://doi.org/10.1214/19-AOAS1296); applied in Farrell et al. 2022,
*J. Animal Ecology* 91:1875–1888, DOI [10.1111/1365-2656.13666](https://doi.org/10.1111/1365-2656.13666).
Latent-score model with a scaled phylogenetic affinity term plus per-species degree effects; phylogeny
and degree together beat either alone.

**Congeneric / taxonomic nearest-neighbour transfer** — not a named method, but the phylogenetic-signal
baseline inside Strydom 2022, Stock 2021 and Foster et al. 2026. Score a held-out plant by the
taxonomically-weighted mean interaction vector of its relatives. ~25 lines, and with median plant
degree 3 it will be competitive.

## K. Co-occurrence — the argument to pre-empt

**Blanchet, Cazelles & Gravel 2020**, *Ecology Letters* 23:1050–1063,
DOI [10.1111/ele.13525](https://doi.org/10.1111/ele.13525), and *Nature Reviews Biodiversity* 2025,
DOI [10.1038/s44358-025-00105-1](https://doi.org/10.1038/s44358-025-00105-1). Co-occurrence is not
evidence of interaction. Our spatial feature block is precisely this paper's target. The defence:
co-occurrence enters as a *necessary-condition prior* supervised against observed interactions, never
as evidence of interaction — and we quantify how far it alone carries the prediction.

**HMSC** — Tikhonov et al. 2020, *Methods Ecol. Evol.* 11:442–447,
DOI [10.1111/2041-210X.13345](https://doi.org/10.1111/2041-210X.13345). Joint species distribution
model; associations read from the residual covariance. Conceptual foil rather than a supervised
comparator, and infeasible to fit jointly at our scale.

## L. What recent papers benchmark against, and with which metrics

| paper | baselines | metrics |
|---|---|---|
| Biton, Puzis & Pilosof 2025, *Nat. Ecol. Evol.* 9:1214–1223 | transductive embedding models; within- vs cross-network training | **precision, F1** — explicitly not ROC-AUC |
| Foster et al. 2026, *Oikos*, DOI 10.1002/oik.11156 | phylogeny-only, trait-only, latent-SVD-only, combined | AUC + RMSE; latent SVD dominates |
| Kampe et al. 2026 | latent factors ± bias correction, ± covariates | held-out AUC, count of revealed links |
| Van Kleunen et al. 2026 (preprint) | stacked RF over bipartite CN, RA, Jaccard, Adamic–Adar, PageRank, low-rank, kNN | ROC-AUC **and** PR-AUC |
| **Poisot 2023**, *Methods Ecol. Evol.* 14:1333–1345, DOI [10.1111/2041-210X.14071](https://doi.org/10.1111/2041-210X.14071) | — | mandates **PR-AUC, MCC, informedness**; states ROC-AUC is uninformative at this imbalance |

At connectance 0.1%, a headline ROC-AUC will draw an objection. Report PR-AUC, recall@k, and MCC.

---

# PHENOLOGY: WHAT IS ALREADY ESTABLISHED

## M. How phenological overlap is actually formulated in this field

The standard operationalisation is the **outer-product phenology probability matrix**: presence or
abundance vectors over sampling periods for each species, outer product, normalised to sum to one.
Used this way by Vizentin-Bugoni, Maruyama & Sazima 2014, *Proc. R. Soc. B* 281:20132397, and
Gonzalez & Loiselle 2016, *PeerJ* 4:e2789, DOI [10.7717/peerj.2789](https://doi.org/10.7717/peerj.2789).
Related niche-overlap statistics: **Schoener's index** (Schoener 1970, *Ecology* 51:408–418) and
**Pianka's index** (Pianka 1974, *PNAS* 71:2141–2145) — both established for niche overlap, neither
established as a link predictor.

⚠️ **Citation correction.** Earlier drafts attributed Δ to Ridout & Linkie 2009, *JABES* 14:322–337.
That paper estimates overlap of **daily (diel) activity patterns from camera-trap data** — its
worked example is Sumatran felids. It is not a seasonal-phenology method, and citing it for a 52-week
seasonal overlap is a misattribution. Corrected throughout the repository.

For a 52-week curve the principled treatment is **circular statistics** (von Mises; Morellato et al.,
*Phenological Research*, Springer, DOI 10.1007/978-90-481-3335-2_16) — which is notably *not* what the
interaction-prediction literature does.

## N. Forbidden links — the reported magnitudes

- **Olesen, Bascompte, Dupont, Elberling, Rasmussen & Jordano 2011**, *Proc. R. Soc. B* 278:725–732,
  DOI [10.1098/rspb.2010.1371](https://doi.org/10.1098/rspb.2010.1371): **phenological uncoupling
  explains 22–28% of possible-but-unobserved links.**
- **Duchenne et al. 2025**, *Ecology Letters* 28:e70073,
  DOI [10.1111/ele.70073](https://doi.org/10.1111/ele.70073): probabilistic reanalysis across 32
  plant–hummingbird communities puts forbidden links at **3–29%** — lower than the deterministic
  literature implies.
- **Jordano 2016**, *Functional Ecology* 30:1883–1893, DOI 10.1111/1365-2435.12763: much apparent
  forbiddenness is undersampling. Cite alongside any forbidden-link claim.

## O. Does phenology improve prediction? The record is split by system

**Phenology wins — vertebrate pollinators, local networks.** Gonzalez & Loiselle 2016 (Andean
bird–flower): phenology model ΔAIC = 0, abundance ΔAIC = 428.2, null ΔAIC = 588.6.
Vizentin-Bugoni et al. 2014 reach the same conclusion for hummingbirds.

**Phenology adds little — insect visitation, once abundance is controlled.** Dormann et al. 2025
(14 networks) find abundance dominant and trait/phylogenetic information adding nothing substantial
beyond it, with overall predictive power low (ρ ≈ 0.4). Benadi et al. 2022 (tapnet) find traits +
phylogeny + abundance on a par with abundance alone.

**Resolution-dependent.** "Seasonal Variation Mediates the Importance of Species Attributes in
Plant–Pollinator Interactions," bioRxiv 2026.07.21.739935 (*authors unverified*): abundance explains
most variation, trait matching secondary, and **phenology plays a minor role at broad temporal
resolution but becomes more important at finer resolution.** This is the closest paper to our framing
and must be cited.

## P. Time-resolved prediction — a genuine gap

CaraDonna et al. 2017, *Ecology Letters* 20:385–394, DOI 10.1111/ele.12740, show week-to-week
interaction turnover is dominated by rewiring, constrained by phenology and abundance. But almost all
link-prediction work **aggregates over time entirely** (Pichler 2020; Terry & Lewis 2020; Biton 2025).
Dormann et al. 2025 test transfer across years; the botanical-gardens preprint varies temporal
resolution. Time-resolved prediction remains rare.

**NECTAR** — bioRxiv 2026.03.30.715389 — builds a spatially explicit Californian metaweb from
distributions, phenometrics and phylogeny: 1,247,081 interactions over 5,131 pollinators × 5,178
plants. The closest existing system to ours; its internal phenology encoding is unverified and should
be checked before we claim novelty against it.

## Q. Positioning consequences

1. **Our claim is a representation claim, not an ecological one.** "Phenology drives interactions" is
   contradicted by Dormann et al. 2025 for insect networks. What is open is that **nobody has compared
   encodings of the same phenological signal — scalar vs. raw curve vs. learned per-cell — holding
   data, split and model class fixed.** The literature varies the predictor set, not the representation.
2. **A co-occurrence-only baseline is mandatory**, or any phenology effect reads as an abundance proxy.
3. **The expected honest finding** is "curves beat scalars, per-cell ≈ curves, all modest relative to
   co-occurrence." That is publishable as a representation benchmark with a clean protocol; it is not
   publishable as a claim that phenology drives interactions.

---

# PRETRAINED REPRESENTATIONS AS INPUTS

## R. Species-level representations

| model | citation | supervision | dim | weights |
|---|---|---|---|---|
| BioCLIP | Stevens et al., CVPR 2024 (arXiv:2311.18803) | CLIP on 10M images × **Linnaean taxonomy strings** | 512 | `imageomics/bioclip`, MIT |
| BioCLIP 2 | Gu et al., NeurIPS 2025 (arXiv:2505.23883) | same objective, TreeOfLife-200M | 768 | `imageomics/bioclip-2`, MIT |
| **TaxaBind** | Sastry, Khanal, Dhakal, Ahmad & Jacobs, **WACV 2025** (arXiv:2411.00683) | six encoders bound to BioCLIP image space | 512 | `MVRL/taxabind-vit-b-16`, Apache-2.0 |
| ProM3E | Sastry, Khanal, Dhakal, Lin, Cher, Jarosz & Jacobs (arXiv:2511.02946, CVPR'26) | masked-modality reconstruction over TaxaBind modalities | — | pending release |

**The supervision determines the content.** BioCLIP's training label *is* the Linnaean string, so
embeddings that behave like taxonomy are the expected outcome, not a defect. TaxaBind inherits BioCLIP
as teacher, so its image and text arms carry the same signal — its **location and environmental arms**
are the non-taxonomic parts and the reason to test it.

## S. Location and range representations

- **SINR** — Cole, Van Horn, Lange, Shepard, Leary, Perona, Loarie & Mac Aodha, ICML 2023
  ([arXiv:2306.02564](https://arxiv.org/abs/2306.02564)). Location encoder gives a 256-d per-coordinate
  feature **and** a per-species 256-d class embedding for 47K species, readable directly off the
  classifier weight matrix. **The most directly usable artifact for us** — a free implicit counterpart
  to our occupancy PCA.
- **LE-SINR** — Hamilton et al., NeurIPS 2024 ([arXiv:2410.10931](https://arxiv.org/abs/2410.10931)).
  Species vector from free-form habitat text; covers species with zero occurrences.
- **FS-SINR** — Lange et al., ICML 2025 ([arXiv:2502.14977](https://arxiv.org/abs/2502.14977)).
  Few-shot species encoding — relevant for rare pollinators.
- **RANGE** — Dhakal, Sastry, Khanal, Ahmad, Xing & Jacobs, **CVPR 2025**
  ([arXiv:2502.19781](https://arxiv.org/abs/2502.19781)), MIT. 1280-d location embeddings, reported to
  beat SatCLIP/GeoCLIP/CSP.
- **SatCLIP** (Klemmer et al., AAAI 2025), **GeoCLIP** (NeurIPS 2023), **CSP** (ICML 2023),
  **MOSAIKS** (Rolf et al., *Nat Commun* 12:4392) — per-location encoders.

**Deriving a per-species vector**: pool location embeddings over a species' occupied cells. This makes
any location encoder a drop-in replacement for our occupancy PCA with no architecture change — exactly
the substitution the explicit→implicit axis calls for.

## T. Environmental representations

**TaxaBind's environmental encoder** (MLP over WorldClim 2.1: 19 bioclim variables + elevation,
projected into the shared 512-d space) is the closest ready-made bioclim embedding, already aligned
with species images and text. **AlphaEarth Foundations** (arXiv:2507.22291) gives 64-d per-pixel
annual embeddings via Earth Engine; cheap to aggregate to 0.5°.

## U. The gap

**No published work feeds BioCLIP, TaxaBind, SINR, SatCLIP, GeoCLIP or RANGE embeddings into a
species-interaction predictor.** Adjacent work uses embeddings learned *from the interaction graph
itself* (Strydom et al. 2022, 2023) or hand-crafted traits (Pichler et al. 2020). LLM work in this
space is interaction *extraction* from text, not prediction.

**"Pretrained representations as inputs to interaction prediction" is unclaimed**, and it is the
natural home for the explicit→implicit axis.

## V. Trait recovery — the one direct probe

BioCLIP 2 (§5.4 of arXiv:2505.23883) reports that embedding geometry aligns with functional meaning —
beak size emerges along a principal axis, habitat classes separate. **But taxonomy and morphology are
confounded, and the paper does not partial out phylogeny.** That control is exactly what a reviewer
would demand, and running it is a contribution we could make cheaply.

## W. Recommended representations to test, ordered

1. **SINR species embeddings** — free, 47K species, occurrence-supervised. *Expected failure:* it is a
   compressed range descriptor and may simply re-encode the range-size shortcut. Must be tested against
   the co-occurrence null explicitly.
2. **TaxaBind** — species, location and environment in one aligned space; in-house. *Expected failure:*
   image/text arms collapse to taxonomy, as BioCLIP did; the environmental arm may add nothing over raw
   bioclim.
3. **LE-SINR text→species vectors** — habitat semantics from language. *Expected failure:* thin
   Wikipedia coverage across our pollinator taxa.
4. **RANGE location embeddings pooled per species** — best-reported location encoder. *Expected
   failure:* 1280-d pooled over 3,162 cells is high-variance, and pooling reintroduces range size.
5. **BioCLIP 2 image embeddings** — as the taxonomy control arm. Expect it to match a taxonomic-distance
   baseline; report that as the negative result it is.

---

# CONSOLIDATED SHORTLIST

Ordered by how strongly each is expected, with who expects it.

| # | method | expected by | cold-start | cost |
|---|---|---|---|---|
| 1 | Popularity / degree null | both | yes | 1 h |
| 2 | Co-occurrence null | both | yes | 1 h |
| 3 | Abundance neutral model (Vázquez 2009) | ecology | yes* | 2 h |
| 4 | Congeneric / taxonomic transfer | ecology | yes | 2 h |
| 5 | t-SVD + taxonomic imputation (Strydom 2022) | ecology | yes | 1 d |
| 6 | Two-step Kronecker KRR (Stock 2021) | both | yes | 1 d |
| 7 | Pichler-style RF/BRT on pair features | ecology | yes | 4 h |
| 8 | Phenology × abundance likelihood (Vizentin-Bugoni 2014) | ecology | yes | 4 h |
| 9 | LightFM (WARP) | ML | yes | 4 h |
| 10 | Two-tower + logQ + mixed negatives | ML | yes | 2 d |
| 11 | BPR-MF / iALS *(warm-only reference)* | ML | **no** | 2 h |
| 12 | LightGCN or GraphSAGE *(warm-only reference)* | ML | **no** | 2 d |

*The degree-marginal variant is not cold-start capable; occupancy or record count must be substituted.

**Also cite without benchmarking:** Blanchet et al. 2020 (co-occurrence ≠ interaction — address
directly), Poisot 2023 (adopt its metrics), Seo & Hutchinson AAAI-18 (closest prior framing),
Aiyappa et al. ICML 2025 (the degree-shortcut result), Dormann et al. 2025 (phenology adds little
over abundance).
