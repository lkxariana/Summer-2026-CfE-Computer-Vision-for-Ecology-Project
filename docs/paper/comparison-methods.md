# Comparison methods

Candidate benchmark methods for cold-start bipartite interaction prediction. Our task is
**Setting B** in the pairwise-prediction taxonomy (Stock et al. 2018): a left node (plant) held out
entirely, all right nodes (pollinators) seen. Every entry states whether it can score a plant with
zero training edges — this disqualifies most of the standard link-prediction toolkit and must be
reported rather than quietly avoided.

*(Ecology-side methods pending; this section covers the ML/CS literature.)*

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
