# 05 — Scope and Limitations

---

## Scope

**Interaction type:** Plant-pollinator flower visitation only. The phenophase throughout is flowering. Frugivory, herbivory, seed dispersal, and other interaction types are out of scope.

**Geography:** CONUS (continental United States), at 0.5° × 0.5° spatial resolution. This matches PhenoField's training domain and the PPE opportunity surface coverage.

**Temporal window:** 2013–2026, matching PhenoField's observation window and the PPE climate averaging period.

**Claim:** ANTHEIA predicts interaction *likelihood* given spatial co-occurrence and phenological alignment — not interaction strength, not fitness consequences, not network-level properties. The distinction between co-occurrence opportunity and realized interaction is load-bearing: Coelho et al. (2024) quantify that only ~20% of species co-occurrences correspond to real interactions. ANTHEIA's predictions should be interpreted as opportunity scores, not confirmed interactions.

---

## Limitations

### Small positive set
The shared pair universe contains only 139 GloBI-documented positive interactions across 6,348 plant species and 24,939 pollinator species. This is a fundamental constraint of GloBI's coverage, not a preprocessing choice. The small positive set drives the seed-to-seed variance observed in results (±0.02–0.04 ROC-AUC) and limits statistical power for fine-grained model comparisons. Differences of less than ~0.005 ROC-AUC between models should not be over-interpreted.

### No true negatives
GloBI documents known interactions, not confirmed non-interactions. Negative pairs in ANTHEIA are randomly sampled non-pairs — pairs with no documented interaction record — which is not the same as confirmed non-interacting pairs. The model is trained on documented vs. undocumented, not interacting vs. non-interacting. This is standard practice in ecological link prediction but limits the interpretability of precision and threshold-based metrics.

### Pollinator-side data quality
GBIF pollinator activity curves (a_curves) are derived from opportunistic citizen science observations biased by sampling effort. Weeks with many observations reflect recorder activity, not genuine pollinator phenology. This is the core asymmetry finding: plant-side temporal enrichment (PPE, model-predicted) consistently improves performance; pollinator-side temporal enrichment (GBIF, observation-biased) consistently hurts. SDM activity curves are the intended solution, but the full 25,466-species SDM is pending.

### Shared iNaturalist provenance
GloBI interaction records, GBIF pollinator occurrences, and PhenoField/PPE training data are all largely iNaturalist-sourced. The model may partially learn observation-density patterns — where observers were active — rather than genuine ecological signal. The SDM comparison is partly motivated by this concern, as SDM-derived activity curves are independent of iNaturalist observation effort. The go/no-go experiments (now superseded) identified this circularity explicitly when validating an iNat-trained probe against iNat-derived ground truth.

### PPE compression behavior
PPE's flowering curves (f_curves) are derived from multi-year-averaged climatology, producing inherently smooth, broad curves that rarely approach zero in any given week. This compresses the distribution of temporal overlap scores Δ toward a moderate middle band — pairs with genuinely low temporal overlap may receive inflated Δ values because PPE's plant curve never drops low enough to produce a confident near-zero score. The observation-based plant curves used in early exploratory analysis showed sharper peaks and troughs, which could produce more discriminating Δ values for some pairs at the cost of observation-density bias. This is a known tradeoff, not a correctable error.

### Logistic regression as classifier
The choice of logistic regression is deliberate for interpretability but imposes linearity. The model cannot learn nonlinear interactions between features. This is appropriate for ablation purposes — any gain from adding a feature is attributable to that feature, not to increased model capacity — but may leave performance on the table relative to more expressive classifiers. The paper's contribution is the feature engineering and the phenological signal, not the classifier architecture.

### SDM results are preliminary
The SDM comparison currently uses 1,615 pollinator species (1,275 after intersection with Vp), yielding 122 positive pairs. The full SDM (25,466 species) is pending. Current SDM results should be treated as directional indicators, not final numbers. The model ranking under SDM — ANTHEIA-PMf-SDM and ANTHEIA-15D-SDM leading — may shift once the full species pool is available.

### Human population bias
Both PhenoField and GBIF draw heavily from iNaturalist, where observation density correlates strongly with human population density. Urban and suburban areas near major cities are systematically overrepresented; rural interior regions, despite potentially high ecological diversity, are data-sparse. This means the plant existence matrix F and pollinator existence matrix P are not geographically uniform samples — they reflect where observers live and travel, not where species actually occur. The PCA embeddings Vf and Vp therefore partially encode human geography rather than pure ecological distribution. This is reflected in Vf's relatively low variance explained (39.9% in 15 components) compared to what a truly unbiased sample might yield. PPE's f_curves partially mitigate this on the plant side by using climate-driven predictions rather than observations, but Vp and a_curves on the pollinator side remain subject to this bias.

### Spatial resolution
At 0.5° resolution (~55 km × 43 km per bin), ANTHEIA cannot capture fine-scale habitat heterogeneity within a bin. A plant and pollinator may co-occur in the same bin but occupy non-overlapping microhabitats. This is an inherent constraint of the data sources (PhenoField and GBIF resolution) rather than a modeling choice.
