# 07 — Reading List

Ordered as a reading path from foundational framing to methods to extensions. All items are relevant to the paper's positioning, methodology, or future directions.

---

## 1. Framework — Read First

**Poisot et al. 2021.** *A roadmap toward predicting species interaction networks across space and time.* Philosophical Transactions of the Royal Society B, 376(1837), 20210063. https://doi.org/10.1098/rstb.2021.0063

The direct framework this paper extends. Poisot establishes the two-stage prediction logic (availability → realized interaction), demonstrates that latent-variable models can identify interactions between species never observed to co-occur, and sets the bar that ANTHEIA's spatial hold-out must clear. The "Netflix problem" framing of interaction network prediction comes from here.

---

## 2. Framing Constraint — Most Important

**Blanchet, Cazelles & Gravel 2020.** *Co-occurrence is not evidence of ecological interactions.* Ecology Letters, 23(7), 1050–1063. https://doi.org/10.1111/ele.13525

Why claims must be framed as *opportunity*, not interaction. This is the paper a reviewer will cite if the scope is overclaimed. Read before writing the abstract.

**Coelho et al. 2024.** *Quantifying the causes and consequences of variation in plant-pollinator interaction rewiring.* Nature Ecology & Evolution. https://doi.org/10.1038/s41559-023-02254-y

Quantifies that only ~20% of species co-occurrences correspond to real interactions. Pairs directly with Blanchet as the empirical grounding for why spatial co-occurrence alone is insufficient.

---

## 3. Methods Hygiene

**Dormann et al. 2018.** *Biotic interactions in species distribution modelling: 10 questions to guide interpretation and avoid false conclusions.* Global Ecology and Biogeography, 27(9), 1004–1016. https://doi.org/10.1111/geb.12759

The cautions a reviewer will hold you to. The key assumption ANTHEIA relaxes: existing interaction SDMs "must assume biotic interactions are constant in space and time." PPE's temporally-resolved flowering surface directly addresses this.

---

## 4. Phenology-Specific Framing

**Rafferty et al. 2025.** *Complexities of phenological shifts for plant-pollinator interactions and ways forward.* Integrative and Comparative Biology. https://doi.org/10.1093/icb/icaf034

Temporal overlap is an *opportunity* metric, not interaction strength or mismatch magnitude. Scope claims accordingly. Also relevant for the brittleness direction — phenological mismatch under climate change is the ecological mechanism Direction 4 is designed to detect.

---

## 5. Observation Bias Vocabulary

**Carlen et al. 2024.** *A framework for contextualizing social-ecological biases in contributory science data.* People and Nature. https://doi.org/10.1002/pan3.10592

Provides vocabulary for the bias types present in GBIF and iNaturalist data: participation bias, detectability bias, sampling bias, preference bias. Plant-side bias in ANTHEIA is primarily detectability and preference (conspicuous flowering species are overrepresented); pollinator-side is primarily sampling (observer effort, not true activity). Useful for the limitations section and for framing the SDM motivation.

---

## 6. Empirical Precedent

**Bosenbecker et al. 2023.** *Contrasting nation-wide citizen science and expert collected data on hummingbird-plant interactions.* Perspectives in Ecology and Conservation, 21(2), 164–171. https://doi.org/10.1016/j.pecon.2023.03.004

Citizen science vs. expert interaction data: fewer than 40% of partners are shared between sources, and photo count drives records. Near one-line motivation for why GBIF-derived interaction data is incomplete and biased. Particularly relevant for the shared iNaturalist provenance limitation.

---

## 7. Overlap Metric

**Ridout & Linkie 2009.** *Estimating overlap of daily activity patterns from camera trap data.* Journal of Agricultural, Biological, and Environmental Statistics, 14(3), 322–337. https://doi.org/10.1198/jabes.2009.08038

The source for the coefficient of overlapping Δ = Σ min(f̃, ã) used as ANTHEIA's temporal overlap scalar. Defined for circular activity data, which day-of-year is (December wraps to January). The correct metric for phenological synchrony — not linear KDE.

---

## 8. Spatial Co-occurrence Baseline

**Galiana et al. 2021.** *The spatial scaling of species interaction networks.* Nature Ecology & Evolution, 5(2), 223–230. https://doi.org/10.1038/s41559-020-01353-4

The source for the shared bin count N (Galiana co-occurrence count) used as the spatial baseline feature in ANTHEIA. Also provides the degree-distribution framing (super-generalists, power-law networks) relevant to the brittleness direction.

---

## 9. Extension: Spatiotemporal Species Mapping

**Cole et al. 2023.** *Spatial implicit neural representations for global-scale species mapping.* ICML 2023. arXiv:2306.02564.

The SINR model — a shared implicit neural representation with per-species heads for presence-only species range mapping at global scale. The natural precedent for Direction 1 (fully differentiable pipeline) and the broader reframing of ANTHEIA as a spatiotemporal SDM. SINR handles cold-start species and scales because heads are per-species, not per-pair.

---

## 10. PPE — Phenological Plant Embeddings

**Cher, D. et al.** *PhenoField / PPE (Phenological Plant Embeddings).* Unpublished manuscript / preprint in preparation.

The climate-driven flowering probability surface that provides the plant-side temporal signal in ANTHEIA. PPE feeds a pretrained CrossModalVAE backbone on weekly PRISM climate windows to produce per-species, per-bin, per-week flowering probability curves across CONUS. These curves are the basis for the f_curves input and the spatiotemporal plant embeddings V_δ used in ANTHEIA-4D and ANTHEIA-15D.

**Note:** PPE has not been formally published as of the time of writing. There is no official citation. When referencing PPE in the paper, coordinate with Dan Cher on the correct attribution format — it may appear as a preprint, a workshop paper, or an internal technical report by the time of submission.
