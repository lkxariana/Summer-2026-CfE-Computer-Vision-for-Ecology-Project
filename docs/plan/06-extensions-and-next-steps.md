# 06 — Extensions and Next Steps

This document covers two categories of future work: (1) the immediate next step that must be completed before the archival paper can be finalized, and (2) four longer-term directions proposed by Dan Cher for potential inclusion in the archival paper. These are research directions, not implementation tasks — they are recorded here to preserve the initial thinking and to give context for the paper's discussion section.

---

## Immediate Next Step: Full SDM Integration

### What SDM Is

SDM (Species Distribution Model) provides an alternative source of pollinator activity curves (`a_curves`) — the 52-week temporal signal representing when a pollinator species is active across the year.

In the current pipeline, `a_curves` are derived from GBIF occurrence records: for each pollinator species, observation counts are binned by week to produce a normalized 52-week histogram. This is purely data-driven but subject to GBIF sampling bias — weeks with many observations reflect recorder activity, not genuine pollinator phenology.

SDM replaces these GBIF-derived curves with model-predicted activity curves produced by Dan Cher's species distribution models, which estimate true phenological activity independently of observation effort.

SDM is **not a new model architecture**. It is a different data source for the pollinator temporal signal. The five model architectures remain unchanged.

### Why SDM Is Needed: The Asymmetry Finding

The need for SDM is directly motivated by the pollinator-side ablation experiments. Testing GBIF-derived temporal enrichment on the pollinator side — both as a continuous observation histogram (PMp) and as a binary temporal existence matrix (TMp) — consistently degraded performance below the Spatial Baseline. The failure is not data scarcity but data quality: GBIF observations are biased by sampling effort, not genuine phenological signal.

The core asymmetry:

| Side | Temporal source | Effect |
|---|---|---|
| Plant | PPE (climate-driven, model-predicted) | Consistently improves |
| Pollinator | GBIF (observation-density biased) | Consistently hurts |
| Pollinator | SDM (model-predicted) | Hypothesis: should improve |

SDM is the attempt to close this asymmetry — giving the pollinator side the same quality of temporal signal that PPE gives the plant side.

### Model Naming

| GBIF a_curves | SDM a_curves |
|---|---|
| Spatial Baseline | Spatial Baseline-SDM |
| ANTHEIA-Scalar | ANTHEIA-Scalar-SDM |
| ANTHEIA-4D | ANTHEIA-4D-SDM |
| ANTHEIA-15D | ANTHEIA-15D-SDM |
| ANTHEIA-PMf | ANTHEIA-PMf-SDM |

### Current Status and What Needs to Be Done

Dan's SDM currently covers **1,615 pollinator species** — a small test set. The full SDM for all **25,466 species** is pending.

Once the full SDM arrives:

1. Rerun the SDM comparison with the full SDM `a_curves`
2. Recompute the shared pair universe (expected to expand from 122 toward 139 positive pairs)
3. Run 5-seed testing (seeds [42, 0, 1, 2, 3]) and record mean ± std
4. Update all results tables accordingly
5. The GBIF vs. SDM comparison becomes the primary ablation in the paper

**This is the first task to complete when the semester resumes in September.**

### Key Technical Notes

- SDM `a_curves` must use the same 52-week format (`week = (doy - 1) // 7`, clipped to [0, 51]) as GBIF `a_curves`
- The shared pair universe must be recomputed under SDM species coverage
- Negative sampling uses `np.random.default_rng(seed)` — same seeds [42, 0, 1, 2, 3]
- Do not mix GBIF and SDM curves within a single run
- SDM curves require a coordinate offset correction: subtract 0.25 from both lat and lon to match the F/P matrix bin convention

---

## Longer-Term Directions (Dan Cher, July 2026)

The following four directions were proposed by Dan as potential extensions for the archival paper. They are not implementation tasks for now — they are recorded here to preserve the initial thinking.

---

### Direction 1 — Fully Differentiable Pipeline

**Dan's proposal:** Currently PCA is used to reduce dimensionality, then the compressed vectors are passed to a logistic regression probe. Could try making the pipeline fully differentiable.

**Current architecture:**
```
F matrix → PCA → Vf (15D) ─┐
P matrix → PCA → Vp (15D) ─┼→ Logistic Regression → P(interaction)
N, Δ ──────────────────────┘
```

**What fully differentiable means:** Replace PCA and logistic regression with a neural encoder trained end-to-end. The encoder would learn species embeddings that are task-specific — optimized for interaction prediction rather than variance preservation — using backpropagation through the full pipeline.

**Initial thinking (Ori):** The main concern is data volume. With only 139 positive pairs, training a neural encoder is likely to overfit. This direction becomes viable when the positive pair set grows — either through expanded GloBI coverage, data augmentation, or a larger geographic scope. A useful intermediate step might be to first expand the positive set and verify that the logistic regression continues to improve before introducing a more expressive classifier.

---

### Direction 2 — Climate Encoder as Temporal Representation

**Dan's proposal:** Pass temporal information in other ways — do you need PPE? Can you just pass weekly observation data directly, or pass climate information from a climate encoder (e.g. ERA5)?

**The key insight (Ori):** If using a climate encoder like ERA5, time as an axis is already encoded into the climate component itself. ERA5 is a temporal climate dataset; a model trained on ERA5 embeddings learns seasonal patterns, temperature cycles, and phenological signal implicitly. In that case, PPE's explicit 52-week flowering curves may be partially redundant — the climate embedding already contains the temporal signal PPE is trying to encode.

**What this would look like experimentally:**
- Current: ANTHEIA-Scalar uses explicit Δ = Σ min(f_t, a_t)
- Alternative: replace Δ with an ERA5 climate embedding at the plant's location
- Compare performance — if ERA5 embedding matches or exceeds Δ, the PPE step may be replaceable

**Connection to Direction 1:** A climate encoder is the natural backbone for a fully differentiable pipeline. ERA5 embeddings could serve as the plant-side representation, replacing both Vf and the PPE curves in a single unified input.

---

### Direction 3 — iNaturalist Imagery Pipeline

**Dan's proposal:** Use the multi-species benchmark pipeline from BioWM to generate interaction predictions from iNaturalist imagery. Instead of using actual occurrence observations, take a species classification model's predictions (knowing whether it is a pollinator or a plant) and pass those predictions to a model that predicts plant-pollinator relationships. A preponderance of predicted co-occurrence in imagery may outperform raw observations. Species classification models are getting good enough that prediction noise is no longer prohibitive.

**What this replaces:** The current pipeline uses GBIF occurrence records to build spatial existence matrices and activity curves. This direction would replace occurrence records with model-predicted species presence derived from ground-level imagery — closer to what a field ecologist would observe directly.

**Why this is interesting:** Ground-level imagery captures the plant and pollinator in the same image at the same location and time — a much stronger signal of potential interaction than two separate occurrence records that happen to share a spatial bin. The noise concern (classification errors) is the tradeoff, but as Dan notes, species classification models have improved substantially.

---

### Direction 4 — Brittle Relationship Identification

**Dan's proposal:** How would you define a brittle plant-pollinator relationship mathematically?

**Proposed criteria (Ori):** A plant-pollinator interaction may be considered brittle if any of the following conditions hold:

1. **Data sparsity:** fewer than 2 observations per 0.5° × 0.5° grid cell for either the plant or pollinator species
2. **Low interaction probability:** P(interaction) < 0.001 under the ANTHEIA model
3. **Declining observations:** observation count decreasing at a rate greater than −0.5 per year for either species
4. **Declining temporal overlap:** overlap coefficient Δ decreasing at a rate greater than −0.3 per year

Criteria 1 and 2 identify currently fragile interactions. Criteria 3 and 4 identify dynamically deteriorating ones — relationships that are not just fragile now but becoming more fragile over time. This distinction is ecologically important: criterion 4 in particular captures phenological mismatch under climate change, where a plant's flowering window and a pollinator's activity window are drifting apart.

**What this requires:** Criteria 3 and 4 require year-level temporal resolution in the PPE opportunity surface or in GBIF observations — a time series of Δ per pair per year rather than a single averaged surface. PhenoField has ~10 years of iNaturalist observations that could support this. Whether the PPE opportunity surface can be queried at year-level resolution is a question for Dan.

**Why this matters for the paper:** Brittle relationship identification is a direct downstream application of ANTHEIA's output — the model's interaction probability and temporal overlap scores can be operationalized into a brittleness flag. This gives the paper a concrete ecological use case beyond link prediction performance numbers.
