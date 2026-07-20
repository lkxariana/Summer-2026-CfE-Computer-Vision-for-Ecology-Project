> Part of [Idea 6](README.md). Prev: [Overview](01-overview.md) · Next: [Experiments and ablations](03-experiments-and-ablations.md).

# PPE integration: the flowering field, the probe, and the overlap scalar

## How PPE is used

Frozen, as a queryable phenology field — no retraining. Query the backbone, run a probe (NPN- or Pheno3M-trained; see *Which probe* below) → a phenophase probability in [0,1]. The opportunity surface is then *computed*, not learned, which lets the method sidestep the negative problem.

**Inputs.**
- To the PPE backbone, per query: species `P`, location `ℓ`, day-of-year `t`, the 365-day PRISM climate window at `ℓ`, and the AlphaEarth embedding at `ℓ` → embedding → probe → `flower(P, ℓ, t)`.
- Pollinator term (separate, no PPE): that species' GBIF occurrence dates → sparse-data activity estimator (KDE or quantile) → `activity(A, ℓ, t)`, pooled regionally.
- Opportunity model input: a tuple `(P, A, ℓ, t)`.

**Output.**
- `opportunity(P, A, ℓ, t) = flower(P, ℓ, t) × activity(A, ℓ, t)` — a continuous surface over CONUS × day-of-year for the pair. Keep the full surface, or integrate over `t` for a seasonal map.

**Objective.**
- No training objective — PPE frozen, pollinator estimator fit by KDE, opportunity is a computed product. The objective lives in *evaluation*: presence-only / SDM-style. Held-out co-observations of the pair should fall at **higher opportunity than background** space-time points sampled from where observers were active. Metric: **Boyce index** (or AUC-against-background), per-pair and pooled. You have presences and no true absences — this presence-only ranking eval is the principled choice and the reason the reframe avoids the negative trap.
- *Learned combiner (variant).* Instead of fixing `opportunity = flower × activity`, learn the combination: `g(flower(P,ℓ,t), activity(A,ℓ,t), [climate, AlphaEarth])` → score, fit **presence-background** (Maxent or logistic-with-background) by maximizing the likelihood of held-out presences against effort-matched background. This keeps the presence-only property (no pseudo-negatives) while letting the model weight the terms, fit nonlinear combinations, and absorb covariates the fixed product cannot. Do not substitute a plain classifier trained on sampled pseudo-negatives — that reintroduces the negative problem.

## Which probe (the flowering-curve source)

The probe is a small classifier on the frozen PPE embedding; what it's trained on determines what its output *means*, not just how accurate it is. Four options, simplest to most processed:

1. *Raw opportunistic (no PPE):* bin/smooth species P's iNaturalist flowering-observation dates. No model. Biased toward conspicuous peak bloom, sparse, exists only where observed. The no-PPE baseline — the thing to beat, not the signal.
2. *PPE + Pheno3M probe:* probe trained on iNaturalist phenophase labels. PPE generalizes across climate/space so the curve is smooth and fills unobserved space-time, but its calibration still reflects iNaturalist's flowering-overrepresentation. An "observed flowering" estimate.
3. *PPE + NPN probe:* probe trained on USA-NPN's scheduled yes/no (including genuine "not flowering"). NPN samples phenophase on a calendar, not by conspicuousness, so this recovers the *true* flowering probability, unbiased and not sparse. **Principled primary.** Practical middle ground: train the probe on the large Pheno3M set for discriminative power, then *recalibrate* outputs on NPN (Platt/isotonic).
Recommendation: run both 2 and 3 as arms (1 = no-PPE baseline). NPN is the unbiased reference — it has real absences. Pheno3M's peak over-representation is largely offset by balanced sampling when fitting the probe (resampling across phenophase and day-of-year), so it is a real arm, not a throwaway. "Better" = higher held-out co-observation ranking (Boyce / PR-AUC); report both. Caveat: NPN fixes the *readout*, but the iNaturalist-trained backbone can retain shoulder-season representational gaps, so this leans on the main-paper PPE↔NPN validation. The pollinator side stays opportunistic (GBIF KDE) regardless — scope strong claims to the plant side.

## Phenophase: this paper uses flowering

This is a plant–pollinator (flower-visitation) paper, so the phenophase is **flowering** for every pair. There is no per-pair phenophase to choose, because there is one interaction type — read PPE's flowering output and stop.

PPE does emit all four phenophases (vegetative, budding, flowering, fruiting), which matters only if you later extend beyond pollination (frugivory / seed-dispersal → fruiting; herbivory → vegetative/budding). Even then you would not research it per pair: GloBI records the interaction *type* as a structured field, so a fixed type→phenophase lookup assigns it at scale (visits-flowers-of → flowering, frugivore-of → fruiting, herbivore-of → vegetative), and a multi-type link-prediction model can skip the lookup entirely — feed all four phenophase curves (or the raw embedding) and let the classifier learn which drives which type. All future scope; for this paper, flowering.

## The overlap scalar

Plant flowering curve `f_P(ℓ,·)` from the probe; pollinator activity curve `a_A(ℓ,·)` from GBIF. The deciding consideration: `f_P` is an *absolute* per-day probability (NPN-calibrated) but `a_A` is only a *relative* distribution (GBIF has no effort correction). So the dimensionally-honest move is to **normalize both to distributions and use a temporal-niche-overlap metric** — scale-free, symmetric, citable, and it isolates synchrony from magnitude. Normalize `f̃(t) = f_P(ℓ,t)/Σ_t f_P(ℓ,t)`, similarly `ã`. Options:

- **Coefficient of overlapping Δ = Σ_t min(f̃, ã)** — shared area under the two normalized curves (Ridout & Linkie 2009). **Default**, because it's defined for *circular* activity data, which day-of-year is (Dec wraps to Jan). Estimate curves with circular (von Mises) kernels, not linear KDE, or the wrap-around overlap is lost — the easiest mistake to make. 0 = phenologically forbidden, 1 = perfectly synchronized; ties directly to the Olesen/Rafferty forbidden-link literature.
- *Schoener's D = 1 − ½ Σ_t |f̃ − ã|*; *Pianka's O = Σf̃ã/√(Σf̃²·Σã²)*; *Bhattacharyya = Σ√(f̃ã)*. Same synchrony family, all citable. (Note: for two normalized distributions, Schoener's D equals the coefficient of overlapping — the same number under two names.)
- *Co-active integral `Σ_t f_P·a_A`* (keep `f` absolute): magnitude/duration-aware (a pair co-active 60 days outranks one co-active 5 days), arguably truer to "opportunity," but mixes an absolute probability with a relative density — fragile until `a_A` is made absolute.
- *Encounter probability `1 − Π_t(1 − p·f_P·a_A)`*: the principled bridge to Galiana — their `1−(1−p)^N` with the co-occurrence *count* `N` replaced by phenology-weighted co-active intensity. Motivates the replacement arm theoretically. Same units caveat.
- *Raw embedding:* skip the scalar, feed PPE's plant phenology vector, let the model learn overlap against pollinator features. Most flexible, least interpretable, plant-only.

The scalar is per-(pair, location) because phenology shifts across space: aggregate over the pair's shared range (mean, or Galiana-style sum over co-occurring sites) for a per-pair feature; keep it as a function of `ℓ` for the opportunity surface and the spatial hold-out. Practically, include 2–3 of these as features and let the ablation decide which carries signal — but lead the *narrative* with the circular Δ, since it's the one that cleanly labels a pair forbidden vs overlapping.

### Worked toy example

Discretize the year into months (the real version uses 365 days). Plant flowering probability `f` from the probe and pollinator counts `a` from GBIF:

- `f` (plant, P(flowering)): Mar 0.1, Apr 0.5, May 0.9, Jun 0.6, Jul 0.2, else 0. (Sum 2.3.)
- `a` (bee, counts): May 1, Jun 4, Jul 8, Aug 6, Sep 3, Oct 1, else 0. (Sum 23.)

Normalize each (`f̃ = f/2.3`, `ã = a/23`), then take the month-by-month minimum:

| Month | f̃ | ã | min(f̃, ã) |
|---|---|---|---|
| Mar | 0.04 | 0 | 0 |
| Apr | 0.22 | 0 | 0 |
| May | 0.39 | 0.04 | 0.04 |
| Jun | 0.26 | 0.17 | 0.17 |
| Jul | 0.09 | 0.35 | 0.09 |
| Aug | 0 | 0.26 | 0 |
| Sep | 0 | 0.13 | 0 |
| Oct | 0 | 0.04 | 0 |

`Δ = 0.04 + 0.17 + 0.09 = 0.30` — the shared area under the two normalized curves. They overlap in May–Jul, but the plant peaks in May and the bee in July, so only about a third of their seasons coincide. Schoener's D gives the same 0.30.
