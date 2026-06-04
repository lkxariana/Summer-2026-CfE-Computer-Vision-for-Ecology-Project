> Part of [Idea 6](README.md). Prev: [PPE integration](02-ppe-integration.md) · Next: [Data](04-data.md).

# Experiments and ablations

## Go/no-go (cheap, before the full pipeline)

Pick a handful of well-documented CONUS pairs. Compute `opportunity` from PPE and from an observation-date baseline. Check held-out co-observations of each pair against both via Boyce / background-AUC, including at held-out locations. If PPE does not rank held-out positives better — especially at unobserved locations, where the date baseline is blind — the premise is weak; stop. If it does, that is Figure 1.

## Computed opportunity surface — ablations

These ablate the simplest instantiation — the computed `flower × activity` opportunity surface (the version the go/no-go tests and that Approach B generalizes). Only #2 is the headline.

1. *Flowering source:* PPE-NPN-probe vs GDD vs observation-date window vs AlphaEarth-probe. PPE should rank held-out positives best.
2. ***Spatial hold-out (the key result):*** spatial block CV — predict opportunity at locations where the pair was never observed. PPE's continuous field has signal there; date-window baselines have none. This is PPE's unique value and the reason the paper exists; it must clear the bar set by Poisot's latent-variable approach.
3. *Temporal hold-out:* held-out years.
4. *Term ablation:* `flower × activity` vs flower-only vs activity-only — both terms should matter.
5. *Pollinator-estimator sensitivity:* bandwidth / quantile / min-records threshold — robustness check, not a contribution.

## Experiments and ablation tables

Two model families (see [extensions-and-next-steps](06-extensions-and-next-steps.md) for the framing): **Approach A** = link prediction (predicts the *realized* interaction; PPE enters as a covariate); **Approach B** = SDM-overlap (predicts *availability* fields; opportunity = field overlap). They predict different objects, so they are complementary, not rival — A is the initial paper, B the extension, Table 6 nests them (Poisot two-stage). The one shared head-to-head is ranking held-out co-observations (Boyce / AUC-against-background). Note on JSDM: a JSDM (e.g. HMSC) is one *joint* way to build B's availability fields — use it for the fields only, never read interactions off its residual species-covariance (Blanchet).

### Approach A — link prediction

Model: shared classifier (shallow NN / RF / logistic) on species-pair features → P(interaction). Positives class-balanced to ~25% (Poisot). Primary metrics PR-AUC and ROC-AUC; Boyce / AUC-against-background reported as the negative-free check.

*Table A1 — feature ablation* (random pair hold-out):

| Arm | Features (cumulative) | Isolates |
|---|---|---|
| A0 | co-occurrence embedding (prob-PCA of co-occ matrix) | Poisot baseline |
| A1 | + traits / phylogeny (if available) | standard covariates |
| A2 | + Galiana `N` (spatial co-occurrence frequency) | strongest co-occ-only competitor |
| A3 | + `Δ` (PPE phenological overlap scalar) | **headline — temporal co-occurrence** |
| A4 | + raw PPE phenology embedding | max PPE information |
| A5 | replace `N` with model-based co-occurrence (encounter-prob) | replacement variant |

Decisive comparison: A3 vs A2.

*Table A2 — generalization* (key arms × hold-out type; metric per cell PR-AUC or Boyce):

| Hold-out | A0 | A2 | A3 / A4 |
|---|---|---|---|
| Random pairs | baseline | — | — |
| Spatial block (unobserved locations) | — | collapses (`N` undefined) | holds (continuous field) |
| Temporal (held-out years) | — | — | — |
| Cold-start species (few/no interaction records) | fails (no embedding) | weak | holds (PPE field exists) |

*Table A3 — diagnostics.* Probe for `Δ`: NPN and Pheno3M (both PPE arms) vs raw-KDE (no-PPE baseline) — report which ranks held-out interactions higher. Scalar: circular Δ vs Schoener's D vs co-active integral vs encounter-prob (robustness). Divergence analysis (interpretable result, not a table): among pairs with high spatial `N` but low `Δ`, count the spatial-only false positives PPE corrects.

### Approach B — SDM-overlap

Plant availability field = frozen PPE (probe). Pollinator availability field = temporal SDM on GBIF. Opportunity = spatiotemporal overlap of the two fields. The ablation is over how the fields are built and combined, not over interaction features. (A JSDM / HMSC is one joint option for building the fields — the B4 "joint" spirit — used for availability, not its covariance.)

*Table B1 — pollinator availability field* (presence-only SDM metrics: AUC-against-background, Boyce; plant field validated separately vs NPN gold standard):

| Pollinator field | Isolates |
|---|---|
| B0 — raw KDE of GBIF dates | no-SDM baseline |
| B1 — coordinates-only INR (SINR-style) | learned, env-free |
| B2 — + climate covariates | proper temporal SDM |
| B3 — + PPE representation as input | **representation transfer (vs B2)** |
| B4 — joint SINR across pollinators | joint vs stacked |

*Table B2 — opportunity field* (overlap of the two fields, vs held-out co-observations):

| Hold-out | metric |
|---|---|
| Random | Boyce / AUC-vs-bg |
| Spatial block | Boyce / AUC-vs-bg |
| Cold-start species | Boyce / AUC-vs-bg |

Sub-rows: overlap operator (Δ / Schoener / encounter-prob); joint vs stacked.

### Table 6 — combined (nested)

Approach A with vs without the Approach-B opportunity field as an input feature/prior. Tests the two-stage (availability → realized link) story; the cleanest single-paper result if it helps.
