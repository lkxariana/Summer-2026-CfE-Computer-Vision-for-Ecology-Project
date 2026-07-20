> Part of [Idea 6](README.md). Prev: [Experiments and ablations](03-experiments-and-ablations.md) · Next: [Scope and limitations](05-scope-and-limitations.md).

# Data

## Data pipeline

- **Interactions (positives):** GloBI bee–plant subset → filter to CONUS, post-2013 (PPE training window), iNaturalist-sourced "visits flowers of" / "visited by" records with coordinates + dates.
- **Plant flowering:** frozen PPE backbone + NPN probe, queried on a CONUS space-time grid.
- **Pollinator activity:** GBIF occurrence dates → sparse-data phenology estimator → regional activity curves for well-sampled taxa (bees, butterflies, hoverflies).
- **Background (for presence-only eval):** space-time points sampled from observer-active locations/dates (target-group background), so the eval controls for effort.

## Data inventory for initial experiments

A and B draw on nearly the same raw sources; the difference is mostly how the data is reshaped.

| Data | Source | Used by | Scope / filter | Status |
|---|---|---|---|---|
| Interaction records (positives) | GloBI — bee/butterfly–plant "visits flowers of" | A (positives), B (held-out co-obs) | CONUS, post-2013, georef + dated, flower-visitation | get |
| Pollinator occurrences | GBIF | A (activity curve), B (pollinator SDM); coverage pre-check | CONUS, post-2013, georef + dated, best-sampled taxa first | get — **first** |
| Plant flowering field | PPE (frozen) + NPN probe | A (`Δ`), B (plant field) | CONUS, PPE species | mostly in hand |
| USA-NPN status data | USA-NPN | probe training/recalibration + plant-field validation | CONUS, focal plant taxa | get |
| Occurrences for co-occ matrix | GBIF / iNat | A only (A0 embedding, A2 Galiana `N`) | focal taxa, chosen spatial grid | get |
| PRISM window + AlphaEarth | PRISM / AlphaEarth | PPE inputs; B pollinator-SDM covariates (B2/B3) | at occurrence points/times | in hand |
| Target-group background | derived from occurrences | B presence-only SDM + AUC-vs-bg | observer-active points/times | derive |
| Traits / phylogeny | TRY, published trees | A1 only | focal taxa | defer (optional) |
| Expert pollination network | Web of Life / a published study | validation spot-check (interaction layer) | one clean CONUS network | optional |

**Pollinator coverage pre-check — do this first.** Pull GBIF occurrences for the candidate pollinator taxa and count how many *species* have enough dated, georeferenced records for a stable activity curve. This one number decides feasibility, the taxonomic resolution (species vs genus vs family), and which groups are usable (bees, butterflies data-rich; hoverflies thinner). If coverage is thin, fall back to plant-side-heavy analysis or coarser taxa. Everything downstream is shaped by this, so it is the first pull and the first analysis.

**Scope decisions to lock up front** (they propagate through every table): geographic = CONUS (matches Pheno3M); temporal = post-2013 (PPE window, iNat research-grade era); taxonomic = start with best-sampled groups; spatial-unit size for the co-occurrence matrix — a genuine methodological choice (Blanchet Arg. 4; Galiana both flag scale-dependence), so pick deliberately and test sensitivity.

**Two data-handling traps.** (1) Taxonomic reconciliation: names must match across GloBI, GBIF, the PPE species list, NPN, and any trait source — unglamorous but where these projects stall; budget for it. (2) Shared provenance / leakage: GloBI interactions, GBIF pollinator occurrences, and PPE's training data are *all* largely iNaturalist-sourced. Keep PPE's training observations out of the interaction test set, and use NPN — not held-out iNat — as the unbiased reference, or "validation" just re-measures shared observation bias. This is why NPN is load-bearing and worth the separate pull.

**Minimal set to get moving:** GloBI interactions + GBIF pollinator occurrences + the PPE/NPN plant field + the gridded occurrence matrix for A's baseline (PRISM/AlphaEarth already in hand). Enough to run the go/no-go and populate Tables A1, A2, and a first B1/B2. Traits/phylogeny (A1) and the expert network (validation spot-check) are genuine deferrals — they sharpen the work but are not needed for the first result.
