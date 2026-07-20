# Idea 6 — PPE for Plant–Pollinator Interaction Opportunity

PPE's continuous phenology field predicts plant–pollinator interaction *opportunity* across space and time — including at locations and dates with no records — validated presence-only against held-out co-observations, beating GDD / observation-date / EO baselines. The contribution is the **input**: a validated, spatially-continuous, temporally-resolved phenological opportunity term. The claim is *opportunity*, not interaction strength.

This is the modular version of the project notes. Suggested reading order:

1. **[01-overview.md](01-overview.md)** — the one-line contribution, the approach (and why the forbidden-link alternative is suboptimal), and where the work sits in the literature.
2. **[02-ppe-integration.md](02-ppe-integration.md)** — how PPE is used (frozen flowering field), which probe to use, and how the phenological-overlap scalar is computed.
3. **[03-experiments-and-ablations.md](03-experiments-and-ablations.md)** — the cheap go/no-go test, the computed-surface ablations, and the two model families (link prediction; SDM-overlap) with their table skeletons.
4. **[04-data.md](04-data.md)** — the data pipeline and the concrete inventory for initial experiments, including the gating coverage pre-check.
5. **[05-scope-and-limitations.md](05-scope-and-limitations.md)** — what the claim is and is not; the honest caveats.
6. **[06-extensions-and-next-steps.md](06-extensions-and-next-steps.md)** — network structure, the degree-distribution bridge, and the SINR-style spatiotemporal extension with its validation story.
7. **[07-reading-list.md](07-reading-list.md)** — curated reading path.

**[models/](models/README.md)** — simple, self-contained write-ups of the initial analysis and the two predictive approaches, each with intuition, steps, math, and pseudocode.

**First action:** the GBIF pollinator coverage pre-check (see [04-data.md](04-data.md)). It decides taxonomic resolution, which groups are usable, and whether the project leans plant-side — so it precedes building anything else.
