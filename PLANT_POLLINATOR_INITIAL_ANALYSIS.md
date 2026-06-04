# Part 1 — Empirical Phenology Overlap (Data-Only Baseline)

## Motivation

Most plant–pollinator interaction work in the literature uses **spatial
co-occurrence** as the proxy for interaction: "if pollinator P and plant Q are
observed in the same place, they probably interact." Time is typically dropped
or coarsened to "year" / "season." That bakes in an implicit assumption that
species in the same place at *different times* still meaningfully interact —
which is exactly wrong for many pollination systems where phenological
matching (or mismatch) is the central question.

Part 1 measures **temporal overlap** between observed plant flowering and
observed pollinator activity, restricted to shared spatial bins, using only
observation data — no model. This gives us the empirical baseline for
"how much does time matter, before any model is involved?"

## Inputs

- **Plant phenology data**: the same observation-level dataset that
  Phenofield is trained on. Required per row: plant species, lat/lon, date,
  and a flowering annotation. Used here directly, no model in the loop.
- **GloBI**: bulk interactions dump
  (`globalbioticinteractions.org → snapshot/target/data/tsv/interactions.tsv.gz`).
  Used **only** for the plant→pollinator taxonomy edge list. GloBI's own
  lat/lon/date columns are sparse and ignored.
- **iNaturalist Open Data**: free S3 mirror
  (`inaturalist-open-data.s3.amazonaws.com/{observations,taxa}.csv.gz`).
  Used for pollinator activity records — no account needed.

## The 5 steps

### Step 1 — Plant flowering events

Filter the plant phenology data to flowering records only, CONUS extent,
post-2013-01-01. The output is a flat table of `(species, lat, lon, date)`
for every observed flowering event.

### Step 2 — Plant→pollinator edge list

From the GloBI dump, keep rows whose `interaction_type` is one of:
`pollinates`, `visits`, `visitsFlowersOf`, `visitedBy`, `flowersVisitedBy`,
`hasFlowerVisitor`, `pollinatedBy`. Restrict the plant side to species in
Step 1. Normalise directionality so plant and pollinator columns are
consistently the plant and the pollinator (the `*By` interaction types invert
source/target). Output: unique `(plant_species, pollinator_species)` pairs.

### Step 3 — Pollinator observations

From iNat Open Data, filter to:

| filter | value | why |
|---|---|---|
| `observed_on` | ≥ 2013-01-01 | match plant window |
| `latitude / longitude` | CONUS bounding box | scope |
| `positional_accuracy` | < 1 km | drop obscured coordinates |
| `quality_grade` | `research` | community-verified IDs only |
| taxonomy | Hymenoptera, Lepidoptera, Diptera, Coleoptera, Trochilidae | the four pollinator insect orders + hummingbirds |

The taxonomy filter is a join through `taxa.csv.gz` on `taxon_id` ancestry.
Output: `(pollinator_species, lat, lon, date)` per observation.

### Step 4 — Per-(species, spatial bin) timing distributions

For both plants (from Step 1) and pollinators (from Step 3), build a
distribution of observed timing across week-of-year, within each spatial
bin and per species. Two representations are interchangeable: a normalised
histogram, or a circular KDE (week-of-year wraps around — use a circular
Gaussian smooth with bandwidth ≈ 2 weeks).

**Spatial bin size is TBD** and should be set after EDA on the plant
phenology data (look at the distribution of distinct species per candidate
bin size — pick something where most bins have ≥ 10 species). Plausible
range to evaluate: 0.25° to 2°. Smaller bins → tighter spatial control but
more sparse curves; larger bins → smoother curves but more spatial
heterogeneity averaged out.

### Step 5 — Overlap coefficient per edge per bin

For each edge from Step 2, in each spatial bin where both species have a
density curve, compute the overlap coefficient:

$$\text{overlap}(p, q, \text{bin}) = \sum_{w=1}^{52} \min\bigl(\text{plant}_p(w), \text{pollinator}_q(w)\bigr)$$

This is in [0, 1]: 0 = no temporal co-occurrence; 1 = identical timing.
Schoener's D is equivalent up to a constant.

Drop bins where either side has fewer than ~10 observations (the threshold is
also part of the EDA — sparse bins make KDEs spiky and overlap unreliable).

## How to show "timing matters"

The point of Part 1 is to demonstrate that incorporating temporal information
changes the picture compared to spatial-only co-occurrence. A few metrics that
make this concrete:

1. **Distribution of temporal overlap scores across all edges.**
   If most known interactions have overlap near 1.0, timing barely
   discriminates — the spatial-only assumption is fine. If overlap is broadly
   spread (or bimodal), timing matters a lot. A median overlap below ~0.6
   already implies many "co-located but phenologically separated" pairs.

2. **Fraction of edges with low temporal overlap despite spatial
   co-occurrence.** Count `(plant, pollinator)` edges that share a spatial
   bin but have temporal overlap < 0.3 (or some threshold). These are the
   pairs that a spatial-only model would predict as plausible interactions
   but that observations say are timing-mismatched. Express as a percentage of
   spatially co-occurring edges. **If this fraction is meaningfully large
   (say, > 15-20%), that's the headline result that justifies bringing
   temporal modelling in at all.**

3. **Per-edge spatial-vs-temporal divergence.** For each edge, plot
   spatial-overlap (do they share bins?) against temporal-overlap (within
   shared bins, do their curves align?). Edges in the high-spatial /
   low-temporal quadrant are the cases spatial-only models get wrong. A
   regression or correlation here quantifies how much new information time
   adds on top of space.

4. **Phenology-aware vs phenology-naive ranking of likely pollinators per
   plant.** For each plant species, rank candidate pollinators two ways:
   (a) by spatial co-occurrence count, (b) by spatial × temporal-overlap.
   Compute rank correlation between the two. Low correlation means timing
   re-orders the top candidates — i.e., a real signal that spatial-only
   methods miss.

5. **Sensitivity to spatial bin size.** Recompute the headline metric
   (#2 above) across the candidate bin sizes from Step 4 EDA. If the
   "fraction of timing-mismatched co-occurrences" stays roughly constant
   across bin sizes, the temporal signal is robust. If it vanishes at large
   bins, the temporal signal only exists at fine grain — also informative.

## Caveats

- **Observation effort drives both signals.** Per-species, per-bin
  normalisation absorbs effort *within* a species but cross-species
  comparisons inherit observer-density bias.
- **No verified visits.** Temporal overlap is a necessary but not sufficient
  condition for actual pollination. We are testing whether timing matters
  for *prediction*, not measuring true interactions.
- **iNat plant phenology coverage is uneven.** Some species/regions have
  rich curves; others are sparse. Sparse-curve edges contribute noise to
  every metric above — flag and report what fraction of edges are
  coverage-limited.

## Part 2 preview

Part 2 replaces the **plant timing distribution** in Step 4 with the
Phenofield model's predicted flowering curve. Steps 1, 2, 3, and the
pollinator side of Step 4 are reused. The diff between Part 1 and Part 2
metrics tells us whether Phenofield adds usable temporal signal — most
informatively, in the sparse-coverage regions Part 1 flagged.
