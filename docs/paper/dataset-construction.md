# Dataset Construction: A CONUS Plant–Pollinator Interaction Network from GloBI

This document specifies, step by step, how the interaction network is built. It is written to be
cited in the Methods section. Each step states the operation, the fields it uses, and the published
precedent for it.

---

## 1. Overview

We construct a bipartite plant–pollinator interaction network for the continental United States from
the Global Biotic Interactions database (GloBI; Poelen, Simons & Mungall 2014, *Ecological
Informatics* 24:148–159, doi:10.1016/j.ecoinf.2014.08.005), an aggregator that harmonises
species-interaction records from museum collections, published literature, citizen-science platforms
and government surveys into a single interpreted schema.

The pipeline takes the interpreted interaction table and produces (i) an edge list of unique
plant–pollinator pairs with provenance and temporal metadata, (ii) node tables for both sides, and
(iii) a yield ledger recording how many records survive each step.

---

## 2. Source data and versioning

The input is the GloBI interpreted interaction table (`interactions.csv.gz`, 92 fields). We pin a
single dated snapshot and record its SHA-256 digest, byte size and row count in the dataset card.

Version pinning follows GloBI's own recommended practice. Elliott, Poelen & Fortes (2020,
*Ecological Informatics* 57:101077, doi:10.1016/j.ecoinf.2020.101077) document that between 5% and
70% of biodiversity-data provider URLs become unresponsive over time, and introduce content-based
addressing so that an analysis can name the exact bytes it consumed. Reporting a digest rather than a
download date makes the dataset reconstructible.

---

## 3. Step-by-step construction

### Step 1 — Restrict to the study region

Retain records whose `decimalLatitude` and `decimalLongitude` fall within the CONUS bounding box
(24.0°–49.5° N, 125.0°–66.0° W).

Records lacking coordinates are retained in a separate *metaweb* edge list and excluded only from the
spatially-modelled set. Regional subsetting of GloBI by coordinate box is standard; Reji Chacko et al.
(2025, *Scientific Data* 12:1164, doi:10.1038/s41597-025-05487-7) apply the same geographic filtering
step when deriving a regional interaction network from GloBI.

### Step 2 — Select interaction types

Interaction types are Relations Ontology (RO) terms. We retain four and assign each to a tier:

| tier | term | RO identifier |
|---|---|---|
| A | `visitsFlowersOf` | RO_0002622 |
| A | `pollinates` | RO_0002455 |
| B | `visits` | RO_0002618 |
| B | `interactsWith` | RO_0002437 |

Tier A terms denote flower visitation explicitly. Tier B terms are retained but flagged: `visits`
carries no floral semantics, and `interactsWith` is the root of the RO interaction hierarchy. All
results are reported for Tier A and for Tier A+B, so the effect of the weaker terms is measurable
rather than assumed.

Terms denoting other ecological relations — `hasHost`, `parasiteOf`, `preysOn`, `eats`, `symbiontOf`,
`hasArbuscularMycorrhizalHost`, `adjacentTo`, `coOccursWith`, `ecologicallyRelatedTo` — are excluded.
Whitelisting interaction terms is the established approach: Reji Chacko et al. (2025) enumerate the
permitted `interactionTypeName` values for their network, and Baiotto et al. (2026, bioRxiv,
doi:10.64898/2026.03.30.715389) reduce GloBI to two terms, using `visitsFlowersOf` for pollination and
`hasHost` for larval host associations. Noori et al. (2026, *Scientific Data*,
doi:10.1038/s41597-026-06970-5) retain `interactsWith` in their curated bee–plant dataset, where it
accounts for a large share of records — motivating our decision to carry it as a separate tier.

### Step 3 — Assign trophic roles from taxonomic kingdom

Each side of a record is assigned a role from `sourceTaxonKingdomName` / `targetTaxonKingdomName`:
plant ← {`Plantae`, `Archaeplastida`, `Viridiplantae`}; animal ← {`Animalia`, `Metazoa`}. Where the
kingdom field is absent or uninformative (`Eukaryota`, null), we fall back in order to
`*TaxonPhylumName`, `*TaxonClassName`, and finally to membership in the study species lists, recording
which rule fired in a `role_source` field.

Role assignment from the taxonomic hierarchy rather than from name lists is necessary because
genus-level homonyms span kingdoms — *Ammophila* denotes both a grass genus (Poaceae) and a wasp genus
(Sphecidae). Dorey et al. (2023, *Scientific Data* 10:747, doi:10.1038/s41597-023-02626-w) implement
the same principle in `BeeBDC::harmoniseR()`, which declines to resolve ambiguous homonyms rather than
assigning them by name match.

### Step 4 — Orient each record

GloBI resolves inverse predicates during interpretation, so records of the retained types are
canonically directed visitor → subject. We therefore assign `plant := target` and
`pollinator := source`, and discard records whose kingdom-derived roles contradict this direction,
logging them with their reason.

Normalising interaction direction is a required step when deriving an undirected species-pair list
from GloBI. Noori et al. (2026) normalise orientation by placing bees consistently on one side, and
Lee, DiRenzo, Diao & Seltmann (2026, *Ecological Applications* 36:e70221, doi:10.1002/eap.70221)
describe standardising "columns where bee and plant species names appeared, given that they could be
in either the target or source columns."

### Step 5 — Resolve taxon names to identifiers

Node identity is taken from the `sourceTaxonIds` / `targetTaxonIds` fields, which carry
pipe-delimited cross-references to external authorities (GBIF, Catalogue of Life, World Flora Online,
ITIS, NCBI, EOL, Wikidata). We prefer the GBIF backbone key (doi:10.15468/39omei), falling back to
Catalogue of Life, World Flora Online, then ITIS. Sub-specific ranks are rolled up to species via
`*TaxonSpeciesName`.

GloBI performs name alignment upstream with `nomer`, so these identifiers are already reconciled
against the source taxonomies; consuming them is preferable to re-resolving name strings downstream.
Where identifiers are absent, standard resolvers apply: TNRS (Boyle et al. 2013, *BMC Bioinformatics*
14:16, doi:10.1186/1471-2105-14-16), World Flora Online (Kindt 2020, *Applications in Plant Sciences*
8:e11388), and the World Checklist of Vascular Plants (Govaerts et al. 2021, *Scientific Data* 8:215;
`rWCVP`, Brown et al. 2023, *New Phytologist* 240:1355).

### Step 6 — Set taxonomic resolution

Nodes are retained at species and genus rank. A genus-rank taxon is a node in its own right and is
never merged into a constituent species. Records identified only to family or coarser are excluded.
Every headline result is reported at both species-only and species-plus-genus resolution.

Retaining genus-level nodes follows Noori et al. (2026), whose curated bee–plant dataset keeps
genus-rank plant taxa alongside species-rank ones. The dual reporting is motivated by two findings:
Renaud, Baudry & Bessa-Gomes (2020, *Ecology and Evolution* 10:3248, doi:10.1002/ece3.6060) recomputed
nine network indices at species, genus and family resolution across 41 plant–pollinator networks and
found the *rank order* of indices strongly conserved between species and genus level, while
Hemprich-Bennett et al. (2021, *Ecology* 102:e03256, doi:10.1002/ecy.3256) show that absolute metric
values are sensitive to node resolution. Discarding coarsely-identified records is itself a source of
bias (Jordano 2016, *PLoS Biology* 14:e1002559, doi:10.1371/journal.pbio.1002559).

### Step 7 — Deduplicate records

Aggregated interaction databases contain the same underlying observation multiple times when it
reaches the aggregator through more than one provider. We deduplicate at record level in four passes,
applying the first key available:

1. exact duplicate rows;
2. `(referenceCitation, sourceTaxonId, targetTaxonId, interactionTypeId)`;
3. `(sourceCatalogNumber, sourceInstitutionCode, sourceTaxonId, targetTaxonId)`;
4. `(sourceTaxonId, targetTaxonId, interactionTypeId, decimalLatitude, decimalLongitude, eventDate)`.

For citizen-science records `referenceCitation` resolves to the individual observation, giving an
exact observation identity; for specimen records the catalogue triple performs the same function.

Deduplication keys of this form are standard. Seltmann, Poelen & the GloBI Community (2025, *Global
Bee Interaction Data* v7.0, Zenodo doi:10.5281/zenodo.17957582) retain "unique records based on the
interaction description and source citation", reducing 3,030,355 records to 1,868,619, and attribute
the duplication to multiple providers sharing the same information. Lee et al. (2026) deduplicate on
"unique code combinations associated with museum specimen collection and catalog numbers". Dorey et
al. (2023) implement four complementary rule sets in `BeeBDC::dupeSummary()`, keyed on catalogue and
institution codes, occurrence identifiers, and coordinate–date–name combinations.

### Step 8 — Filter by life stage

Pollinator-side records whose `sourceLifeStageName` denotes a pre-adult stage (larva, caterpillar,
nymph, pupa, egg) are excluded; records with no life-stage annotation are retained and flagged.
Immature insects recorded on plants represent larval host associations rather than flower visitation,
a distinction Baiotto et al. (2026) encode by assigning larval host records to a separate interaction
class.

### Step 9 — Aggregate to unique interactions

Surviving records are grouped by `(plant_id, pollinator_id)`. For each pair we retain the number of
supporting records, the number of distinct observations, the number and identity of contributing
source datasets, the interaction types, the tier, and the first and last observation years.

Retaining per-edge provenance supports two analyses used later: leave-one-source-out validation, and
stratification of results by evidential support. Provenance-aware reporting is recommended for
aggregated interaction data by Ollerton, Taliga, Salim, Poelen & Drucker (2025, *Journal of
Pollination Ecology* 38:151–160, doi:10.26786/1920-7603(2025)844), who note that measures of data
quality — the evidence by which an animal is determined to interact with a plant — are rarely reported.

### Step 10 — Intersect with feature coverage

The modelled network is the subgraph whose species carry the spatial and phenological features used by
the models. The full edge list from Step 9 is released alongside it, since the interaction network is
useful independently of feature availability.

---

## 4. Output schema

`edges.parquet` — one row per unique interacting pair:

| field | type | description |
|---|---|---|
| `plant`, `pollinator` | string | resolved taxon labels |
| `plant_id`, `pollinator_id` | string | canonical identifier, e.g. `GBIF:3034002` |
| `plant_rank`, `pollinator_rank` | string | `species` or `genus` |
| `plant_family`, `pollinator_family`, `pollinator_order` | string | higher taxonomy |
| `tier` | string | `A` or `B` |
| `types` | string | contributing interaction types |
| `n_records` | int | supporting records after deduplication |
| `n_observations` | int | distinct observation identities |
| `n_sources` | int | contributing source datasets |
| `sources` | string | source dataset identifiers |
| `n_inat` | int | records attributed to iNaturalist |
| `first_year`, `last_year` | int | observation year range |
| `role_source` | string | rule that assigned trophic roles |

`nodes_plants.parquet`, `nodes_pollinators.parquet` — identifier, label, rank, higher taxonomy,
kingdom, and feature-coverage flags.

`dataset_card.md` — snapshot digest, per-step yield ledger, and summary network statistics.

---

## 5. Yield ledger

Recorded during execution; reported in the paper as a data-flow table.

| step | operation | records in | records out | edges | plants | pollinators |
|---|---|---|---|---|---|---|
| 1 | region filter | | | | | |
| 2 | interaction-type selection | | | | | |
| 3 | role assignment | | | | | |
| 4 | orientation | | | | | |
| 5 | identifier resolution | | | | | |
| 6 | rank policy | | | | | |
| 7 | deduplication | | | | | |
| 8 | life-stage filter | | | | | |
| 9 | aggregation | | | | | |
| 10 | feature intersection | | | | | |

---

## 6. Verification

### Per-step exploratory checks

1. Record density mapped over the study region.
2. Record counts by interaction type and tier; overlap between tiers.
3. Rate of role assignment by rule; genera resolved by fallback inspected for cross-kingdom homonyms.
4. Orientation-violating records counted and inspected.
5. Identifier coverage by authority; names mapping to multiple identifiers and identifiers mapping to
   multiple names.
6. Node counts and degree distributions at species and genus rank.
7. Deduplication rate by source dataset; distribution of records per observation.
8. Life-stage value counts.
9. Degree distributions, connectance, source counts per edge, temporal span.
10. Composition of the loss at feature intersection, tested for taxonomic bias.

### Automated tests

**Structure.** No self-pairs; edge list unique on `(plant_id, pollinator_id)`; referential integrity
between edges and node tables; only species and genus ranks present.

**Semantics.** Every plant identifier resolves to a plant kingdom and every pollinator identifier to an
animal kingdom; no orientation violations survive; Tier A edges carry only Tier A interaction types;
plant and pollinator node sets are disjoint.

**Counts.** `n_records ≥ 1`; `n_sources ≥ 1`; `n_inat ≤ n_records`; `n_observations ≤ n_records`;
`first_year ≤ last_year` and both within plausible bounds; summed per-edge record counts do not exceed
records surviving Step 8.

**Reproducibility.** Deduplication is idempotent; two executions from the pinned snapshot produce
byte-identical outputs.
