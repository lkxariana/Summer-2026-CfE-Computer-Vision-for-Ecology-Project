# Edge Generation Protocol v2

**Status: DRAFT — awaiting Dan's approval. Nothing is executed until approved.**
Supersedes `scripts/rebuild_edges.py` (v1, 62,832 edges). Every rule below was verified against our
actual file, not against documentation or a live snapshot — verification notes are inline.

---

## 0. Inputs and pinning

| input | path | pin |
|---|---|---|
| GloBI interactions | `/scratch/ariana.l/CfE2026CVforEcology/rawpollinatordata/interactions.csv.gz` | record SHA-256 + row count + column count in the dataset card |
| Plant feature coverage | PPE / PhenoField species list | 6,348 species |
| Pollinator feature coverage | GBIF `pollinator_observations_v2.csv` | 24,939 species |

The file has 92 columns. Column semantics differ between the bulk CSV (camelCase) and the API docs
(snake_case) — we read our own header.

---

## 1. Spatial filter

**Rule.** Keep rows with `decimalLatitude` ∈ [24.0, 49.5] and `decimalLongitude` ∈ [−125.0, −66.0].
**Verified yield:** 5,333,308 CONUS records of 24.5M.
**Note.** Rows with null coordinates are dropped here. Quantify them — a record with no location is
still evidence of an interaction, and we may want them for the metaweb even if not for the spatial features.

---

## 2. Interaction-type tiering

**Rule.** Assign a `tier`, do not discard Tier B.

| tier | types | RO id | CONUS records (verified) |
|---|---|---|---|
| **A** | `visitsFlowersOf` | RO_0002622 | 575,792 |
| **A** | `pollinates` | RO_0002455 | 65,334 |
| **B** | `visits` | RO_0002618 | 71,476 |
| **B** | `interactsWith` | RO_0002437 | 1,821,085 |
| — excluded | `hasHost`, `adjacentTo`, `coOccursWith`, `symbiontOf`, `eats`, `preysOn`, `parasiteOf`, `hasArbuscularMycorrhizalHost`, `ecologicallyRelatedTo` | | |

**Rationale.** Tier A is unambiguous flower visitation. `visits` has no floral semantics;
`interactsWith` is the *root* of the RO interaction hierarchy, i.e. semantically empty — but it is
1.8M CONUS records and ~46% of bee–plant records in comparable published work, so discarding it
silently is also a decision. Tiering lets us **ablate A vs A+B** and report sensitivity.
Exclusions are on semantics: `hasHost` is larval/parasite association, `adjacentTo` is topology,
`coOccursWith` is spatial not interactional.

---

## 3. Role assignment by kingdom (replaces the membership heuristic)

**Rule.** Assign each side a role from `sourceTaxonKingdomName` / `targetTaxonKingdomName`:
- PLANT ← {`Plantae`, `Archaeplastida`, `Viridiplantae`}
- ANIMAL ← {`Animalia`, `Metazoa`}
- ambiguous ← {`Eukaryota`, null} → fall back to `*TaxonPhylumName`/`*TaxonClassName`, then to
  species-list membership, and flag `role_source` ∈ {kingdom, phylum, membership}.

**Verified vocabulary** (CONUS Tier A sample, target side): Plantae 402,392 · null 112,021 (19%) ·
Archaeplastida 34,714 · Viridiplantae 18,800 · Eukaryota 4,212 · **Animalia 4,103 (role violations)**.
Source side: Animalia 387,973 · Metazoa 152,112 · Eukaryota 28,172 · null 7,060 · **Plantae 1,143**.

**Rationale.** This fixes a hole membership could never close: *Ammophila* is both a grass genus
(Poaceae) and a wasp genus (Sphecidae). Our v1 "F∩P overlap = 0" check gave no protection — it only
meant one homonym silently landed on one side.

---

## 4. Orientation from interaction type; drop violations

**Rule.** For all four kept types GloBI canonicalises to **visitor → plant**, so:
`plant := target`, `pollinator := source`. Rows where the roles contradict this
(source is PLANT and target is ANIMAL) are **logged and dropped**, not swapped.

**Verified.** The interpreted product contains **zero** `flowersVisitedBy` (RO_0002623),
`pollinatedBy` (RO_0002456) or `visitedBy` (RO_0002619) rows — GloBI already collapses inverse
predicates. So a reversed row is genuine upstream error, not an alternative encoding.

**Change from v1.** v1 *swapped* by membership. That silently rescued error records. Membership is
demoted to an assertion that raises if it disagrees with the kingdom-derived role.

---

## 5. Name resolution by identifier

**Rule.** Parse `sourceTaxonIds` / `targetTaxonIds` (pipe-delimited). Prefer `GBIF:`, then
`COL:` → `WFO:` → `ITIS:`. Use the id as the canonical node key; keep `*TaxonName` as a label.
Roll subspecies / variety / form up to species via `*TaxonSpeciesName`.
**Drop the regex binomial fallback entirely.**

**Verified.** `sourceTaxonIds` and `targetTaxonIds` are **100% populated** on flower-visitation rows.
Example: `COL:79QLB | EOL:596911 | GBIF:3034002 | ITIS:18158 | NCBI:46945 | WD:Q584469 | WFO:0001071258`.
All 245,495 genus-rank targets carry an id — **v1's 207,319 genus-only losses were a join failure,
not a resolution failure.**

---

## 6. Rank policy — keep genus nodes (approved)

**Rule.** Emit `plant_rank` / `pollinator_rank` ∈ {species, genus}. A genus node is a distinct node,
never merged into a member species. Records coarser than genus (family, tribe, order) are dropped.

**Rationale / caveat.** Rank-order conclusions are robust to node resolution (Renaud et al. 2020,
*Ecol. Evol.* 10:3248) but absolute connectance/modularity are not (Hemprich-Bennett et al. 2021,
*Ecology* 102:e03256). **Therefore: report every headline result at both species-only and
species+genus resolution.**

---

## 7. Deduplication

**Rule.** Applied in order, at record level, before aggregation:
1. exact-row duplicate removal;
2. collapse on `(referenceCitation, sourceTaxonId, targetTaxonId, interactionTypeId)`;
3. else `(sourceCatalogNumber, sourceInstitutionCode, sourceTaxonId, targetTaxonId)`;
4. else `(sourceTaxonId, targetTaxonId, interactionTypeId, decimalLatitude, decimalLongitude, eventDate)`.

**Verified.** On iNaturalist flower records (76% of our network): `referenceCitation` 100% populated
**and it is the observation URL** (`inaturalist.org/observations/90668398`) — a true observation
identity. `sourceCatalogNumber`/`InstitutionCode`/`CollectionCode` 99.8%, `sourceId`/`targetId` 100%,
`eventDate` 100%. My earlier "0% populated" claim was a sampling error (the file's head is
EuPPollNet/CropPol, which genuinely lack these).

**Expected effect.** Published GloBI work removes 38–41% of *records*. The **edge set should move much
less** — dedup removes repeated evidence for pairs we already hold. It does change `n_records`, the
Tier-1 definition, and any effort-weighted feature.

---

## 8. Life-stage filter

**Rule.** Drop pollinator-side records whose `sourceLifeStageName` indicates a non-adult stage
(larva, caterpillar, nymph, pupa, egg). Retain nulls, flagged.
**Verified.** `sourceLifeStageName` is 57.7% populated on iNat flower records — partial, so this is a
flag-and-filter, not a guarantee. Larval records are host-plant associations, not flower visitation.

---

## 9. Aggregate to edges + provenance

Group surviving records by `(plant_id, pollinator_id)` and compute the schema below.

---

## 10. Feature-coverage intersection

Intersect with plant/pollinator feature coverage. **Emit the pre-intersection edge list too** — the
metaweb is a contribution independent of whether we happen to have features for both species.

---

## Output schema

**`edges_v2.parquet`** — one row per unique (plant, pollinator):

| column | type | notes |
|---|---|---|
| `plant`, `pollinator` | str | resolved labels |
| `plant_id`, `pollinator_id` | str | `GBIF:3034002` — canonical key |
| `plant_rank`, `pollinator_rank` | str | species \| genus |
| `plant_family`, `pollinator_family`, `pollinator_order` | str | from path fields |
| `tier` | str | A \| B (best tier supporting the edge) |
| `types` | str | comma-joined interaction types |
| `n_records` | int | after dedup |
| `n_sources` | int | distinct `sourceNamespace` |
| `sources` | str | namespace list — **drives leave-one-source-out** |
| `n_inat` | int | iNaturalist-attributed records |
| `n_observations` | int | distinct `referenceCitation` — independent-evidence count |
| `first_year`, `last_year` | int | **drives the temporal split** |
| `role_source` | str | kingdom \| phylum \| membership |

Plus `nodes_plants.parquet` / `nodes_pollinators.parquet` (id, label, rank, family, order, kingdom,
feature-coverage flags) and `edges_v2_card.md` (the yield ledger below, filled).

---

## Yield ledger (filled on execution)

| # | step | records in | records out | edges | plants | pollinators | note |
|---|---|---|---|---|---|---|---|
| 1 | CONUS filter | 24.5M | 5,333,308 | — | — | — | verified |
| 2 | type tiering | | | | | | A / B split |
| 3 | role assignment | | | | | | ambiguity rate |
| 4 | orientation | | | | | | violations dropped |
| 5 | id resolution | | | | | | unresolved rate |
| 6 | rank policy | | | | | | species vs genus |
| 7 | dedup | | | | | | shrinkage % |
| 8 | life stage | | | | | | non-adult dropped |
| 9 | aggregate | | | | | | |
| 10 | coverage ∩ | | | | | | |

---

## Per-step EDA (run and inspect before proceeding)

1. **Spatial** — map record density; confirm no ocean/Canada leakage.
2. **Types** — count by tier; Tier-A/B overlap (edges supported by both).
3. **Roles** — ambiguity rate by namespace; list top genera resolved by fallback (homonym check).
4. **Orientation** — count and *inspect* dropped violations; they should look like real errors.
5. **Resolution** — id-namespace coverage; count names mapping to >1 id and ids with >1 name.
6. **Rank** — species vs genus node counts; degree distribution of each.
7. **Dedup** — shrinkage by namespace; distribution of records per observation.
8. **Life stage** — value counts; sanity-check dropped examples.
9. **Edges** — degree distributions (both sides), connectance, `n_sources` histogram, temporal span.
10. **Coverage** — what is lost at intersection, and is the loss taxonomically biased?

---

## Unit tests (`tests/test_edges_v2.py`)

**Structural**
1. No self-pairs (`plant_id != pollinator_id`).
2. Edge list is unique on `(plant_id, pollinator_id)`.
3. Every id in edges appears in the corresponding node table (referential integrity).
4. `plant_rank`/`pollinator_rank` ∈ {species, genus}; no coarser rank survives.

**Semantic**
5. Every `plant_id` resolves to a plant kingdom; every `pollinator_id` to an animal kingdom.
6. Zero surviving orientation violations.
7. `tier == 'A'` ⟹ `types ⊆ {visitsFlowersOf, pollinates}`.
8. No plant node is also a pollinator node (and if a homonym forces it, the test names it).

**Counting**
9. `n_records ≥ 1`; `n_sources ≥ 1`; `n_inat ≤ n_records`; `n_observations ≤ n_records`.
10. `first_year ≤ last_year`; both within [1800, current year].
11. Sum of per-edge `n_records` ≤ records surviving step 8 (no invention).

**Idempotence / regression**
12. Re-running dedup on the output changes nothing.
13. Pipeline is deterministic: two runs produce byte-identical parquet.
14. Every v1 edge is either present in v2 or appears in `dropped_edges.csv` **with a reason**.

---

## Open decisions for Dan

1. **Tier B in the primary edge list, or a separate file?** Recommend: same file, `tier` column, ablated.
2. **Genus nodes in the primary edge list, or separate?** Recommend: same file, `*_rank` column,
   with every headline result reported at both resolutions.
3. **Null-coordinate records** — currently dropped at step 1. They are valid metaweb evidence but have
   no spatial features. Recommend: keep in the pre-intersection edge list, exclude from the modelling set.
4. **`argumentTypeId`** — verified a no-op on this file (all rows `supports`), and the separate
   refuted-claims file **failed verification** as a negative set (50.4% of our v1 positives appear in
   it; 76% of its rows are correctly-oriented animal→plant under a single boilerplate EOL reason).
   Recommend: no refuted-based filtering until the GloBI maintainers clarify what the flag denotes.
