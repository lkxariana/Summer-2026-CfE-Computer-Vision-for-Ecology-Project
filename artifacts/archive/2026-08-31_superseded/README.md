# Superseded results (archived 2026-09-02)

Every result in this directory was computed on an earlier interaction network built from a June 2026
GloBI export using name-string matching. It has been superseded by the network built under
`docs/paper/dataset-construction.md` from the pinned 2026-08-26 snapshot.

**These numbers are not comparable to current results.** The underlying network differs in:

- taxon identity — nodes are now keyed on authority identifiers (GBIF and equivalents) rather than
  name strings;
- taxonomic resolution — genus-rank taxa are now retained as nodes;
- interaction scope — records are now tiered (flower-visitation terms vs. general association terms);
- deduplication — repeated records of the same observation are now collapsed;
- role assignment — trophic roles now derive from taxonomic kingdom rather than species-list membership.

Retained for provenance: the experiment log in `EXPERIMENTS.md` references these files, and the
methodological findings they support (evaluation design, the two-objective result, the temporal
encoding ladder) remain informative even though the absolute numbers do not carry over.
