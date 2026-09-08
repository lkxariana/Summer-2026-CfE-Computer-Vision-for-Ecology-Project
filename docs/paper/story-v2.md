# Story v2 — three axes, and why the spatio-temporal one has looked useless (draft 2026-09-07)

Working title candidates: *Marginalisation hides the niche: what spatio-temporal fields can and
cannot do for cold-start plant–pollinator prediction* · *Three axes of a pollination network*.

Target: workshop paper. One mechanistic claim with an unconfounded test, two supporting negative
results with controls, a dataset and a protocol. Not a leaderboard paper — the best model beats
"guess *Apis* and *Bombus*" by 0.09 nR@10 and a reviewer will say so.

## 1. Setting

Cold-start link prediction (Stock et al. 2018 setting B): rank all 13,124 pollinators for a plant
with no recorded interactions. Tier-A GloBI network, 101k interactions, 11,031 × 13,124 modelled
universe, 93.5% feature coverage. Frozen degree-stratified plant split; nrecall@10 primary,
PR-AUC for cross-plant comparability; nulls in every table (`docs/paper/evaluation-protocol.md`).

## 2. Three nearly independent input axes

| axis | what it is | independence from the other two |
|---|---|---|
| **how much** — prevalence | range size, record count, degree | — |
| **who** — identity | taxonomy, BioCLIP-2 text (≈ soft taxonomy), phylogeny | text is 99% predictable from taxonomy-shaped inputs |
| **where and when** — niche | (cell × week) presence field | 64% (plants) / 84% (pollinators) of its variance is unexplained by prevalence + taxonomy + text; the pair statistic is 87.5% independent of N and range size |

Measurement: held-out ridge from the other axes onto the 256-D field coordinates (EXPERIMENTS.md,
field-embedding phase). The third axis is real and large. The question is why it never helps.

## 3. Each axis has a depth in the ranked list

Validation, tier A, 663 plants, each feature alone over all candidates:

| feature alone | per-plant AUC | nR@10 | R@200 |
|---|---:|---:|---:|
| popularity | 0.941 | **0.244** | **0.594** |
| N (range overlap) | 0.782 | 0.105 | 0.331 |
| per-cell co-activity (grid surfaces) | 0.851 | 0.011 | 0.260 |

Prevalence owns the head of the list. Identity owns the conditional re-rank (taxonomic affinity is
the strongest single signal; congeneric transfer 0.293). The niche axis orders the *whole* list
better than N (AUC 0.851 vs 0.782) and cannot pick the head (0.011). Confirmed three independent
ways: SVD grid surfaces; the SDM's pollinator head vector (step 1, 3 seeds, neutral in the full
model, −0.021 without text); the joint two-kingdom field model (encounter integral alone 0.06–0.08,
re-ranking popularity's top 200 lifts nR@50 0.353 → 0.42–0.45 and PR-AUC 0.066 → 0.09–0.10).

**Consequence for the field.** Any evaluation at small k, or any pooled metric, will read the niche
axis as useless. It is a filter and a mid-list re-ranker, and it should be reported as one.

## 4. The mechanistic claim: marginalisation destroys the signal

Everything the literature computes from spatio-temporal data is a marginal collapsed before the
comparison: range overlap (weeks summed out), phenological overlap Σmin / Schoener's D (cells summed
out), or a scalar of both. Twelve marginalised encodings we tested are null for retrieval (Δ,
seven standard overlap statistics, 52-week curves, 260-D trajectories — EXPERIMENTS.md, encoding
ladders). The per-cell statistic is the only spatio-temporal input that has ever cleared
significance (bilateral ladder: +0.018–0.019, p≈0.006; SDM-marginal on the same data: +0.010, n.s.).

The unconfounded test (`eval/run_marginalisation_test.py`): one joint field model, one presence grid
P[c,w] = σ(u·h(c,w)), four expected co-presence statistics of the *same two surfaces* that differ
only in what is summed out first. Re-ranking popularity's top 200, nR@50:

| scheme | joint | space (range overlap) | time (phenology overlap) | scalar |
|---|---:|---:|---:|---:|
| uniform | **0.448** | 0.404 | 0.268 | 0.252 |
| tg_spatial | **0.418** | 0.385 | 0.236 | 0.197 |
| tg_spatiotemporal | **0.426** | 0.406 | 0.225 | 0.208 |
| slds | **0.419** | 0.393 | 0.225 | 0.207 |

Strict ladder joint > space > time > scalar under every scheme; every joint-vs-marginal contrast
p ≤ 0.0024. **Replicated on the production surfaces over the full protocol universe** (663 plants,
13,124 candidates; `eval/run_marginalisation_surfaces.py`): re-ranking nR@50 0.360 / 0.344 / 0.255 /
0.200, all contrasts p ≤ 0.008, including on the unseen-genus stratum; pooled PR-AUC 0.077 against
popularity's 0.053 (+46%). On the production surfaces the niche term improves cross-plant ordering
(PR-AUC) but not within-plant recall at 50; the learned joint field did both. Figure:
`results/figures/marginalisation_ladder.png`. **Phenology retains almost nothing when marginalised (time ≈ scalar) and adds over range
overlap only when kept per cell (joint − space = +0.044, p<0.001).** That is the original ANTHEIA
asymmetry hypothesis, finally tested without changing the data or the model between arms, and it is
the reason marginal-curve overlap statistics have found so little.

## 5. Supporting negative results (each with its control)

- **Identity collapses where it is empty.** Boosted ranker 0.351 on seen-genus plants, 0.086 on
  unseen-genus plants (79 of 663). Smoothing identity fails three ways (back-off crosses, BioCLIP
  kernel, phylogeny). Removing the text block *raises* the unseen stratum (0.119 → 0.146) while
  costing 0.062 overall: identity misleads exactly where it has nothing to say.
- **Niche transfer degrades gracefully but does not win there.** Taxonomy-free joint field model,
  3 seeds: unseen-genus 0.188 against seen-genus 0.207 (ratio 0.91); the reference model's ratio is
  0.39. The contrast against the taxonomy-free surface control (+0.042) is not significant on 79
  plants (p=0.21). SVD-with-taxonomic-imputation (0.256) and popularity (0.234) remain better on that
  stratum. Report as a pattern and a mechanism, not a method.
- **Commensurability was not the bottleneck.** Putting both sides in one learned field space (joint
  SINR over both kingdoms) is null in the full model (all arms p >= 0.10, 3 seeds) and does not beat
  the SVD grid projection as the sole spatio-temporal input.
- **Prevalence is signal.** Three interventions that remove it (logQ off, prevalence-normalised
  encounter term, degree-matched negatives) all cost 0.04–0.09, p<0.001. Methods sold on removing
  abundance confounding lose accuracy on this data.
- **The SDM sampling scheme moves the SDM, not the retrieval, and does not cancel effort.**
  Effort-matched backgrounds improve held-out species discrimination (top-10 0.158 → 0.193) but the
  unit-normalised species vector predicts record count *better* under them (0.52 → 0.62 plants,
  0.78 → 0.84 pollinators). Against a background that follows effort, exceeding the effort surface
  is what the direction encodes.

## 6. What the paper does not claim

A model that beats the nulls by a wide margin; a temporal effect that survives marginal encoding;
that effort can be removed from occurrence-derived embeddings by negative sampling; that niche
transfer is the best cold-start fallback (it is fourth).

## 6b. Current standing (2026-09-08, end of the ladder)

**One system:** an R-GCN retriever (species, genus/family and cell x month nodes; two relation-typed layers;
leave-own-edges-out; the retriever objective) whose top-500 are re-scored by an identity-token re-ranker trained on the
retriever's own hard negatives with a pooled objective.

| | universe AUPR | AUPR 1:3 / 1:1 | AUROC | nR@10 | nR@50 | unseen-genus | local mean AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| trees (our earlier hand-engineered system) | 0.103 | 0.81 / 0.91 | 0.883 | 0.336 | 0.447 | 0.256 | 0.222 [0.205, 0.239] |
| best external baseline (SVD + taxonomic) | 0.095 | 0.85 / 0.92 | 0.895 | 0.291 | 0.449 | 0.256 | 0.179 |
| **system** | **0.188** | **0.93 / 0.97** | **0.970** | **0.383** | **0.550** | **0.260** | 0.204 [0.19, 0.22] |

Ablations: R-GCN alone 0.150 / 0.368 / local 0.204; embedding retriever alone 0.159 / 0.282 / 0.166; re-ranker on the
embedding retriever 0.191 / 0.323 / 0.164; field tokens in the re-ranker inert (joint = space = time = identity-only).
Mechanisms, each with its control: (i) re-ranking on the retriever's own confusers under a pooled objective, +0.03-0.04
AUPR on either retriever; (ii) message passing through genus and cell nodes supplies the warm-plant information the pair
models lack -- within-site AUPR 0.165 -> 0.205, the only neural route that moved it (co-occurrence negatives and three
genus-token constructions did not). The marginalisation ladder stands as a result about the inputs; it is not the
model's engine.

Residual gap: within sites the system ties the trees (each inside the other's CI) but trails on the head
(precision@L 0.233 vs 0.252) and on cold plants (0.24 vs 0.27).

## 7. Open items before writing

- 3-layer R-GCN probe (M3.2) for the within-site residuals; M2.12 local seed 1.
- Dan's call: headline as 'leads every universe column, ties the trees within sites', or push the within-site gap first.
- Split battery (S1-S4) and prospective holdout for the frozen system; test split once.
- Test split, held for Dan's call on one model vs two.
- Whether to include the trained-sub-universe numbers (379 plants, 5,354 candidates) as the
  marginalisation table, or re-run with text-imputed vectors so the full universe can be used. The
  former is cleaner; the latter is comparable to Table 2.
