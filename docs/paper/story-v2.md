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

## 6b. Current standing (2026-09-10 02:30, clean protocol, three seeds everywhere)

**Candidate final system (M5.0, pending Dan's call):** species nodes carrying frozen BioCLIP-2 text, connected by training
interaction edges and by presence-weighted edges to cell x month nodes, two relation-typed R-GCN layers, **symmetric
leave-own-edges-out** (a random 30% of plants and of pollinators lose their edges each epoch, so both kinds of cold species are
rehearsed), pair head with the retriever objective; top-500 re-scored by an identity-token re-ranker trained on the retriever's
own hard negatives. No taxon nodes: the text already carries taxonomy and pooled genus/family nodes over-smooth it.

| regime | final (M5.0) | System v2 (taxon nodes) | best other model |
|---|---|---|---|
| cold plant (AUPR / AUROC / nR@10) | **0.213** / 0.970 / 0.376 | 0.209 / 0.972 / 0.383 | Wide & Deep 0.144 |
| cold pollinator | **0.135 / 0.917** / 0.399 | 0.117 / 0.915 / 0.361 | pair GBM 0.044 |
| cold both | **0.105 / 0.888 / 0.341** | 0.094 / 0.886 / 0.288 | ANTHEIA v1 scalar 0.044 / nR@10 0.324 |
| warm | 0.051 / 0.974 / **0.304** | 0.050 / 0.976 / 0.298 | SVD + taxonomic **0.105** |
| within sites (mean AUPR / precision@L) | 0.216 / **0.241** | 0.214 / 0.235 | congeneric transfer 0.222 / 0.239 (tie) |

Paired bootstraps (System v2 as reference on cold plant): every comparison model p < 0.001. Within sites: vs congeneric
transfer -0.008 (p = 0.13); vs our earlier trees -0.008 (p = 0.016); vs v1 +0.010 (p < 0.001).

**What the ablations say the model is** (all on the symmetric retriever, seed 42 unless noted):
- Month-collapsed cell nodes: 0.167 / 0.210 / 0.099 vs 0.169 / 0.215 / 0.101 (cold plant / within sites / cold pollinator).
- No cell nodes: retriever alone 0.165 / 0.212 / 0.097 (near-inert); but the *system* without them loses ~0.01 on every table (cold plant 0.206 vs 0.215, within sites 0.208 vs 0.216, cold pollinator 0.116 vs 0.129): the presence edges shape the top-500 lists the re-ranker sees. Cell nodes stay in the final model.
- No taxon nodes (3 seeds): 0.174 / 0.218 / 0.129. Better everywhere; largest on the pollinator side.
- Attention aggregation: 0.155 / 0.196 vs 0.150 / 0.204 for mean, 12x the time. The layer is not the lever.
- Plant-only rehearsal (v1 scheme): cold pollinator 0.006, cold both 0.011 -- at identity-free null level.
The model's work is done by the text features, the interaction edges and the rehearsal scheme.

**The spatio-temporal result, restated.** Presence is not inert; the graph route for it is. Handed to the model explicitly the
same surfaces move the numbers: pair co-presence statistic in the head +0.024 on cold plant (scalar 0.166 < time 0.171 ~ space
0.172 < joint 0.174: the marginalisation ordering, mostly prevalence); per-species presence embedding +0.04 on cold pollinator
(0.143 vs 0.101 for the retriever). Every such route costs 0.006-0.025 within sites, where the survey fixes presence. So:
(i) co-presence is a between-site signal -- it decides who can meet, not who interacts once they meet; (ii) the form of
integration matters more than the source -- explicit pair or species-level presence beats presence diffused through averaging
edges; (iii) season adds little beyond range in every model we built (the joint-vs-space gap appears only in the re-ranking
ladder). A presence-augmented variant (R6b retriever + re-ranker, M5.2) is reported as one labelled row for the
pollinator-cold regime, not as a second system.

**Evaluation-protocol corrections made before use:** filtered ranking on warm; pollinator negative pool restricted to training
pollinators on the pollinator-side splits, including the re-ranker's hard negatives.

Residual gaps: warm AUPR (SVD completes the plant's own row; we lead AUROC and recall); within sites a tie with congeneric
transfer, concentrated on cold plants inside sites (0.233 vs 0.260) and on the GloBI-derived surveys (web-of-life: four-way tie).

## 7. Open items before writing

- 3-layer R-GCN probe (M3.2) for the within-site residuals; M2.12 local seed 1.
- Dan's call: headline as 'leads every universe column, ties the trees within sites', or push the within-site gap first.
- Split battery (S1-S4) and prospective holdout for the frozen system; test split once.
- Test split, held for Dan's call on one model vs two.
- Whether to include the trained-sub-universe numbers (379 plants, 5,354 candidates) as the
  marginalisation table, or re-run with text-imputed vectors so the full universe can be used. The
  former is cleaner; the latter is comparable to Table 2.
