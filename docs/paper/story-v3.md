# Story v3 — one model for every cold-start regime of a pollination network (draft 2026-09-10)

Working title: *Rehearsing absence: a relational model that predicts plant–pollinator interactions for species with no
records, in any direction.*

Target: CV-for-ecology workshop, 8 pages. One system, one evaluation protocol, four regimes plus within-site completion,
three mechanism results with controls, two protocol corrections the field needs. All numbers below are validation, clean
protocol, three seeds unless marked; test split once, at the end.

## 1. The problem in one paragraph

Most plant–pollinator pairs that exist have never been recorded. GloBI covers 11,031 plants x 13,124 pollinators for the
continental US and most species have few or no partners on file; every field survey is incomplete. Prediction has to work for a
plant with no records, for a pollinator with no records, for both at once, and inside a surveyed community where the species
list is known but the interactions are not. The literature evaluates one of these at a time, usually the first, usually with
sampled negatives. We evaluate all of them under one protocol at network prevalence.

## 2. The model (Figure 1)

A relational GNN over species nodes (frozen BioCLIP-2 text of the species name, projected) and cell x month nodes (frozen
joint-field coordinate encoding), connected by training interaction edges and presence-weighted occurrence edges; two R-GCN
layers (Schlichtkrull 2018); a pair head with the retriever objective (sampled softmax with logQ + BCE). Its top-500 per query
are re-scored by a single-stream identity re-ranker trained on the retriever's own confusers with a pooled objective.

The one non-standard ingredient is the training scheme: **symmetric leave-own-edges-out**. Each epoch 30% of plants and 30% of
pollinators lose their interaction edges before the forward pass, so the model rehearses both kinds of cold species. The
plant-only version of this (DropoutNet-style, what we started with) collapses to the level of identity-free nulls the moment a
pollinator is unseen.

Design choices that were tested and rejected, each with a number in Table 3: genus/family nodes (hurt: the text already carries
taxonomy), attention aggregation (equal at 12x cost), a per-species memory vector, an explicit co-presence statistic, degree
encoding, presence embeddings as node inputs, co-occurrence negatives, a third layer, direct genus->partner edges.

## 3. Evaluation (Section 3 of the paper)

- **Regimes** (DTI-style entity-disjoint splits, frozen, degree-stratified): cold plant (663 plants x 13,124), cold pollinator
  (1,927 x 1,312), cold both (249 x 1,312), warm (10% of edges held out; filtered ranking). AUPR at network prevalence is
  primary; AUROC and normalised recall@10 alongside; re-expressed AUPR (1:3, 1:1) and recall@50 in the appendix.
- **Within-site completion**: 91 surveyed networks (71 Guzman et al. 2022 British Columbia + LaManna, 19 web-of-life); every
  local pair removed from training; an absent pair among surveyed species is an observed non-interaction. Mean AUPR over
  networks with a bootstrap CI, precision@L, NODF of the predicted network at matched connectance.
- **Two corrections other papers should adopt**: (i) filtered ranking on warm splits -- a plant's known partners are not
  negatives; (ii) on pollinator-side splits, held-out pollinators must never appear in training, *including as sampled negatives
  and as the re-ranker's hard negatives*. Without (ii) the re-ranker is trained to call the evaluated pollinators negative and
  the split is meaningless (we measured 0.010 vs 0.128 for the same model).
- **Comparison set**: nulls (popularity, co-occurrence N), ecological baselines (phenology x abundance, congeneric transfer,
  SVD + taxonomic imputation, pair-feature GBM, NECTAR-style plausibility), ANTHEIA v1 (our prior work), recommender
  architectures (two-tower, Wide & Deep, DCN-V2), all under the same harness.

## 4. Results

### 4.1 Table 1 -- regimes (AUPR / AUROC / nR@10)

| | cold plant | cold pollinator | cold both | warm |
|---|---|---|---|---|
| best comparison model | Wide & Deep 0.144 / DCN-V2 0.945 / ANTHEIA v1 0.383 | pair GBM 0.044 / 0.858 / ANTHEIA v1 0.305 | ANTHEIA v1 0.044 / pair GBM 0.853 / ANTHEIA v1 0.324 | SVD **0.105** / 0.928 / 0.250 |
| **final system** | **0.213** / 0.970 / 0.376 | **0.135 / 0.917 / 0.399** | **0.105 / 0.888 / 0.341** | 0.051 / 0.974 / **0.304** |

Cold plant: 1.5x the best published architecture, every comparison p < 0.001 (final vs System v2: tie, p = 0.47). Cold pollinator and cold both: final vs System v2 +0.018 (p < 0.001) and +0.012 (p = 0.008); vs the best comparison model +0.085 and +0.058. Cold pollinator and cold both: 3x and 2.4x the
best model; every identity-based method (congeneric, SVD, trees, pair MLP) is at chance there. Warm: best whole-list ordering
and recall; SVD owns the head of the list (it completes the plant's own row, which leave-own-edges-out deliberately does not
exploit) -- stated as a design trade-off.

### 4.2 Table 2 -- within-site completion

Final 0.216 [0.201, 0.232], precision@L 0.241 (best of any model), NODF 52 (observed 36; congeneric 70, popularity 92).
Congeneric transfer 0.222 (tie, p = 0.21); our earlier hand-engineered trees 0.222 (p = 0.085); pair GBM 0.203 (p = 0.006).
On the 19 web-of-life networks everything strong ties at 0.236; the residual sits in the British Columbia surveys and on cold
plants inside sites (0.233 vs congeneric 0.260), where an exact genus lookup still beats a learned one.

### 4.3 Table 3 -- what the model is (ablations, symmetric retriever, seed 42 unless 3 seeds)

| change | cold plant | within sites | cold pollinator |
|---|---:|---:|---:|
| full model (final) | 0.174 | 0.218 | 0.129 |
| + taxon nodes (System v2 retriever, 3 seeds) | 0.170 | 0.215 | 0.101 |
| plant-only rehearsal (v1 retriever, 3 seeds) | 0.150 | 0.204 | 0.006 |
| no cell nodes (system level: -0.01 on every table) | 0.165 | 0.212 | 0.097 |
| month-collapsed cells | 0.167 | 0.210 | 0.099 |
| attention aggregation (12x time) | 0.155 | 0.196 | -- |
| + memory vector (R1) | 0.170 | 0.179 | -- |
| + co-presence statistic in head (R4) | 0.174 | 0.195 | -- |
| + degree encoding (R5) | 0.167 | 0.197 | -- |
| + presence embedding as node input (R6b) | 0.165 | 0.200 | 0.143 |
| co-occurrence negatives (R7) | 0.152 | 0.204 | -- |
| re-ranker on top (final) | +0.039 | -0.003 | +0.006 |

Three readings. (a) The rehearsal scheme is the lever: plant-only -> symmetric moves cold pollinator from chance to 0.10.
(b) The layer is not: attention = mean. (c) Every continental prior helps the universe and hurts the site.

### 4.4 The spatio-temporal result: opportunity is not affinity (Section 5)

Interaction = opportunity (co-presence in space and time) x affinity (who interacts given they meet). The two factors behave
differently in every experiment we ran:

- **Opportunity is a between-site quantity.** Explicit co-presence lifts continental ranking (+0.024 cold plant; +0.04 cold
  pollinator as a node input) and lowers within-site accuracy every time (-0.006 to -0.025), because the survey fixes
  co-presence. Models built on it alone (ANTHEIA v1, pair GBM) are the best non-neural models exactly where identity is empty.
- **How the product is formed matters more than the source.** Same surfaces, three routes: diffused through occurrence edges
  (+0.004 retriever, ~0.01 system), per-species embedding (+0.04 pollinator side), explicit pair statistic (+0.024 plant side).
- **The marginalisation ladder** (Section 5.1, unchanged): joint > space > time > scalar for re-ranking, all contrasts p <= 0.008;
  reproduced inside the head at compressed magnitude (0.174 > 0.172 ~ 0.171 > 0.166, two-thirds prevalence). Phenology adds
  little beyond range in every model we built; the season axis on the graph is inert.
- **Affinity is what the graph learns**: text + edges + rehearsal. It wins every cold regime and ties within sites.

Closing experiment (A4, running): the final model with an explicit opportunity term in the head, scored with both terms on the
continent and with the opportunity term switched off within sites. If it holds, the final model carries an explicit space-time
factor without paying for it inside sites; if not, the factorisation is reported as an analysis and the presence variant as one
labelled row.

## 5. Figures

1. Model + rehearsal schematic (one panel: graph, dropped edges, re-ranker).
2. Regime bars: final vs best comparison model per regime (AUPR, log scale), with the v1-retriever collapse on the pollinator
   side as the "before".
3. Within-site: predicted vs observed network for one BC site (final, congeneric, popularity) with NODF; mean AUPR by dataset.
4. Opportunity vs affinity: the ladder (re-ranking) and the in-head ladder side by side; within-site cost vs universe gain
   scatter over all prior-adding arms.
5. Qualitative: a cold plant's top-10 with the meeting cell x month map (the use case).

## 6. What we do not claim

That the season axis helps inside the graph; that the model beats matrix factorisation on warm AUPR; that it beats an exact
genus lookup inside a site; that occurrence-derived presence is free of effort bias (prevalence is two-thirds of the
co-presence gain).

## 7. To do before writing

- A4 read-out (opportunity term on/off within sites); M5.2 presence variant across regimes; M5.1 confirms cells stay.
- Dan: adopt M5.0 (or A4 if it holds) as the final; unlock the test split; decide on a prospective (post-cutoff-year) holdout.
- Calibration curves on validation; low-degree stratum table (final leads: nR@10 0.35 vs <= 0.27); per-dataset within-site table.
- Figures 2-5; one qualitative example.
