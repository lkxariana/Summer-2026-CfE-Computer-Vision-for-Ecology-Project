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

Interaction = **opportunity** (the two species are in the same place at the same time) x **affinity** (they are the kind of
species that interact given they meet). The model computes both.

*Affinity.* A relational GNN over species nodes (frozen BioCLIP-2 text of the species name, one shared 768->128 projection) and
cell x month nodes (frozen joint-field coordinate encoding), connected by training interaction edges and presence-weighted
occurrence edges; two R-GCN layers (Schlichtkrull 2018), d = 128.

*Opportunity.* One scalar per pair, c(p,q) = z-score of log(1 + u_p . u_q), where u is the 256-D SVD projection of the species'
presence surface, so u_p . u_q is the expected number of cell-weeks in which both species are present. Concatenated with the
pair's affinity features into the head MLP.

*The switch.* Inside a surveyed site every candidate pair is co-present by construction, so opportunity is constant there and
only affinity should be scored: c is held at its mean. Same trained weights, one inference-time flag, decided by the task.

*Second stage.* The retriever's top-500 per query are re-scored by a single-stream identity re-ranker (transformer over
[CLS, plant name, candidate name], its own projection of the frozen names) trained on the retriever's own confusers with a
pooled objective, added as a residual to the retriever score.

The one non-standard training ingredient is **symmetric leave-own-edges-out**. Each epoch 30% of plants and 30% of
pollinators lose their interaction edges before the forward pass, so the model rehearses both kinds of cold species. The
plant-only version of this (DropoutNet-style, what we started with) collapses to the level of identity-free nulls the moment a
pollinator is unseen.

Design choices that were tested and rejected, each with a number in Table 3: genus/family nodes (hurt: the text already carries
taxonomy), attention aggregation (equal at 12x cost), kingdom-specific text projections (a wash), a per-species memory vector,
degree encoding, presence embeddings as node inputs, co-occurrence negatives, a third layer, direct genus->partner edges,
re-ranker tokens taken from the retriever's own representations (its graph outputs make the second stage agree with the
first: 0.182 vs 0.230).

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

### 4.1 Table 1 -- regimes (AUPR / AUROC / nR@10), three seeds

| | cold plant | cold pollinator | cold both | warm |
|---|---|---|---|---|
| best comparison model | Wide & Deep 0.144 / DCN-V2 0.945 / ANTHEIA v1 0.383 | pair GBM 0.044 / 0.858 / ANTHEIA v1 0.305 | ANTHEIA v1 0.044 / pair GBM 0.853 / ANTHEIA v1 0.324 | SVD **0.105** / 0.928 / 0.250 |
| affinity only (no opportunity term) | 0.213 / 0.970 / 0.376 | 0.135 / 0.917 / 0.399 | 0.106 / 0.888 / 0.341 | 0.051 / 0.974 / **0.304** |
| **final (opportunity x affinity)** | **0.230 / 0.972 / 0.386** | **0.162 / 0.926 / 0.427** | **0.142 / 0.907 / 0.403** | 0.051 / 0.974 / 0.296 |

Cold plant 1.6x the best published architecture; cold pollinator 3.7x and cold both 3.2x the best model of any kind, where every
identity-based method (congeneric, SVD, trees, pair MLP) is at chance. Warm: best whole-list ordering and recall; SVD owns the
head of the list (it completes the plant's own row, which leave-own-edges-out deliberately does not exploit) -- a stated design
trade-off. The opportunity term is worth +0.017 / +0.027 / +0.036 AUPR on the three cold regimes and nothing on warm.

### 4.2 Table 2 -- within-site completion

| | mean AUPR | precision@L | NODF (obs 36) |
|---|---|---|---|
| congeneric transfer | **0.222** [0.205, 0.242] | 0.239 | 70 |
| boosted trees (our earlier system) | 0.222 [0.205, 0.239] | **0.252** | 62 |
| pair-feature GBM | 0.203 | 0.218 | 83 |
| **final, opportunity term held at its mean** | 0.217 | 0.249 | 54 |
| final, opportunity term left on | 0.213 | 0.238 | 50 |

The switch is worth +0.004 AUPR and +0.011 precision@L on the system and +0.010 / +0.010 on the retriever alone (0.223 / 0.248
vs 0.213 / 0.238), on every seed. Mean AUPR is a statistical tie with congeneric transfer (-0.005, p > 0.1 on the paired
network bootstrap); precision@L is within 0.003 of the trees; the predicted network's nestedness is the closest to observed of
any model above 0.20 AUPR. Chance AUPR (mean connectance) is 0.135, so the whole field is between 1.2x and 1.7x chance here:
an absent pair among surveyed species is often an unobserved real interaction, which caps AUPR for everyone.

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

**The factorised model (A4, one seed; seeds 0/1 and the re-ranked version running).** The final retriever plus an explicit pair
co-presence term in the head, scored with both terms on the continent and with the opportunity term switched off inside a site:

| scoring | cold plant | cold pollinator | within sites: mean AUPR / precision@L / warm plants / cold plants |
|---|---:|---:|---|
| affinity-only retriever (final) | 0.174 | 0.129 | 0.218 / 0.241 / 0.220 / 0.233 |
| both terms everywhere | 0.197 | 0.155 | 0.215 / 0.239 / 0.217 / 0.237 |
| opportunity term off within sites | 0.197 | 0.155 | **0.224 [0.209, 0.240] / 0.249 / 0.226** / 0.203 |

Same trained weights, one switch that follows from the factorisation, and the model gains 0.02-0.03 on the continent while
becoming the best within-site model measured (congeneric transfer 0.222, our trees 0.222; precision@L 0.249 vs the trees' 0.252).
The one place the switch costs is cold plants inside sites (0.237 -> 0.203): for a plant with no edges the co-presence term
acts as a useful prior even where co-presence is nominally fixed, which is itself consistent with the factorisation (the affinity
term has less to say about an edgeless plant). The switch is a task-dependent inference choice and is stated as such; if the
seeds hold, the paper's final model is the factorised one and Table 1/2 carry it.

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
