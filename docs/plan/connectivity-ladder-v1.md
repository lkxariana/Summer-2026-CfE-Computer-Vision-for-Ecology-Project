# Plan: connectivity-first model ladder (v1, 2026-09-08)

For review. Everything here is on validation (663 tier-A plants) until the final readout; the test
split is used once, at the end, on the models this plan selects.

## 0. Problem, metric, and what the ladder must show

**Task.** Cold-start link prediction: for a plant with no recorded interactions, score every one of
13,124 pollinators. Modelled universe 11,031 × 13,124, tier-A scoring, training on A+B
(`docs/paper/evaluation-protocol.md`).

**Primary metric: AUPR at network prevalence** (chance 0.00132), reported in one row with AUPR
re-expressed at 1:3 and 1:1 (negatives re-weighted from the full ranking, `eval/pool_prauc.py`) and
AUROC. Secondary: nrecall@10 and nrecall@50. Every table carries the nulls (popularity, N) and the
external baselines (congeneric, SVD+taxonomic) and our two reference models (boosted ranker /
routed; current embedding model). Paired bootstrap over plants, 3 seeds {42, 0, 1}, per-seed metrics
averaged (score averaging is an ensemble). Unseen-genus stratum (79 plants) reported for every model,
as a pattern, not a headline.

**Evidence the plan is built on.**

| finding | source |
|---|---|
| embedding model leads PR-AUC (0.159) at 3x the ecological baselines, trails on nR@10 (0.282 vs routed 0.336) | leaderboard |
| spatial blocks carry 19% of its PR-AUC (p<0.001) and little ranking; identity carries 0.068 and all ranking | block ablation, 3 seeds |
| within-plant softmax cancels every plant-side quantity; the model was never trained for the metric it leads on | architecture |
| joint per-cell statistic > space > time > scalar, p<=0.008, two universes | marginalisation ladder |
| the field signal re-orders the list below the head; it does not pick the head | three representations, nine seeds |
| taxonomic smoothing fails (3 ways); prevalence removal fails (3 ways) | earlier phases |
| text-imputed field vectors are noise (R^2 0.047); a trained text head is a different thing | step 1 |

**What the ladder must show to be a paper.** (a) A model of ours is best on the field's metric under
a protocol the field recognises. (b) The gain over the vector head comes from integrating identity
and the spatio-temporal field *without marginalising*, shown by controls inside the architecture, not
only by the statistics ladder. (c) The claims survive four splits and an evaluation with real
negatives.

## 1. Shared encoders and caches (Step 0, prerequisite, ~3 days)

Three encoders, frozen after this step, used unchanged by every model so the ladder is a controlled
comparison of integration rules.

### 1.1 Identity
BioCLIP-2 text embedding of the binomial, 768-D, frozen (exists). One token per species. No rank
tokens (hierarchy is already in the embedding; rank tokens are an ablation, §3). Per-side projection
to model width inside each model. **No learned taxon embeddings** (four forms tested, none helped).

### 1.2 Domain (location x time)
The joint field's coordinate encoder, frozen: `[Fourier(lat, lon), Climplicit(c, month(w)),
Fourier(week)] -> MLP -> h(c, w)`, 256-D (exists; `pipelines/sdm/train_joint_field.py`). Climplicit is
monthly, so time enters through the climate the species experiences at that place in that week. No
AlphaEarth. Cached for the whole grid as `grid_h [3335, 52, 256]` (exists).

### 1.3 Presence — joint field v2 with a text-conditioned head (the only new training in Step 0)
The current joint field covers 41% of taxa. Retrain with `u_s = MLP_text(BioCLIP2_s) + r_s`, where
`r_s` is a free residual for species with occurrences (zero-init, weight decay, dropped with p=0.3
during training so the text path learns) and `r_s = 0` at inference for zero-shot species. This is
LE-SINR's idea plus DropoutNet's, not the post-hoc ridge that failed. Scheme: `tg_spatiotemporal`
(best held-out SDM top-10; retrieval was indifferent to scheme).

*Validation, and the gate.* Hold out 5% of rows (pos-NLL, within-kingdom top-10, as now) **and 5% of
species entirely**: their presence must be predicted from text alone. Gate: held-out-species AUROC of
observed rows vs background >= 0.80. If the gate fails, zero-shot taxa fall back to the production
surfaces (PPE / SDM) and the paper says so; models 2 and 3 then run on mixed sources with a
`field_source` stratum.

*Exports.* `u` for all 24,155 taxa with `field_source in {observed, zeroshot}`; presence computed on
demand as `sigmoid(u . grid_h)`; **sparse token cache** per species: top-k (cell, week, log p),
k = 256, selected by presence with a floor of 2 tokens per month so the whole season is represented;
**raster cache** `[52, H, W]` float16 for figures and the raster ablation only.

### 1.4 Splits (frozen, `data/splits/`)
Keep the cold-plant split (75/10/15, degree-stratified) as primary. Add, following the DTI
convention (Pahikkala et al. 2015):

| split | train pairs | evaluated pairs | note |
|---|---|---|---|
| cold-plant (S2, exists) | train plants x all polls | val plants x all polls | primary |
| warm (S1) | 85% of edges among train plants x train polls, plus all zeros | held-out 15% of those edges vs all zeros among the same species | message passing and residuals are allowed to help here |
| cold-pollinator (S3) | all plants x train polls | train plants x val polls | pollinator split degree-stratified, seed 42 |
| cold-both (S4) | train plants x train polls | val plants x val polls | the true test of generalised encapsulation |

Direction is always "score pollinators for a plant"; candidate sets are the split's pollinator set;
AUPR is pooled over the split's evaluated pair universe. Models are trained once per split.

### 1.5 Local-network completion (the evaluation with real negatives)
Source-holdout networks (Web of Life, expert field networks in `edges.parquet` provenance). For each
network with >= 10 plants and >= 10 pollinators: pairs = plants present x pollinators present;
positives = recorded; **an absent pair is an observed non-interaction** because the site was surveyed.
Score AUPR per network, average, bootstrap over networks. Field models also score with presence
restricted to the network's cell(s) and months, which is the per-cell prediction the plan is for.
Prerequisite check: do these networks carry coordinates / dates in our provenance? If not, global
scores only, and the eval is weaker; say so.

## 2. Model 1 — retriever: pair head with a pooled objective (~2 days)

**Role.** Baseline for the ladder and candidate generator for models 2–3. Attribute-level
integration: presence enters as the per-species SVD projection, as now.

**Architecture.** Blocks per side (text, surface, pca, scale) -> per-side projection -> `h_p, h_q`.
Head: MLP over `[h_p, h_q]` **plus** a low-rank bilinear term `h_p^T U V^T h_q` (rank 64); no
`|h_p - h_q|`. **Degree heads:** two one-layer MLPs predicting log-degree from `h_p` and from `h_q`,
added to the logit (cold-start capable: inputs are content, not counts). BiasHead (documentation
effort, dropped at inference) kept. Reference head (`[h_p, h_q, h_p*h_q, |h_p-h_q|]`) kept as an
ablation row.

**Loss (the change that matters).** Pooled binary cross-entropy: positives (p, q) from training
edges; negatives (p', q') with p' uniform over train plants and q' from the popularity-tilted
proposal with logQ correction, 64 per positive; class-prior-aware weighting (nnPU, Kiryo et al.
2017) with prior pi = training connectance as an ablation. Optional within-plant softmax term with
weight lambda in {0, 0.25, 1} to *measure* the trade between AUPR and nR@10 — this is the two-task
story quantified.

**Experiments.**

| id | arm | question |
|---|---|---|
| M1.0 | current loss (reference) | have: 0.159 / 0.282 |
| M1.1 | pooled BCE only | is 0.159 objective-limited? |
| M1.2 | pooled BCE + softmax, lambda sweep | the AUPR-vs-nR@10 trade-off curve |
| M1.3 | M1.1 + degree heads | does explicit plant-side prevalence help pooled ranking? |
| M1.4 | M1.3 + nnPU weighting | does treating unlabelled as unlabelled help AP? |
| M1.5 | head: reference vs concat+bilinear; shared vs per-side projections | two one-flag ablations |

3 seeds each, ~5 min per run; ~3 h per split. **Decision.** Best AUPR arm becomes the retriever. If
no arm beats 0.159 by >= 0.01 (paired p < 0.05), the bottleneck is features not objective, and the
plan proceeds to model 2 with M1.0 as retriever. Also record retriever recall@500 (must be >= 0.85
for the re-ranker to have headroom; else K = 1000).

## 3. Model 2 — single-stream fusion re-ranker over identity and field tokens (~6 days)

**Role.** The model the marginalisation result asks for: the two species meet per (cell, week).
Token-level integration.

**Tokens per pair.** `[CLS] [id_p] [f_p,1..k] [id_q] [f_q,1..k]`. Identity token = projected
BioCLIP-2 text. Field token content = projected `h(c, w)` from the frozen domain encoder concatenated
with `log p_s(c, w)`. Four learned **type embeddings** (plant/pollinator x identity/field), added.
No positional embeddings: coordinates are the content. k = 128 default.

**Model.** L = 3 pre-norm transformer layers, width 192, 4 heads, dropout 0.1, self-attention over the
whole set (single-stream, ViLT/UNITER; Bugliarello et al. 2021 for the parity result over
dual-stream). Output: `logit = s_retriever(p, q) + MLP(CLS)` — residual re-ranking, so the fusion
model learns a correction and cannot do worse than the retriever at initialisation.

**Training.** Candidates per train plant = retriever top-K (K = 500) union the plant's positives union
64 random pollinators; loss = model 1's best pooled loss over these pairs. Hard negatives come free
from the top-K. Frozen encoders; trainable = projections, type embeddings, fusion layers, head.

**Inference.** Re-rank the retriever's top-K per plant; pairs outside keep the retriever score shifted
below the re-ranked block, so the full 13,124-way ranking and pooled AUPR remain defined.

**Experiments.**

| id | arm | question |
|---|---|---|
| M2.1 | identity tokens only (CLS + 2) | is attention over identity better than the vector head? expected wash — the control |
| M2.2 | identity + field tokens, k = 128 | the main result vs model 1 |
| M2.3 | k in {32, 128, 256}; selection top-k vs month-stratified | how much of the field is needed |
| M2.4 | presence as token feature vs additive attention bias | how presence should enter |
| M2.5 | dual-stream cross-attention (two streams, co-attention) | the architecture ablation the VL literature says is a wash |
| M2.6 | L in {2, 4} | capacity |
| **M2.7** | **marginal tokens**: field tokens built from the space marginal (cells, week-summed presence) or the time marginal (weeks, cell-summed presence) | **the marginalisation ladder inside the architecture** — joint tokens must beat both |
| M2.8 | attention maps: where and when the model places held-out true pairs | interpretability figure; quantitative: attention mass on cells where both are present |
| M2.9 | rank tokens (7 per species) | the addressability idea; expected null |

Cost: ~259 tokens per pair, width 192, L = 3; training ~30–40 min per run on one 4090, evaluation
over 663 x 500 pairs in seconds. 3 seeds for M2.2 and M2.7; single seed for the sweeps unless a
difference is within 0.01.

**Decision.** M2.2 vs model 1 on AUPR (paired plant bootstrap) and on local-network AUPR; M2.7 is
the mechanistic control the paper leads with if M2.2 wins.

## 4. Model 3 — R-GCN over species, taxa and cell x month (~5 days)

**Role.** Graph-level integration, for completeness and for the warm-split rows where message
passing is expected to earn its place. The 2024–25 link-prediction literature predicts it will not
beat model 2 on cold links; the point is to show that on our data.

**Graph.** Nodes: species (24,155), genus and family for both kingdoms (~5k), cell x month (40,020).
Edges: interaction (training edges of the split only), membership (species–genus, genus–family),
occurs-in (species -> cell x month, weight log presence, top-64 per species). Node init: species =
projected BioCLIP-2 text; taxa = learned; cell x month = projected frozen `h(c, month)`.

**Layers.** Two relation-typed message-passing layers (R-GCN with basis decomposition, 8 bases),
width 128, plain-torch sparse ops (neither PyG nor DGL is installed; sparse matmul per relation is
sufficient at this size). **Leave-own-edges-out**: each epoch, all interaction edges of a random 30%
of training plants are dropped when computing their embeddings, so the model learns to embed a plant
from taxon and cell edges alone — the situation every test plant is in. (This is the train/test shift
that cost 0.07 in the matrix-factorisation attempt.)

**Head and loss.** Model 1's best head and pooled loss on the message-passed embeddings.

**Experiments.**

| id | arm | question |
|---|---|---|
| M3.1 | full graph | main row |
| M3.2 | no cell x month nodes | taxonomy graph only |
| M3.3 | no taxon nodes | space-time graph only |
| M3.4 | warm split with vs without interaction edges at inference | the value of message passing where it should exist |
| **M3.5** | cell-only nodes (month collapsed) | the marginalisation control in graph form |

## 5. Cross-cutting experiments

| id | what | why |
|---|---|---|
| X1 | split battery: models 1–3 and all baselines on S1–S4 | robustness; fair to model 3 |
| X2 | local-network completion AUPR | real negatives; the connectivity product |
| X3 | unseen-genus stratum for every model | the failure mode, tracked |
| X4 | prospective holdout (train <= 2020, score 2021+) for the finalists | exists from an earlier phase; rerun |
| X5 | `field_source` stratum (observed vs zero-shot species) | guards the coverage fix |
| X6 | 3 seeds, paired bootstraps, mean ± sd everywhere | protocol |

## 6. Tables and figures

- **T1 main**: models x {AUPR@prev, AUPR@1:3, AUPR@1:1, AUROC, nR@10, nR@50}, cold-plant, val then test.
- **T2 split battery** (X1). **T3 local networks** (X2).
- **T4 mechanism**: statistics ladder (exists) + M2.7 + M3.5 — marginalisation destroys the signal, shown in the statistic, the architecture and the graph.
- **T5 ablations** per model. **T6 two-task trade-off** (M1.2 lambda sweep).
- **Figures**: marginalisation ladder (exists); attention maps of predicted interaction location for held-out pairs (M2.8); PR curves at the three prevalences.

## 7. Self-review against practice and against our own evidence

- *Objective matches metric.* AUPR is pooled; models 1–3 train with a pooled loss. The within-plant softmax that cancels plant-side information is now a measured ablation, not the default.
- *Frozen foundation encoders, light adapters* (LiT / adapter practice): BioCLIP-2 and the joint field are frozen; only projections, type embeddings, fusion or GNN layers train. Appropriate for ~130k positives.
- *Single-stream fusion with type embeddings* (ViLT, UNITER; Bugliarello et al. 2021): chosen over dual-stream, with dual-stream as an ablation.
- *Retrieve-then-re-rank* (Nogueira & Cho 2019; Covington et al. 2016): bounds compute and matches what the field signal does on this data (re-orders below the head). Residual re-ranking guarantees no regression at init.
- *Sampled negatives with logQ correction* (Yi et al. 2019) and *PU-aware weighting* (Kiryo et al. 2017): the negatives are unlabelled, and AP is sensitive to that.
- *Inductive GNN without leakage*: leave-own-edges-out; the exact failure we already paid for once.
- *Cold-start split battery S1–S4* (Pahikkala et al. 2015) and *AUROC + AUPR at native prevalence* (DTI convention), with re-expressed AUPR for comparability with ecology.
- *No taxonomic smoothing or learned taxon tables by default* — tested and failed three to four times; rank tokens and taxon nodes appear only as ablations (M2.9, M3.2).
- *Presence stays joint*: field tokens are (cell, week); the marginal versions exist only as controls (M2.7, M3.5). Consistent with the ladder at p<=0.008.
- *Controls for every claim*: identity-only fusion (M2.1) controls for "attention helps"; marginal tokens (M2.7) control for "the field helps"; the zero-shot species holdout controls for "coverage is real".
- *Reporting discipline*: 3 seeds, per-seed metrics, paired plant bootstrap, validation for every decision, test once.

**Risks, and what happens if they bite.** (1) Text head zero-shot quality fails the 0.80 gate -> mixed sources with a stratum. (2) Retriever recall@500 < 0.85 -> K = 1000. (3) Local networks lack coordinates -> global-score version only. (4) Model 2 does not beat model 1 -> the paper is model 1 (still best on the field's metric) plus the statistics ladder; M2.7 is then a negative that still supports the mechanism if joint tokens beat marginal tokens *within* model 2. (5) Compute: everything above fits two 4090s in the stated times.

**Timeline.** Step 0: 3 days. Model 1: 2 days. Model 2: 6 days. Model 3: 5 days. Cross-cutting and
tables: 3 days. About three weeks, with the model 1 result available on day 5.
