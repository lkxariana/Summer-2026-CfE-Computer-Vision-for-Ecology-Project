# ANTHEIA

Predicting plant–pollinator interactions for species that have **no interaction records at all**, in either
direction. Most pairs that exist in nature have never been observed: GloBI covers 11,031 plants x 13,124
pollinators for the continental US, and most of those species have few or no partners on file. A useful model has
to work for a plant with no records, for a pollinator with no records, for both at once, and inside a surveyed
community where the species list is known but the interaction list is incomplete. This repository holds the
model, the comparison set, and one evaluation protocol that covers all four cases at network prevalence.

## The model

Interaction = **opportunity** (the two species are in the same place at the same time) x **affinity** (they are
the kind of species that interact, given that they meet). The model computes both and keeps them separable.

- *Affinity* — a relational GNN (R-GCN, 2 layers, d = 128) over species nodes, each carrying a frozen BioCLIP-2
  text embedding of its name through one shared 768->128 projection, and cell x month nodes carrying a frozen
  joint-field coordinate encoding. Edges are training interactions and presence-weighted occurrences. There are
  no genus or family nodes; the text already carries taxonomy.
- *Opportunity* — one scalar per pair, `c(p,q) = z-score of log(1 + u_p . u_q)`, where `u` is a 256-D SVD
  projection of the species' presence surface, so the dot product is the expected number of cell-weeks in which
  both species are present. It joins `[h_p, h_q, h_p*h_q, |h_p-h_q|]` in the pair head.
- *Symmetric leave-own-edges-out* — every epoch, 30% of plants **and** 30% of pollinators lose their interaction
  edges before the forward pass, so the model rehearses cold species on both sides. The plant-only version of
  this collapses to identity-free nulls the moment a pollinator is unseen.
- *Re-ranker* — a single-stream transformer over `[CLS, plant name, candidate name]` (its own 768->192 projection
  of the frozen names) re-scores the retriever's top-500; its output is added to the retriever score.
- *The switch* — inside a surveyed site every candidate pair is co-present by construction, so the opportunity
  term carries no information there and is held at its mean (`pair_stat_at_inference: false`). Same weights, one
  inference-time flag, chosen by the task.

Loss: sampled softmax with logQ correction plus BCE. The arm file is [`configs/arms/final.json`](configs/arms/final.json)
and the bundle name is `M6.0_system_factorised`.

## Results

Validation, three seeds {42, 0, 1}, AUPR at network prevalence. Full tables with AUROC and normalised recall@10:
[`results/tables_publish.md`](results/tables_publish.md).

| | cold plant | cold pollinator | cold both | warm | within sites |
|---|---:|---:|---:|---:|---:|
| **ANTHEIA (opportunity x affinity)** | **0.230** | **0.162** | **0.142** | 0.051 | 0.217 |
| Best comparison model | 0.144 (Wide & Deep) | 0.044 (pair-feature GBM) | 0.044 (ANTHEIA v1) | **0.105** (SVD + taxonomic imputation) | 0.222 (congeneric transfer) |
| Chance | 0.0013 | 0.0023 | 0.0029 | 0.0003 | 0.135 |

On the warm split SVD + taxonomic imputation leads on AUPR while our model leads on AUROC. Within sites the
three leading methods are a tie: mean AUPR 0.217 for ours against 0.222 for congeneric transfer and for our own
earlier feature-engineered trees; on precision@L ours is 0.249 against 0.239 and 0.252. Per-network detail:
[`results/tables_localnet.md`](results/tables_localnet.md).

## Reproduce the tables

```bash
pip install -e .                             # the antheia package (src/antheia)
export ANTHEIA_DATA=/scratch/cher/antheia-data   # the frozen inputs; see src/antheia/paths.py
export ANTHEIA_RUNS=/path/to/runs                # optional, defaults to ./runs

scripts/run_arm.py configs/arms/final.json   # 4 regimes x 3 seeds + within-site; skips anything already bundled
python -m antheia.eval.tables                # -> results/tables_publish.md
python -m antheia.eval.tables_localnet       # -> results/tables_localnet.md
python -m antheia.eval.tables_appendix --split cold_plant/val   # -> results/tables_ladder_*.md
```

`run_arm.py` is the only launcher. Each (split, seed) runs as its own subprocess, so GPU memory is released
between runs and one crash does not stop the rest. Useful flags: `--splits cold_plant`, `--seeds 42`,
`--part test`, `--no-local`, `--only-local`, `--gpu 1`, `--dry-run`. Comparison models need no arm file:

```bash
python -m antheia.eval.ladder --model svd_taxonomic --split cold_plant --seed 42
```

Tables are always built from bundles under `runs/`, never from a model in memory.

## Repository layout

```
src/antheia/            installable package
  paths.py              every location outside the repo (ANTHEIA_DATA, ANTHEIA_RUNS)
  store.py              loading the built matrices, edges, splits
  bundle.py             the artifact contract: one bundle per (model, config, split, seed)
  metrics.py            AUPR at prevalence, AUROC, recall@k
  negpool.py            the pollinator negative pool (protocol invariant)
  models/               rgcn.py, fusion.py (re-ranker), embednet.py, pairnet.py, neural.py, twotower.py
  baselines/            REGISTRY of comparison models (popularity, congeneric, SVD+taxonomic, pair GBM, ...)
  eval/                 ladder.py (one retriever run), rerank.py (retrieve-then-re-rank),
                        localnets.py / localnets_rerank.py (within-site), report.py (paired bootstraps),
                        tables.py / tables_appendix.py / tables_localnet.py, figures.py, fig_data.py,
                        marginalisation_test.py, marginalisation_surfaces.py, site_time.py, verify_splits.py
scripts/run_arm.py      the only launcher (see its docstring)
configs/arms/           one JSON per model row: final.json, v1_*, v2_*, v3_*, ablations/*.json
                        README.md explains the keys
configs/pipelines.yaml  heavy paths for the input pipelines
pipelines/              everything that builds inputs
  network/              edges, canonical nodes, modelled universe, splits, local networks
  features/             text and image embeddings, surfaces, tokens, field embeddings, caches
  sdm/                  pollinator occurrence field models (joint field, deliverable curves)
  ppe/                  plant flowering-opportunity surface (e98 backbone)
  run_all.sh, config.py, README.md
data/splits/            frozen entity-disjoint splits (committed)
data/network/           edges, nodes, local networks (built, gitignored except modelled_universe.json)
data/features/          matrices, surfaces, embeddings (built, gitignored)
runs/                   bundles (gitignored; the experimental record)
results/                tables_publish.md, tables_ladder_*.md, tables_localnet.md, figures/
tests/                  test_protocol.py, test_invariants.py, test_edges.py, test_splits.py, test_features.py
docs/paper/             outline.md (the current story) and the methods sources
docs/plan/              the design documents
docs/archive/           superseded plans and the v1 write-ups
legacy/v1-notebooks/    the July 2026 notebook pipeline, kept for reference, not maintained
EXPERIMENTS.md          the lab notebook: one dated entry per result
```

Python on the lab machine `crow`: `/home/cher/miniconda3/envs/donuts/bin/python3`. Two GPUs are available;
`run_arm.py --gpu` selects one.

## Evaluation protocol

Four entity-disjoint regimes, frozen and degree-stratified, plus within-site completion:

| regime | queries x candidates |
|---|---|
| cold plant | 663 plants x 13,124 pollinators |
| cold pollinator | 1,927 x 1,312 |
| cold both | 249 x 1,312 |
| warm | 10% of edges held out, filtered ranking |
| within sites | 91 surveyed networks, every local pair removed from training |

Primary metric is AUPR at network prevalence, with AUROC and normalised recall@10 alongside. Within sites an
absent pair among the surveyed species counts as an observed non-interaction, and precision@L and NODF are
computed from the top-L pairs (L = observed links). Three seeds, {42, 0, 1}.

Two protocol invariants, both tested in [`tests/test_protocol.py`](tests/test_protocol.py):

1. **Held-out pollinators never enter training** — not as sampled negatives, and not as the re-ranker's hard
   negatives. Every pollinator negative is drawn from `antheia.negpool`. Without this the re-ranker is trained to
   call the evaluated pollinators negative and the split means nothing.
2. **Warm evaluation uses filtered ranking** — an evaluated plant's known partners are excluded from the
   negatives (`evaluate_scores(..., exclude=)`), removed from pooled metrics and ranked below every candidate.

Everything reported is validation. The test split is held and has not been scored.

## Data

No model inputs live in this repository. They sit under `$ANTHEIA_DATA` (default `/scratch/cher/antheia-data`):
text embeddings, the pollinator SDM and joint presence field, plant occurrences, and the PPE flowering-opportunity
surface. Every one of those locations is declared in [`src/antheia/paths.py`](src/antheia/paths.py) — add new ones
there rather than writing a path into a module.

Committed: the frozen splits in `data/splits/`, `data/network/modelled_universe.json`, the tables and figures in
`results/`, the configs, the code. Never committed: `runs/`, `logs/`, `data/features/`, built parquet, `*.npz`,
`*.pt`, and any raw data.

## Status

Validation only. The test split and the prospective holdout are pending, to be run once at the end. The current
story and the figure and table plan are in [`docs/paper/outline.md`](docs/paper/outline.md); the full experimental
record, one dated entry per result, is in [`EXPERIMENTS.md`](EXPERIMENTS.md).

## Citation / authors

ANTHEIA — Multimodal Vision Research Laboratory (MVRL), Washington University in St. Louis.
Kexing Li, Dan Cher, Nathan Jacobs.

> Citation to be added on release.
