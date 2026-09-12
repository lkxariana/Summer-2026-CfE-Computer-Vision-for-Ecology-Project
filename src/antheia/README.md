# antheia

The installable package (`pip install -e .` from the repo root).

| module | role |
|---|---|
| `paths.py` | every external location, from `ANTHEIA_DATA` / `ANTHEIA_RUNS` |
| `store.py`, `globi.py`, `taxonomy.py`, `pairs.py` | the species universe, its features and the edge list |
| `bundle.py`, `metrics.py`, `negpool.py` | the artifact contract, ranking metrics with bootstraps, the pollinator negative pool |
| `models/` | `rgcn` (the retriever), `fusion` (the identity re-ranker), `embednet` / `pairnet` / `neural` / `twotower` (neural comparison models) |
| `baselines/` | the REGISTRY of comparison methods (see its README) |
| `eval/` | `ladder` (one retriever run), `rerank` (retrieve-then-re-rank), `localnets` / `localnets_rerank` (within-site completion), `report` (paired bootstraps), `tables`, `tables_appendix`, `tables_localnet`, `figures`, `fig_data`, `marginalisation_*`, `site_time`, `verify_splits` |

Run a model row with `scripts/run_arm.py configs/arms/<arm>.json`; build tables with `python -m antheia.eval.tables`.
