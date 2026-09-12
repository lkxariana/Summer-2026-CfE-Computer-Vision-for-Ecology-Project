# Arm files

One JSON per model row in the paper. `scripts/run_arm.py <arm.json>` runs every (regime, seed) and the within-site
evaluation that does not already have a bundle under `runs/`, so re-running is free and the tables are always built from
bundles (`python -m antheia.eval.tables`).

| key | meaning |
|---|---|
| `name` | bundle name (directory under `runs/`; the row key in the table generators) |
| `retriever_model` | REGISTRY key (`rgcn`, `embednet`, `popularity`, ...) |
| `retriever` | kwargs of that model; `field_dir` defaults to `antheia.paths.FIELD_DIR` |
| `reranker` | FusionConfig kwargs; present = retrieve-then-re-rank system, absent = retriever alone |
| `within_site_retriever_overrides` | retriever kwargs that change for the within-site task (the final model holds the opportunity term at its mean: `pair_stat_at_inference: false`) |
| `within_site_name` | bundle name for the within-site run when it differs (retriever-alone rows scored affinity-only) |

`final.json` is the paper's model. `v1_*`, `v2_*`, `v3_*` are the earlier systems kept as rows; `ablations/` holds every
Table 3 row. Baselines have no arm files: `python -m antheia.eval.ladder --model <key>` with the default config is the row.
