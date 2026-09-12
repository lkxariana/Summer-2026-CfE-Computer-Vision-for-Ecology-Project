# CLAUDE.md

Guidance for Claude Code sessions in this repository.

## What this is

ANTHEIA (MVRL, WashU; Kexing Li, Dan Cher, Nathan Jacobs) predicts plant–pollinator interactions for species with
no interaction records, in either direction. The final system is "opportunity x affinity": a species-node R-GCN
retriever with a co-presence scalar, plus an identity re-ranker on the top-500. Read `README.md` first, then
`docs/paper/outline.md`.

## Layout and the one launcher

- `src/antheia/` is the package (`pip install -e .`): `paths.py`, `store.py`, `bundle.py`, `metrics.py`,
  `negpool.py`, `models/` (rgcn, fusion, embednet, pairnet, neural, twotower), `baselines/` (the REGISTRY of
  comparison models), `eval/` (ladder, rerank, localnets, localnets_rerank, report, tables*, figures, ...).
- `pipelines/` builds every input (`network/`, `features/`, `sdm/`, `ppe/`); see `pipelines/README.md`.
- **`scripts/run_arm.py` is the only launcher.** `scripts/run_arm.py configs/arms/final.json` runs every regime x
  seed plus the within-site evaluation that has no bundle yet, one subprocess each. Read its docstring before
  adding anything. A model row is a JSON in `configs/arms/` (`configs/arms/README.md` documents the keys);
  comparison models need no arm file: `python -m antheia.eval.ladder --model <registry key> --split cold_plant --seed 42`.

## The bundle contract

Every run is one bundle per (model, config, split, seed) under
`runs/<name>/<config_hash>/<split>/<part>/s<seed>/` (within-site runs drop `<part>`: `.../localnet/s<seed>/`),
holding `scores.npy`, `Y.npy`, `per_query.parquet`, `config.json`, `metrics.json`. **Tables and bootstraps are
built from bundles and nothing else** (`python -m antheia.eval.tables`, `tables_localnet`, `tables_appendix`).
Never hand-edit a results table; regenerate it.

The bundle git stamp is taken at write time, so a bundle can carry a commit newer than the code that produced it —
identify stale bundles by **start time = `config.json:time` minus `metrics.json:wall_s`**, not by the stamp.

## Environment

- Interpreter: `/home/cher/miniconda3/envs/donuts/bin/python3`. Nothing else has torch.
- Two GPUs on the machine `crow`; pick one with `run_arm.py --gpu 0|1` and check for a running job first.
- Data root: external inputs live under `$ANTHEIA_DATA` (default `/scratch/cher/antheia-data`). **Add or change
  locations in `src/antheia/paths.py`; never hardcode a `/scratch` path in a module or script.**

## Protocol invariants (tested in `tests/test_protocol.py`)

1. **Never let held-out pollinators into any negative pool, including the re-ranker's hard negatives.** Every
   pollinator negative is drawn through `antheia.negpool`; the split is set from the training pollinators.
2. **Warm evaluation uses filtered ranking**: an evaluated plant's known partners are excluded from the negatives
   via `evaluate_scores(..., exclude=)`, removed from pooled metrics and ranked below every candidate.

Three seeds {42, 0, 1}. Everything reported is validation; the test split is held and is not to be scored without
Dan's say-so.

## Deciding whether a change is real

Pre-registered adoption rule: **adopt a change only if within-site mean AUPR rises by >= 0.01 with no universe
AUPR loss > 0.01 and every current lead retained; three seeds.** A change that wins on one regime and loses
elsewhere is an ablation row, not the system.

**Never decide on one-epoch smoke runs.** Smoke runs check that code executes; they rank arms wrongly.

## Operational notes

- **Never put a literal process name in a `pkill`/`pgrep` pattern** — it matches the watcher itself and kills the
  session. Use file-based waits (poll for `metrics.json`) and the bracket trick (`pgrep -f "[p]ython.*ladder"`).
- Run long jobs detached with output to `logs/<name>.log` and follow with `grep --line-buffered`.
- Commit often, and **never commit `runs/`, `logs/`, `data/features/`, `*.npz`, `*.pt`,** or raw data. The frozen
  splits, `data/network/modelled_universe.json`, and the generated tables and figures in `results/` are committed.
- `EXPERIMENTS.md` is **append-only**: one dated entry per result, with the command, the numbers, and what it
  settled. Add the entry when the result lands, not later.
- `docs/paper/outline.md` is the paper's current story — update it when the story changes, and keep the table
  numbers there consistent with `results/tables_publish.md`.

## History, not current practice

The v1 notebook world lives in `legacy/v1-notebooks/` and is not maintained. The GloBI orientation bug (all three
interaction types are pollinator->plant; the old notebooks renamed `sourceTaxonName -> plant_species` without
swapping, which left 139 positive pairs and an evaluation that an N-only model won at 0.99 ROC-AUC) is fixed and
gone. Both are recorded in `docs/archive/`. Do not resurrect either; do not cite v1 numbers except as the
"ANTHEIA v1" comparison rows produced by the current harness.

Git: work happens on branch `pipelines-port` (default branch is `main`; origin is
`lkxariana/Summer-2026-CfE-Computer-Vision-for-Ecology-Project`).
