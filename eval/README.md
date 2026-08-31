# eval

- `verify_splits.py` — split-stability analysis: repeated 15% plant holdouts vs 3-fold vs 5-fold grouped CV, with within-split bootstrap CIs. Justifies the single-frozen-test-set protocol.
- `run_baselines.py` — corrected baselines on the frozen `artifacts/split_v1.json` plant split: null models (degree, N-only, range-only) plus the five ANTHEIA feature sets, reporting PR-AUC and per-plant recall@k with bootstrap CIs.

Run with the lab env: `/home/cher/miniconda3/envs/donuts/bin/python3 eval/run_baselines.py`.
