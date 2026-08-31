# antheia

Importable pipeline package. `config.py` resolves paths from `configs/*.yaml` (env `ANTHEIA_DATA_ROOT` overrides the data root). `globi.py` builds the orientation-corrected edge list. `store.py` loads all species artifacts aligned to the coverage universe and computes N/Δ features. `pairs.py` holds plant-grouped splits and negative samplers. `models.py` is the standardized LR probe over named feature specs; `metrics.py` the ranking/pair metrics and bootstraps.

Install: `pip install -e .` (or add `src/` to `sys.path` as the scripts do).
