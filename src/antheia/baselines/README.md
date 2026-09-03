# Comparison methods

One module per method. Each implements `Baseline` from `base.py`:

```python
b = REGISTRY["congeneric"]().fit(train_edges, store)
scores = b.score_plant(plant_index)      # one score per pollinator in the candidate universe
```

Every class carries its own citation and an explicit `cold_start` flag. Methods with
`cold_start = False` cannot score a plant that has no training edges; they are reported on the warm
subset only, as an upper reference on what an edge-requiring method can achieve, and are not
comparable to the cold-start rows.

| module | method | reference | cold start |
|---|---|---|---|
| `popularity` | rank by pollinator degree | Aiyappa et al. 2025, ICML | ✓ |
| `cooccurrence` | shared occupied cells | Blanchet et al. 2020, Ecol Lett 23:1050 | ✓ |
| `abundance` | abundance product neutral model | Vázquez et al. 2009, Ecology 90:2039 | ✓ |
| `congeneric` | partners of taxonomic relatives | cf. Strydom et al. 2022 | ✓ |
| `phenology_likelihood` | abundance × phenological overlap | Vizentin-Bugoni et al. 2014, Proc R Soc B | ✓ |
| `svd_taxonomic` | low-rank factorisation + taxonomic imputation | Strydom et al. 2022, MEE 13:2308 | ✓ |
| `kron_krr` | two-step Kronecker kernel ridge regression | Stock et al. 2021, Ecol Modelling 451 | ✓ |
| `pair_gbm` | gradient boosting on pair features | Pichler et al. 2020, MEE 11:281 | ✓ |
| `lightfm` | hybrid factorisation, WARP loss | Kula 2015, arXiv:1507.08439 | ✓ |
| `bpr_mf` | Bayesian personalised ranking | Rendle et al. 2009, UAI | ✗ |

Optional dependencies: `lightfm`, `implicit`. Methods requiring them raise on import inside `fit`,
so the rest of the suite runs without them.

**Store contract.** A baseline receives a feature store exposing `plants`, `polls`, `idx_plants`,
`idx_polls`, occupancy matrices `F`/`P`, annual curves `FC`/`AC`, range sizes `Frs`/`Prs`, and the
co-occurrence matrix `N_full`. Anything derived from training edges must be computed in `fit` — never
read from the store — or held-out plants will leak.
