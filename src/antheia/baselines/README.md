# Comparison methods

One module per method. Each implements `Baseline` from `base.py`:

```python
b = REGISTRY["congeneric"]().fit(train_edges, store)
scores = b.score_plant(plant_index)      # one score per pollinator in the candidate universe
```

Every method here is **cold-start capable**: it can score a plant with no training interactions,
which is the setting this work evaluates (Setting B of Stock et al. 2018, *Neural Computation*
30:2245–2283). Methods that cannot are excluded rather than reported with caveats — see below.

| module | method | reference |
|---|---|---|
| `popularity` | rank by pollinator degree | Aiyappa et al. 2025, ICML |
| `cooccurrence` | shared occupied cells | Blanchet et al. 2020, Ecol Lett 23:1050 |
| `abundance` | abundance product neutral model | Vázquez et al. 2009, Ecology 90:2039 |
| `congeneric` | partners of taxonomic relatives | cf. Strydom et al. 2022 |
| `phenology_likelihood` | abundance × phenological overlap | Vizentin-Bugoni et al. 2014, Proc R Soc B 281:20132397 |
| `svd_taxonomic` | low-rank factorisation + taxonomic imputation | Strydom et al. 2022, MEE 13:2308 |
| `pair_gbm` | gradient boosting on pair features | Pichler et al. 2020, MEE 11:281 |
| `lightfm` | hybrid factorisation, WARP loss | Kula 2015, arXiv:1507.08439 |

Optional dependency: `lightfm`. Imported inside `fit`, so the rest of the suite runs without it.

## Methods considered and not compared against

**Transductive factorisation and graph models** — Bayesian personalised ranking (Rendle et al. 2009,
UAI), latent factors with implicit feedback (Seo & Hutchinson 2018, AAAI-18), LightGCN (He et al.
2020, SIGIR), and message-passing models generally. These learn a free latent vector per node, so a
plant absent from training has no representation and receives no score. They are not applicable to
cold-start prediction and are omitted rather than reported on a different subset, which would not be
comparable.

**Kronecker kernel ridge regression** (Stock et al. 2021, *Ecological Modelling* 451:109508;
van Laarhoven et al. 2011). Applicable in principle, and the source of the four-setting evaluation
framework adopted here. Omitted for scale: the method requires dense species-by-species kernels on
both sides, which at this network's size is computationally impractical, and it tests no hypothesis
distinct from the low-rank and gradient-boosting baselines already included.

**Morphological trait matching** (e.g. Klumpers et al. 2012, *Oecologia*). Requires proboscis length
and corolla depth. Corolla depth is recorded for fewer than fifty species worldwide, so the
comparison cannot be made at continental scale and is reported as a limitation.

## Store contract

A baseline receives a feature store exposing `plants`, `polls`, `idx_plants`, `idx_polls`, occupancy
matrices `F`/`P`, annual curves `FC`/`AC`, range sizes `Frs`/`Prs`, and the co-occurrence matrix
`N_full`. Anything derived from training interactions must be computed in `fit` — never read from the
store — or held-out plants will leak.
