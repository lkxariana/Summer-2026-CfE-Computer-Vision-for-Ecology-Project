# pipelines/

Regenerates the two model-derived inputs for the interaction models: the plant flowering
opportunity surface (frozen **e98** phenology backbone + multi-species SINR) and the pollinator
activity curves (temporal SINR head + zero-shot LE-SINR hybrid).

All heavy paths live in `configs/pipelines.yaml`; override any entry with `PIPELINES_<KEY>`.
Model assets (e98 checkpoint + code, vocab, grid inputs) fall back to the private HF repos
`dcher95/phenofield-e98` and `dcher95/antheia-pollinator-sdm` when the local path is missing,
so on a new machine only the HF token and the big covariates (climplicit lookup, e98 feature
caches, PRISM parquet) need providing.

## Plant side (GPU recommended for full scale)

```bash
python pipelines/ppe/build_prism_weekly.py            # only if the grid changed; else the HF copy is used
python pipelines/ppe/generate_opportunity.py          # one parquet per species -> <out_dir>/opportunity_surface/
```

`generate_opportunity.py --species-file data/network/modelled_universe.json` restricts to the
modelled universe; `--max-species N` for validation runs. Resumable (skips existing parts).

## Pollinator side

```bash
python pipelines/sdm/build_occ.py                     # geo-prior extract + locked list -> pollinator_occ.npz
python pipelines/sdm/text_embeddings.py --set occ     # BioCLIP-2 text conditioners
python pipelines/sdm/text_embeddings.py --set zeroshot
python pipelines/sdm/build_deliverable.py --device cuda   # curves + weights -> <sdm_data>/deliverable_v2/
```

`build_deliverable.py --smoke` runs 1 epoch / 40 species into a `smoke/` subdir. `--universe`
restricts species to `modelled_universe.json` pollinators. The first `build_deliverable` or
`build_occ` run after a new occurrence set pays a one-time climplicit gather (33 GB lookup,
needs ~40 GB RAM); it is cached to `<sdm_data>/clim_gathered.npz`.

## e98 loading rule

The checkpoint must be built from the bwei model code (`vendor/bwei_ppe` in the PhenoField repo,
mirrored under `models/` in the HF repo). The cherd-tree `CrossModalVAE` silently drops
`dynamic_encoder.null_species_emb` — the parameter e98's species dropout trains.
`pipelines/ppe/backbone.py` enforces a clean (0 missing / 0 unexpected) load.
