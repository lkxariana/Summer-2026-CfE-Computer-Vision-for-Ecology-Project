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

`--head text` swaps the learned species-ID embedding for a projection of the frozen BioCLIP-2
text embedding; `--extra-species taxa.pt` (built by `species_text.py --names taxa.json`) then
emits zero-shot curves (`species_id = -1`, `part_zs_*.parquet`) for taxa with no observations.
Prompt = the raw binomial string, the exact format `species_embeddings_v2.pt` was built with.
The species_matrix embedding path is numerically identical to the id path (verified 0.0 diff).

```bash
python pipelines/ppe/export_embeddings.py   # species_static.npz [n_vocab,576] + grid_zdyn_null.npz [K,192]
```

exports the two cacheable e98 embeddings: the per-species z_static (constant over space/time —
the phenology-model species embedding) and the species-free per-(cell,week) z_dynamic under the
trained null-species conditioning. The dense per-species z_dynamic grid (~TB) is deliberately
not cached — re-encode on demand.

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
