# Consolidation & migration plan — HPC → crow

Goal: consolidate everything onto **crow** (the machine the ANTHEIA repo targets), so the
TODO.md caches can be built for the new 12,347 × 15,921 universe, and so the two model-derived
inputs (PPE opportunity surface, pollinator SDM activity curves) can be **regenerated or
retrained there** whenever the universe or models change. Drafted 2026-09-03 from a full audit
of both machines' code and data; awaiting Dan's approval.

---

## 1. Where things stand today

**This machine (HPC, `/projects/bdbl/`)** has all the *model-side* assets and none of the
*network-side* ones:
- PhenoField repos + checkpoints (PPE-L `FVA_v4_b2_supcon_cc_k10_bioclip_q_PPE_L`, legacy e68)
- Frozen feature caches, PRISM parquet, Climplicit lookup, AlphaEarth grid
- Pollinator SDM training data + the built deliverable (`Pollinator_SDM_Deliverable/`, 1,617 spp)
- PPE opportunity surface (8.8 GB, 6,697 spp) and all the temporal-SDM experiment code
  (`Plant_Pollinator_Initial_Analysis/PhenoField_Opportunity/`)

**crow (`/scratch/ariana.l`, `/scratch/cher`)** has all the *network-side* assets and none of
the model-side ones:
- The verified network parquets (`edges.parquet` etc. — **only on crow**; this machine has just
  the logs), the pinned GloBI snapshot, the Stage-4/5 feature CSVs, `artifacts/cache/`,
  the TreeOfLife-200M embedding caches
- Copies of the two deliverables (the new repo's `build_sdm_surfaces.py` / `build_plant_surfaces.py`
  read them from `/scratch/ariana.l/Stage 6 Seed Testing/` and `/scratch/ariana.l/ppe-outputs/`)

So the transfer is one-directional: move the **regeneration/retraining kit** from HPC to crow.
Nothing network-side needs to come back here.

Two existing distribution channels make this easier than raw rsync:
- **GitHub** `dcher95/PhenoField`, branch `production` ("make inference work off-cluster") is
  pushed and in sync with the local fork. It contains `utils/paths.py` (`PHENOFIELD_DATA` env
  rebasing — exactly the mechanism needed, including for the `species_embedding_path` baked
  inside checkpoints) and `vendor/bwei_ppe/` (the correct CrossModalVAE for e68-style ckpts).
- **HuggingFace** `dcher95/phenofield` (dataset) already hosts the Pheno3M train/val/test
  parquets + `species_vocab.json`. The local clone at `/projects/bdbl/cherd/data/phenofield`
  just has the extra files (PRISM parquet, embeddings) sitting *untracked* next to it.

---

## 2. What the TODO actually needs from this machine

Mapping TODO.md items to sources — most of item 2 derives from artifacts that are already on
crow; the HPC is needed only for **regeneration ability**:

| TODO item | source | already on crow? |
|---|---|---|
| 2.1 `F.npy` / 2.3 `FC.npy` / 4.8 plant surfaces | PPE opportunity surface parquets | yes (`/scratch/ariana.l/ppe-outputs/opportunity_surface/`) |
| 2.2 `P.npy` / 2.4 `AC.npy` | `pollinator_observations_v2.csv` (GBIF) | yes (`/scratch/ariana.l/Plant Pollinator Initial Analysis/`) |
| 4.9 pollinator per-cell surfaces | SDM deliverable parquet + manifest | yes (`/scratch/ariana.l/Stage 6 Seed Testing/`) |
| 2.5–2.7, 3.x, 4.1–4.7, 4.11–4.14 | derived / HF downloads | n/a |
| 4.10 / 7.1 phenology-model embeddings | PPE-L checkpoint (read-off, see §6) | **no — transfer** |
| regenerate opportunity surface (new universe / model change) | PPE-L stack (§3 tier 1a) | **no — transfer** |
| retrain / extend pollinator SDM (TODO 7.3 coverage gap) | SDM stack (§3 tier 1b) | **no — transfer** |

⚠️ Before relying on the crow copies: verify the two deliverables there are byte-identical to
the HPC masters (`pollinator_activity_curves.parquet` 529 MB, opportunity surface 6,697 part
files / 8.8 GB, `flowering_curves_all.parquet` 114 MB — the SDM's `cell_idx` is aligned to the
latter, so they must travel as a set).

---

## 3. Transfer manifest (HPC → crow)

### Tier 1a — PPE (plant) regeneration kit, ~26 GB

| file (HPC path) | size | why |
|---|---|---|
| `PhenoField/checkpoints/FVA_v4_b2_supcon_cc_k10_bioclip_q_PPE_L/last.ckpt` | 518 MB | the frozen PPE-L backbone every extractor loads |
| `data/phenofield/embeddings/species_embeddings_v2.pt` | 20 MB | baked into ckpt config; model won't construct without it |
| `data/phenofield_multi_365/species_vocab.json` | 203 KB | species↔id (already on HF too) |
| `data/phenofield_cache/FVA_v4_b2_supcon_cc_k10_bioclip_q_PPE_L/` (4 npz + meta) | 13.3 GB | frozen features; `generate_opportunity.py` trains its SINR on these |
| `data/phenofield/hf_prism_365.parquet` | 11 GB | rebuild `prism_weekly.npz` if the grid ever changes |
| `…-old/…/Part2_PhenoField_Outputs/data/prism_weekly.npz` (+ `_peryear`, `coverage.csv`) | 540 MB | skip Stage A entirely while grid is frozen |
| `data/conus_eval_grid/alphaearth_2017.parquet` | 4.6 MB | per-cell AlphaEarth |
| `…-old/data/grid_centroids_0.5deg.csv`, `plant_flowering_events.parquet` | 4.2 MB | grid + labels |

### Tier 1a′ — e98 phenology backbone kit (~34 GB) — **Dan: use e98, not e68**

The phenology foundation model going forward is **`e98_large_specdrop_dyn4l`** (adaln:
`dynamic_encoder_type: adaln_dit`, 576+192 dims like e68, plus species dropout 0.1→0.3 —
the model behind the "e98_adaln" name). The other e98 variant, `e98_e83_dyn_rnc_cyclic`
(60 MB, dynamic-only head, no train cache), is not it.

| file (HPC path) | size | why |
|---|---|---|
| `/projects/bdbl/bwei/PhenoField/logs/e98_large_specdrop_dyn4l/version_0/checkpoints/last.ckpt` (+ `hparams.yaml`) | 446 MB | the backbone |
| `data/phenofield_cache/e98_large_specdrop_dyn4l/inat_train.npz` | 28 GB | frozen train features — **e98 has a full train cache** (e68 never did), so probes/SINRs train on the proper split |
| `…/inat_test_{random,spatial,species}.npz` | 6 GB | eval splits |
| `data/phenofield/embeddings/bioclip_v2/` | 4.7 GB | only if caches will be *re-extracted* from raw obs on crow (dataloader dep); skip if transferring the cache |

Baked-in checkpoint paths (`species_embedding_path` → `species_embeddings_v2.pt`,
`dataset_dir` → `phenofield_multi_365`, `embedding_dir` → `bioclip_v2`) are handled by
`utils/paths.py` rebasing on the `production` branch.

**Loading caveat (matters more for e98 than e68):** cherd's own `models/cross_modal_vae.py`
silently drops `field_branch.dynamic_encoder.null_species_emb` under `strict=False`. e98 is
the *species-dropout* model — that null-species embedding is load-bearing. Always load e98
through `vendor/bwei_ppe/` (the production branch documents this exact failure).

Open sub-question: does e98 replace only the e68 role (flowering-curve/feature extraction), or
also PPE-L inside `generate_opportunity.py`? The opportunity SINR trains on whatever feature
cache it's pointed at, and e98 has the full train cache PPE-L's pipeline had to work around —
so swapping the cache path is a one-line config change if desired. If e98 replaces PPE-L
everywhere, tier 1a shrinks by ~13.3 GB (drop the PPE-L cache) and the PPE-L ckpt becomes
archive-only.

### Tier 1b — pollinator SDM retraining kit, ~5 GB (+33 GB conditional)

| file (HPC path) | size | why |
|---|---|---|
| `data/pollinator_sdm/pollinator_occ.npz` | 51 MB | 3.3 M CONUS occurrences, 2,994 taxa — the training set |
| `…-old/data/pollinator_species_text.pt`, `data/pollinator_sdm/zeroshot_text.pt` | 11 MB | BioCLIP-2 text embeddings (re-derivable from HF hub) |
| `…-old/data/pollinator_species_locked.csv`, `data/pollinator_sdm/groupB_zeroshot_species.csv` | 133 KB | locked species lists |
| `sinr/data/train/geo_prior_train.csv` + `geo_prior_train_meta.json` | 3.5 GB | rebuild `pollinator_occ.npz` for an expanded species list |
| `…-old/data/gbif_download_pollinators.py`, `gbif_qc_report.py` | 10 KB | the scripts for the TODO-7.3 new GBIF download |
| **`data/phenofield/embeddings/climplicit/climplicit_monthly_lookup.pt`** | **33 GB** | climate covariate for train *and* inference — see note |
| `data/pollinator_sdm/clim_gathered.npz` | 6.3 GB | optional: cached KD-tree gather; re-derivable from the lookup (~1 job) |
| `…/Pollinator_SDM_Deliverable/model_*.pt`, `model_standardization.npz`, `predict_pollinator_activity.py` | 9 MB | current weights + standalone inference (retrain is ~2 min on an A100 anyway) |

**Climplicit note.** The 33 GB lookup is the single biggest item and is needed for any SDM
retrain or any run of `extract_ppel_field.py`. Two options: (a) transfer once (recommended —
exact, one-time), or (b) regenerate on crow from the public Climplicit checkpoint (saves the
transfer, costs a script + GPU time, risks tiny numeric drift). If pollinator coverage is
expanded (TODO 7.3) the full lookup is required — `clim_gathered.npz` only covers the current
3.3 M points.

### Tier 2 — verify-then-skip (should already be on crow)

Opportunity surface (8.8 GB), `Pollinator_SDM_Deliverable/` (530 MB),
`flowering_curves_all.parquet` (114 MB), `stage4_globi_conus_broad.csv` (77 MB),
`pollinator_observations_v2.csv` (997 MB). Transfer only if the crow copies are missing/stale.

### Tier 3 — recommend **not** transferring (archive on HPC / tape)

- **Legacy e68 pipeline** (e68 ckpt 444 MB, e68 caches 5.9 GB, `grid_field_features_all.npz`
  14 GB). Superseded by the PPE-L opportunity surface; RESULTS.md documents why the probe
  readout was abandoned ("muted curves"). Its one live artifact, `flowering_curves_all.parquet`,
  is kept (tier 2) purely as the `cell_idx` grid reference. The four Part-2 stage scripts only
  exist in git history anyway (`git show f50d506:…`; `run_all_cells.sbatch` is lost entirely).
- Experiment checkpoints (19 GB `PhenoField/checkpoints/`, minus PPE-L), `results/` 167 GB,
  `phenofield_cache/` other ~100 model dirs (253 GB), raw GBIF dump (9.1 GB — the derived
  `_v2.csv` suffices; a *new* download supersedes it anyway).
- Experiment npz's in `Part2_PhenoField_Outputs/data/` (`ppe_neg_*`, `ppe_querysp_*`,
  `bonap_*`, ~7 GB) — inputs to already-published ablation tables; regenerable from tier 1a.

**Totals: ~65 GB must move (~98 GB with Climplicit), vs ~580 GB of PhenoField bulk that should
not. Drops to ~85 GB total if e98 replaces PPE-L outright (PPE-L cache not needed).**

---

## 4. Packaging: HuggingFace for durable assets, rsync for the rest

Recommended split (all under the existing `dcher95` account, private):

1. **HF model repo `dcher95/phenofield-ppe-l`** (~550 MB): PPE-L `last.ckpt`, `hparams`,
   `species_embeddings_v2.pt`, `species_vocab.json`, a `README` with the `forward_field`
   contract (inputs, 576+192 output, the `doy=None` convention). Load path on any machine:
   `pip install git+https://github.com/dcher95/PhenoField@production` + `PHENOFIELD_DATA=…`.
   → makes "re-query the model" a two-line setup anywhere, not a machine.
2. **HF dataset `dcher95/phenofield` (existing)** — add the currently-untracked inputs that
   conceptually belong to it: `hf_prism_365.parquet`, the Climplicit lookup (chunk to <20 GB
   shards or keep as one file; HF's per-file ceiling is 50 GB), the PPE-L feature cache npz's.
3. **HF dataset `dcher95/antheia-inputs`** (new, ~6 GB): pollinator occ npz, text embeddings,
   locked lists, `geo_prior_train.csv(.gz)`, grid files, SDM weights + standardization. This is
   the repo that gets a new *version tag* each time the modelled universe changes.
4. **Deliverables** (opportunity surface, activity curves): keep as files on crow, optionally
   mirrored to `antheia-inputs` once the new-universe versions are built — versioning these on
   HF is genuinely useful because they will be rebuilt.

Direct `rsync`/Globus HPC→crow works too and is faster for the one-time bulk; the reason to
route the durable core (items 1–3) through HF anyway is the stated requirement: *easy to
recreate caches or re-query models after changes*, from whichever machine survives. Suggest:
HF for 1–3, plain rsync for tier-2 verification copies.

---

## 5. Code moves into the ANTHEIA repo

The cache-generation code should live in the new repo as a `pipelines/` (or `scripts/pheno/`)
package, ported from `PhenoField_Opportunity/` — **with paths driven by config, not constants**
(every current script hardcodes the pre-rename repo path and is broken as-is):

| new module | ported from (old repo `PhenoField_Opportunity/`) | function |
|---|---|---|
| `pipelines/ppe/build_prism_weekly.py` | same name (+ `_peryear`) | Stage A climate windows |
| `pipelines/ppe/ppel_field.py` | `extract_ppel_field.py` | `load_ppel`, `climplicit_by_cell` — the one place the checkpoint loads |
| `pipelines/ppe/generate_opportunity.py` | same name | **the plant deliverable** (multi-species SINR → partitioned parquet) |
| `pipelines/sdm/build_occ.py` | occ-building logic + `gbif_download_pollinators.py` | occurrence npz from GBIF/iNat, parameterized by species list |
| `pipelines/sdm/text_embeddings.py` | `precompute_pollinator_text.py`, `precompute_zeroshot_text.py` | BioCLIP-2 text vectors |
| `pipelines/sdm/build_deliverable.py` | `build_pollinator_deliverable.py` (+ `predict_pollinator_activity.py`) | **the pollinator deliverable** (head + LE-SINR hybrid) |
| `pipelines/common.py` | `common.py` | overlap, week/coord encodings |

Conventions for the port:
- One `configs/pipelines.yaml` (or extend `default.yaml`) with `phenofield_data`, `ppe_ckpt`,
  `climplicit_lookup`, `sdm_data`, `out_dir` — resolved the same way `ANTHEIA_DATA_ROOT`
  already works. No absolute paths in code. Also fix the ~12 hardcoded `/scratch/...`
  constants the audit found in existing `scripts/` and `eval/` files.
- **Species lists come from `data/network/modelled_universe.json`** (TODO item 1), so a
  universe change → rerun is one command per side.
- PhenoField becomes a pinned pip dependency (`git+…@production`), not `sys.path.insert` into
  a personal directory. Use its `utils/paths.py` rebasing; do not vendor unless crow can't
  reach GitHub.
- The 26 sbatch files stay behind (HPC-specific). On crow, plain shell / a small runner. Keep
  one `slurm/` escape-hatch template only if crow's GPU turns out to be insufficient (see §7).
- The experiment harnesses (`run_row6*`, `run_temporal_*`, BONAP, NPN, etc.) are *paper code*,
  not pipeline code — leave them in the old repo/archive; RESULTS.md already records their
  conclusions. (If ported later: `plot_npn.py` renders **retracted** numbers — do not carry
  it forward as-is.)

---

## 6. Answers to TODO §7 blockers (from the audit)

- **7.1 PPE species embeddings: yes, exportable.** `z_static` (576-d) is a deterministic
  function of `species_id` only — constant across cells/weeks. Export = one forward pass per
  vocab species (or read the species-projection of `species_embeddings_v2.pt`). `z_dynamic`
  (192-d, climate-driven) is the per-cell-week part.
- **7.2 SDM architecture: shared space exists.** The head model is one shared encoder
  (Fourier(loc) + Climplicit + Fourier(week) → 256-d) with per-species *linear* heads — so
  each pollinator has a 256-d weight vector in a common space, and LE-SINR gives text-derived
  embeddings in the same space for zero-obs species. It is *not* the same space as PPE's, so a
  symmetric plant↔pollinator embedding comparison needs a learned bridge, but both sides have
  well-defined species vectors.
- **7.3 coverage gap: needs a new GBIF download.** Current occ set = 2,994 taxa from the 2023
  iNat geo-prior snapshot. The download + QC scripts exist (`gbif_download_pollinators.py`);
  scope decision (all 15,921 network taxa vs the 6,217 species-rank ones) is Dan's call.
  Retraining after the download is cheap (~2 min A100 for the models; the expensive step is
  one KD-tree Climplicit gather over the new points).

---

## 7. Execution order

1. **Decide the open questions** (below), especially crow disk/GPU.
2. **Push/publish**: create HF model repo (PPE-L kit); add PRISM + Climplicit + PPE-L caches to
   `dcher95/phenofield`; create `antheia-inputs` with the SDM kit. (Or rsync equivalents.)
3. **Verify tier-2 artifacts on crow** against HPC checksums (`sha256sum` on both ends).
4. **Port pipeline code** into the ANTHEIA repo per §5, on a branch; smoke-test each stage on
   crow with `--cells used`-scale subsets before any full run.
5. **Then start TODO item 1** (`modelled_universe.json`) — the pipelines take it as input, so
   the port should land first.
6. Once new-universe deliverables are rebuilt on crow, archive the HPC copies (tier 3 +
   superseded outputs) and retire this machine from the project.

## GPU assessment: 2× RTX 4090 on crow is sufficient (Dan asked 2026-09-03)

Everything on the roadmap fits comfortably:
- **All models are small.** e98 446 MB / PPE-L 518 MB / SDM ~5 MB — inference and feature
  extraction batch-fit easily in 24 GB VRAM, and a 4090's fp16/fp32 throughput is in the same
  class as (often above) an A100 for models this size. Batch sizes tuned for A100-80GB may need
  halving; wall-clock impact is minor.
- **`generate_opportunity.py`** (the 14 h full-scale A100 job) is the worst case: expect the
  same order of magnitude on one 4090, and it shards trivially over species → **~half the
  wall-clock on both cards**.
- **SDM retrain is ~2 min**; probes and per-species SINRs are minutes.
- The binding constraints on crow are **host RAM and disk, not GPU**: the pollinator jobs
  allocated 96–160 GB RAM (the 33 GB Climplicit lookup is loaded into memory for the KD-tree
  gather) and e98's `inat_train.npz` is 28 GB (mmap-able). Crow needs ≥128 GB RAM for the
  current code as written (or a one-time chunked rewrite of the gather), plus ~100–150 GB free
  disk for tier-1 inputs + regenerated outputs.
- The only workload where the HPC still wins is a **full PhenoField backbone retrain**
  (multi-day, 381 GB Pheno3M locally) — not on the roadmap, and Pheno3M is on HF if it ever is.

**Recommendation: consolidate fully onto crow.** Keep one HPC sbatch escape-hatch template in
the repo until the first full-scale opportunity-surface run on crow has passed, then retire it.

## Open questions for Dan

1. **crow host RAM + free disk** (see GPU assessment — GPUs are fine; RAM ≥128 GB and
   ~100–150 GB disk are what to confirm).
2. **Climplicit**: transfer the 33 GB lookup (recommended) or regenerate on crow?
3. **e98 scope**: does e98 replace only e68's role, or also PPE-L inside
   `generate_opportunity.py`? (See tier 1a′ — one-line cache swap, and it removes the
   fit-on-test-split workaround.)
4. **e68 legacy pipeline**: OK to archive rather than port? (Recommended; §3 tier 3.)
5. **HF repos**: private under `dcher95`, or a lab org? (`geo_prior_train.csv` is derived from
   iNat data — keep private.)
6. **GBIF download scope** for the pollinator gap (7.3): all network taxa or species-rank only?
