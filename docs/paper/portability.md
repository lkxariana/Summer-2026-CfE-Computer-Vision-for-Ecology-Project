# Moving this project to another machine

Two categories: **inputs that only exist on the lab machine** and must be copied, and **inputs that
are public** and should be re-fetched rather than shipped. Everything else is derived and rebuilds
from those.

---

## 1. Must be copied — lab-private (≈10.3 GB)

| data | path | size | needed for |
|---|---|---:|---|
| PPE opportunity surface | `/scratch/ariana.l/ppe-outputs/opportunity_surface/` (6,697 parquet files) | **8.8 GB** | plant flowering curves; per-cell plant phenology |
| GBIF pollinator occurrences | `/scratch/ariana.l/Plant Pollinator Initial Analysis/pollinator_observations_v2.csv` | **997 MB** | pollinator occupancy, activity curves, abundance |
| PhenoField plant occupancy | `/scratch/ariana.l/Stage 4 Link Prediction Model/stage4_F_existence_phenofield.csv` | 82 MB | plant occupancy matrix |
| Pollinator SDM output | `/scratch/ariana.l/Stage 6 Seed Testing/` | 529 MB | per-cell pollinator phenology (1,275 taxa) |

These are the irreplaceable ones — they are model outputs and processed extracts that cannot be
regenerated from public sources without rerunning the upstream models.

**Compressed transfer:** the PPE directory is 6,697 small parquets; tar it rather than copying file
by file.

```bash
tar -cf - -C /scratch/ariana.l ppe-outputs/opportunity_surface | zstd -T0 -3 > ppe_surface.tar.zst
```

---

## 2. Should be re-fetched, not copied — public

| data | source | size |
|---|---|---:|
| GloBI interaction snapshot | `depot.globalbioticinteractions.org/snapshot/target/data/csv/interactions.csv.gz` | 2.3 GB |
| BioCLIP text encoder | HuggingFace `imageomics/bioclip` | small |
| BioCLIP-2 image embeddings | HuggingFace `imageomics/TreeOfLife-200M-Embeddings` | streamed, not stored |
| SINR weights *(if used)* | `data.caltech.edu` record `dk5g7-rhq64` | small |

⚠️ **The GloBI snapshot must be the same version.** Results are pinned to the 2026-08-26 snapshot.
Re-downloading later gives a different file, since GloBI is updated continuously. Either copy our
2.3 GB file, or verify the digest recorded in the dataset card matches before rebuilding. **If the
digest differs, the network must be rebuilt and all numbers recomputed** — do not mix.

---

## 3. Worth copying to skip a rebuild (optional)

| artifact | size | rebuild cost if skipped |
|---|---:|---|
| `data/network/` — the built network, node tables, ledger | **5 MB** | ~15 min |
| `artifacts/cache/plant_surfaces_*.npy` | 2.1 GB | ~12 min from PPE parquets |
| `artifacts/cache/sdm_surfaces_*.npy` | 419 MB | ~10 min from SDM output |
| `artifacts/cache/bioclip*_text_*.npy` | 45 MB | ~30 s on a GPU |

**Always copy `data/network/` (5 MB).** It is tiny, and it is the verified artifact — copying it
guarantees the other machine works from the identical network rather than a rebuild that might
differ.

---

## 4. Not worth copying

`artifacts/archive/` (superseded results), `artifacts/cache/N_full_*.npy` and other derived matrices,
`__pycache__`. All rebuild in minutes, and the archive is history rather than input.

---

## 5. Minimal viable transfer

To reproduce the network and run the current analyses:

```
data/network/                      5 MB   ← always
artifacts/external/globi_*.csv.gz  2.3 GB ← or re-fetch and verify the digest
PPE opportunity surface            8.8 GB
GBIF pollinator occurrences        997 MB
PhenoField F matrix                82 MB
SDM output                         529 MB
                                 ─────────
                                  ~12.7 GB
```

Plus the git repository, which carries all code, the protocol, and the docs.

---

## 6. Environment

Python 3.11 with pandas, numpy, scipy, scikit-learn, pyarrow (all analyses) and torch (neural
models). On the lab machine this is the conda environment `donuts` for CPU work and `phenoHR` for GPU
work. `lightfm` is an optional dependency for one baseline.

GPU is needed only for the neural ranker and for generating species embeddings; every baseline in
`src/antheia/baselines/` is CPU-only.

---

## 7. Paths

All input paths are currently absolute and point at `/scratch/ariana.l/`. `src/antheia/config.py`
resolves them, and honours the `ANTHEIA_DATA_ROOT` environment variable — set that on the new machine
rather than editing paths.
