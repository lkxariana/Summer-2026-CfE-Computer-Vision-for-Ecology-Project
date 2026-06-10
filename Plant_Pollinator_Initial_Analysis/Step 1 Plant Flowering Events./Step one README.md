# Step 1 — Plant Flowering Events

Extract every observed **flowering** event for the 50 target plant species from
the PhenoField dataset, and save them as a flat table of
`(species, lat, lon, doy, year)`.

This folder produces the plant-side input for the plant–pollinator co-occurrence
analysis.

---

## Step-by-step summary

### 1. Data source

| | |
|---|---|
| **Dataset** | [`dcher95/phenofield`](https://huggingface.co/datasets/dcher95/phenofield) (Hugging Face) |
| **Split** | `train` |
| **Shards** | 107 `.arrow` shards (`train/data-{00000..00106}-of-00107.arrow`) |
| **Total rows scanned** | 2,650,448 |
| **Unique species in dataset** | 6,466 |

PhenoField is plant phenology imagery/observation data. Each record carries a
`species`, coordinates (`latitude`, `longitude`), a `doy` (day of year, 1–365),
a `year`, and a `label` describing the phenological state.

> **Note:** PhenoField has **no `date` column** — only `doy` and `year`. Storing
> `doy` is also convenient downstream, since Step 4 bins observations by week of
> year (`doy // 7`).

### 2. Selecting the 50 target species

Before extracting events, we tallied species frequency across all 107 shards to
find the most-populated species, then took **ranks 2–51** as the targets.
Rank 1 (`unknown`, 19,416 records) is excluded because it isn't a real species.

- Code: **`download_dataset.py`** (streams all shards, counts species with a
  `collections.Counter`, reports the top 55)
- Export: **`export_top_species.py`** (parses the run log into
  `top55_species.parquet` and `top55_species.docx`)
- Run log: **`species_tally_top55.log`**

The top 55 species account for 463,105 records (17.5% of the dataset); the 50
targets are ranks 2–51, ending at *Larrea tridentata* (5,924 records).

### 3. Filtering

For each shard, two filters are applied and combined:

| Filter | Rule |
|---|---|
| **Flowering only** | `label == 2` (label value 2 = flowering in PhenoField) |
| **Target species** | `species` is one of the 50 targets |

Only the columns `species`, `lat`, `lon`, `doy`, `year` are kept
(`latitude`/`longitude` are renamed to `lat`/`lon`). Each shard's blob is
evicted from the cache right after processing so `/scratch` doesn't fill up
while streaming ~2.5 GB per shard.

### 4. Code

| File | Role |
|---|---|
| **`export_flowering_events.py`** | **Main Step 1 script.** Streams 107 shards, applies the flowering + species filters, writes the flat table. |
| `download_dataset.py` | Species tally — used in step 2 above to pick the 50 targets. Counts only; does **not** save events. |
| `export_top_species.py` | Exports the tally log to `top55_species.parquet` / `.docx`. |

> **Suggested rename** for clarity (pipeline order):
> `download_dataset.py` → `01_species_tally.py`,
> `export_top_species.py` → `01b_export_top_species.py`,
> `export_flowering_events.py` → `02_export_flowering_events.py`.

### 5. Output

| | |
|---|---|
| **File** | `plant_flowering_events.parquet` |
| **Location** | `/scratch/ariana.l/plant_flowering_events.parquet` |
| **Records** | 208,567 flowering events |
| **Columns** | `species`, `lat`, `lon`, `doy`, `year` |

---

## How to run

Cache is routed to `/scratch` (the home directory disk is full):

```bash
cd /home/ariana.l/CfE2026CVforEcology/rawplantsdata

# (one-time, if needed) fix tqdm notebook widget bug
pip install --upgrade tqdm ipywidgets --user

# extract flowering events (streams all 107 shards — ~50 min)
nohup python3 -u export_flowering_events.py 2>&1 | tee flowering_export.log &
```

### Requirements

- Python 3.10+
- `pandas`, `pyarrow`, `datasets`, `huggingface_hub`
- `python-docx` (only for the tally's Word export)

```bash
pip install pandas pyarrow datasets huggingface_hub python-docx
```

---

## To verify / open questions

The Step 1 instruction specifies **"CONUS extent, post-2013-01-01."** The current
extraction does **not** explicitly apply either filter — it relies on PhenoField
already being scoped that way. Before finalizing, confirm one of:

- PhenoField is CONUS-only and post-2013 by construction (then no change needed), **or**
- Add an explicit bounding-box filter on `lat`/`lon` and a `year >= 2013` filter
  to `export_flowering_events.py`.

(The overlap map shows all records falling within the continental US, which is
consistent with CONUS scoping, but this is worth confirming against the dataset
card or with your mentor.)
