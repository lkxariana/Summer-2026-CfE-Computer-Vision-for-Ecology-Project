# Step 1 — Plant Flowering Events

**Goal:** Pull out every recorded flowering observation for our 50 target plant
species from the PhenoField dataset, and save them as a clean flat table with
one row per observation: `(species, lat, lon, doy, year)`.

This table is the plant-side input for the rest of the plant–pollinator
co-occurrence analysis.

---

## Step-by-step summary

### 1. Where the data comes from

The plant data comes from **PhenoField**, a publicly available dataset on
Hugging Face created by the research team at dcher95:

> https://huggingface.co/datasets/dcher95/phenofield

PhenoField is a plant phenology dataset — meaning it records what plants are
doing (leafing out, budding, flowering, fruiting) at specific locations and
times. Each row in the dataset represents one field observation of one plant
species, and includes:

- `species` — the plant species name (e.g. *Asimina triloba*)
- `latitude`, `longitude` — where the observation was made
- `doy` — day of year (a number from 1 to 365, e.g. doy 125 = May 5th)
- `year` — the year of the observation
- `label` — what the plant was doing, coded as a number (confirmed from the
  dataset card provided by our mentor):
  - `0` = vegetative (just leaves, no flowers)
  - `1` = flower budding
  - `2` = **flowering** ← this is what we want
  - `3` = fruiting

The dataset was pre-filtered by our mentor to **CONUS** (the continental United
States) and **post-2013**, so we do not need to apply those geographic or date
filters ourselves — they are already baked into the version of the dataset we
received.

The dataset is stored as **107 separate `.arrow` shard files**, each roughly
2.5 GB. In total, they contain **2,650,448 rows** covering **6,466 unique
species**.

> **Why `doy` instead of a date?** PhenoField does not have a `date` column —
> it stores `doy` (day of year) and `year` separately. This is actually
> convenient for our analysis: in Step 4, we compare plant and pollinator
> timing using week of year, which is just `doy // 7`.

---

### 2. Choosing the 50 target species

We can't work with all 6,466 species in PhenoField — most of them have very
few observations. So our first job was to figure out which species have enough
data to be worth analysing.

We streamed all 107 shards and counted how many observations each species has
across the whole dataset. The script reads each shard, extracts the `species`
column, feeds the names into a running tally (a Python `Counter`), and then
discards the shard to save disk space. At the end, it prints the top 55
most-recorded species.

The results showed:
- **Rank 1:** `unknown` (19,416 records) — excluded because it's not a real
  species name
- **Ranks 2–51:** the 50 real species we use as targets, from
  *Asimina triloba* (19,032 records) down to *Larrea tridentata* (5,924 records)
- Together these 50 species account for **463,105 records (17.5%
  of the full dataset)**

The tally output is saved in `top55_species.parquet` and `top55_species.docx`
for reference.

**Code for this step:**
- `download_dataset.py` — streams all shards and counts species frequencies;
  does **not** save individual records, only the counts
- `export_top_species.py` — reads the tally log and exports the ranked list to
  Parquet and Word formats
- `species_tally_top55.log` — the full console output from running the tally

---

### 3. Extracting the flowering events

Once we had the 50 target species, we ran a second pass over all 107 shards to
pull out the actual observation records. This time, for each shard, we applied
two filters simultaneously:

1. **Keep only flowering observations:** `label == 2`
   (throwing away vegetative, budding, and fruiting records)
2. **Keep only our 50 target species**
   (throwing away the other ~6,416 species)

For each row that passes both filters, we keep five columns:
`species`, `lat`, `lon`, `doy`, `year` — and discard everything else.

After processing each shard, we immediately delete it from the cache to prevent
the `/scratch` partition from filling up (each shard is ~2.5 GB, and we have
107 of them). All the filtered rows are collected in memory and written to a
single Parquet file at the end.

**Code for this step:**
- `export_flowering_events.py` — the main extraction script; streams all
  107 shards, applies both filters, and saves the output

---

### 4. Output

| | |
|---|---|
| **File name** | `plant_flowering_events.parquet` |
| **Saved to** | `/scratch/ariana.l/plant_flowering_events.parquet` |
| **Number of records** | 208,567 flowering observations |
| **Columns** | `species`, `lat`, `lon`, `doy`, `year` |

Each row is one flowering observation of one plant species at one location on
one day of a given year. This table feeds directly into Step 4, where we
spatially and temporally match plant flowering with pollinator observations.

---

## File summary

| File | What it does |
|---|---|
| `download_dataset.py` | Streams all 107 shards and tallies species counts. Run first. |
| `export_top_species.py` | Exports the tally results to `.parquet` and `.docx`. |
| `species_tally_top55.log` | Console log from the tally run — records of all 55 top species and summary stats. |
| `top55_species.parquet` | Ranked table of top 55 species (rank, species, count). |
| `top55_species.docx` | Same table in Word format for easy reading. |
| `export_flowering_events.py` | Main Step 1 extraction script. Filters to label==2 and 50 target species, saves flowering events. Run second. |
| `plant_flowering_events.parquet` | **Final output.** 208,567 flowering records with species, lat, lon, doy, year. |

---

## How to run

```bash
cd /home/ariana.l/CfE2026CVforEcology/PlantPollinatorInitialAnalysis/Step1

# Step A: tally species frequencies (streams all 107 shards — ~45 min)
nohup python3 -u download_dataset.py 2>&1 | tee species_tally_top55.log &

# Step B: export tally results to parquet and Word
python3 export_top_species.py

# Step C: extract flowering events (streams all 107 shards again — ~50 min)
nohup python3 -u export_flowering_events.py 2>&1 | tee flowering_export.log &
```

> Cache is routed to `/scratch/ariana.l/hf_cache` to avoid filling the home
> directory disk.

### Requirements

```bash
pip install pandas pyarrow datasets huggingface_hub python-docx
```
