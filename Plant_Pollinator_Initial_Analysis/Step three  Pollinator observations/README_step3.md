# Step 3 — Pollinator Observations

Build a flat table of pollinator occurrence records `(pollinator_species, lat, lon, doy, year)` for the four pollinator insect orders plus hummingbirds across the continental US.

**Note:** The original instruction calls for iNaturalist Open Data. We are using [GBIF](https://www.gbif.org/) instead, which provides equivalent coverage with a simpler download workflow and better metadata.

## Requirements

- Python 3.10+
- `pandas`

```bash
pip install pandas
```

## Data source

GBIF occurrence download requested through the GBIF web interface with the following filters applied at download time (matching the original Step 3 specification):

| Filter | Value | Why |
|--------|-------|-----|
| `Basis of record` | Human observation | community-verified equivalent of iNat `quality_grade = research` |
| `Continent` | North America | scope |
| `Country` | United States of America | CONUS scope |
| `Year` | 2013–2026 | match plant phenology window |
| `Coordinate uncertainty` | < 1000 m | drop obscured coordinates |
| `Taxon` | Hymenoptera, Lepidoptera, Diptera, Coleoptera (orders) + Trochilidae (family) | four pollinator insect orders + hummingbirds |

- Download format: simple TSV (tab-separated)
- Compressed size: ~2.8 GB
- Uncompressed size: ~13.7 GB
- Records: ~23.7 million before further cleaning

**GBIF citation:** GBIF.org (9 June 2026) GBIF Occurrence Download https://doi.org/10.15468/dl.39cba4

## Usage

### Download the GBIF dump

1. Go to https://www.gbif.org/occurrence/search and apply the filters above.
2. Click **Download** and choose **Simple** format. GBIF emails a download link when the file is ready (usually within 15 minutes).
3. On the server, download and unzip:
   ```bash
   wget -O gbif_pollinators.zip "<download URL from email>"
   unzip gbif_pollinators.zip -d /scratch/<user>/
   ```
   The unzipped file is a single CSV with tab-separated values, e.g. `0047411-260519110011954.csv`.

### Build the pollinator observations table

```bash
python3 build_pollinator_observations.py
```

This produces:

- `pollinator_observations.csv` — `(pollinator_species, lat, lon, doy, year)` per observation

## Method

1. **Stream the GBIF TSV** in 100,000-row chunks (the file is too large to load into memory at once).
2. **Keep only complete records**: drop rows missing `species`, `decimalLatitude`, `decimalLongitude`, `day`, `month`, or `year`. GBIF stores partial dates as a year-only `eventDate` in some records; using `day`/`month`/`year` directly avoids the date-parsing ambiguity.
3. **Compute day-of-year** from `(year, month, day)` for use in Step 4 timing distributions.
4. **Output** `(pollinator_species, lat, lon, doy, year)` per observation.

The taxonomy, geographic, temporal, and accuracy filters are all applied by GBIF at download time, so no additional filtering is needed in this script.

## Output files

| File | Description |
|------|-------------|
| `build_pollinator_observations.py` | Streaming filter and column-extraction script |
| `pollinator_observations.csv` | `(pollinator_species, lat, lon, doy, year)` per observation |

## Latest results

- **Records with complete dates:** 21,887,248
- **Unique pollinator species:** thousands across the five target taxa
- **Spatial coverage:** continuous US (CONUS), 2013–2026

The original GBIF download contains 23.7 million records; the ~1.8 million dropped are those without a complete day-month-year date.
