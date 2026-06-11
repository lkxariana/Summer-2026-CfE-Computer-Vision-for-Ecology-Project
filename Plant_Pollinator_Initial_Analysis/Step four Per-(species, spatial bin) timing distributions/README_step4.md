# Step 4 — Spatial Binning and Overlap Region Identification

Combine the plant flowering events (Step 1) and pollinator observations (Step 3) into spatial bins across CONUS, visualize their density coverage, identify the bins with the highest plant–pollinator overlap, and group those bins into geographic regions for downstream timing-distribution analysis.

## Requirements

- Python 3.10+
- `pandas`
- `numpy`
- `matplotlib`
- `pyarrow` (for reading the plant parquet file)
- `geopy` (for reverse geocoding bin coordinates to states)

```bash
pip install pandas numpy matplotlib pyarrow geopy
```

## Data sources

- `plant_flowering_events.parquet` — output from Step 1, columns: `species, lat, lon, doy, year` (208,567 records, 50 target plant species)
- `pollinator_observations.csv` — output from Step 3, columns: `pollinator_species, lat, lon, doy, year` (21,887,248 records, five pollinator taxa)

## Usage

The analysis runs in four sub-steps. Each can be executed standalone via the corresponding Python script, or together as cells in `step4_overlap_analysis.py`.

```bash
# Pick a spatial bin size
python3 spatial_bin_eda.py

# Render plant and pollinator density maps separately
python3 plot_density_maps.py

# Identify and rank the top 20 overlap bins, save coordinates and states
python3 find_top20_overlap_bins.py

# Group the top 20 bins into geographic regions and re-plot
python3 plot_regions_overlay.py
```

## Method

### 4.1 Choose a spatial bin size

For each candidate bin size (0.25°, 0.5°, 1.0°, 2.0°), count the observations per `(lat_bin, lon_bin)` cell and report the percentage of bins meeting a coverage threshold. The threshold question — "how many plant observations per bin do we need?" — was clarified with the mentor: enough total observations to draw a smooth flowering histogram across the year, not enough distinct species. After this discussion we use the **observation count** (not distinct species count) as the EDA criterion.

We chose **0.5° × 0.5°** as the working spatial resolution. At ~40°N this corresponds to approximately **55 km × 43 km** per bin (the size of a small US county), which gives a meaningful regional signal without averaging across drastically different ecosystems.

### 4.2 Density maps

For each dataset:

1. Add `lat_bin` and `lon_bin` columns by flooring lat/lon to the chosen bin size: `lat_bin = (lat // 0.5) * 0.5`, same for lon.
2. Count observations per `(lat_bin, lon_bin)` cell.
3. Draw one rectangle per bin, colored by observation count using a sequential colormap (Greens for plants, Reds for pollinators).

### 4.3 Top 20 overlap bins

For each bin that contains both plants and pollinators, compute a normalized overlap score:

```
score = (plant_count / max_plant_count) × (pollinator_count / max_pollinator_count)
```

This multiplicative score rewards bins where both signals are strong; a bin with many pollinator observations but few plants (or vice versa) scores low. Rank all bins by score and keep the top 20. For each top bin, reverse-geocode the bin center to a US state using `geopy.Nominatim`.

Bins falling on coastal water (e.g., the bin just west of San Francisco) are manually labeled with the adjacent state.

### 4.4 Group top bins into regions

The top 20 bins fall into three geographic clusters. We assign each bin to a region using simple coordinate rules:

| Region | Rule | Bounding box (lon_min, lon_max, lat_min, lat_max) | # bins |
|--------|------|---------------------------------------------------|--------|
| East | `lon_min ≥ -90` | (-88.0, -71.0, 35.5, 44.5) | 8 |
| Pacific Northwest | `lat_min ≥ 45` and not East | (-123.0, -122.0, 45.5, 48.0) | 2 |
| California | otherwise | (-123.0, -117.0, 32.5, 38.5) | 10 |

These three regions become the spatial units for the per-region timing-distribution analysis in subsequent steps.

## Output files

| File | Description |
|------|-------------|
| `spatial_bin_eda.py` | EDA across candidate bin sizes |
| `plot_density_maps.py` | Plant and pollinator density maps |
| `find_top20_overlap_bins.py` | Compute overlap score, rank top 20, reverse-geocode states |
| `plot_regions_overlay.py` | Plot top 20 bins with region bounding boxes |
| `step4_overlap_analysis.py` | All four sub-steps combined |
| `plant_map.png` | Plant flowering density across CONUS (0.5°×0.5° bins) |
| `pollinator_map.png` | Pollinator observation density across CONUS (0.5°×0.5° bins) |
| `overlap_map_top20.png` | Combined density map with top 20 overlap bins circled |
| `overlap_map_regions.png` | Same map with the three regions outlined and labelled |
| `top20_overlap_bins.csv` | Coordinates and state for each of the top 20 bins |
| `top20_regions.csv` | Bounding box and bin count for each region |
