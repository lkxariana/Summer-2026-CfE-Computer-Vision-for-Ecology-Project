# Phenological Overlap Analysis

Computing and visualizing phenological overlap between plants and their pollinators using observational data binned by space (1.5° × 1.5°) and time (52 weeks).

## Overview

For each plant-pollinator edge in the interaction network, this step computes how much the seasonal activity of the plant overlaps with that of the pollinator within shared geographic regions. The overlap coefficient is defined as the area of intersection between the two normalized 52-week density distributions.

## Data Inputs

| File | Description |
|------|-------------|
| `plant_flowering_events.parquet` | Plant flowering observations with `species`, `lat`, `lon`, `doy` |
| `pollinator_observations_v2.csv` | Pollinator observations with `pollinator_species`, `lat`, `lon`, `doy` |
| `plant_pollinator_edges.csv` | Edge list of 54 plant-pollinator interactions |
| `top20_overlap_bins.csv` | Pre-computed top 20 most observation-dense 0.5° bins across all species |

All data is stored on the crow server under `/scratch/ariana.l/`.

## Parameters

```python
BIN_SIZE = 1.5    # spatial bin size in degrees
MIN_OBS  = 10     # minimum observations per (species, bin) to include
N_WEEKS  = 52     # weeks in a year (week 53 clipped to 52)
```

## Pipeline

### 1. Spatial Binning
Observations are assigned to 1.5° × 1.5° lat/lon bins using floor division. The lower-left corner of each bin serves as its identifier.

### 2. Weekly Density
For each (species, spatial bin) combination, the `weekly_density()` function builds a normalized 52-week histogram. Bins with fewer than `MIN_OBS=10` observations are dropped to avoid noisy distributions.

### 3. Overlap Coefficient
For each edge × spatial bin where both plant and pollinator have ≥10 observations:

```
overlap = sum(min(p_density, q_density))
```

This is the area of intersection between the two normalized weekly distributions, ranging from 0 (no overlap) to 1 (identical distributions).

### 4. Filter to Top 20 Bins
The top 20 most observation-dense 0.5° bins (across all species) are mapped to their parent 1.5° bins and used to restrict the analysis to the most data-rich regions.

### 5. Visualization
For each surviving edge, the 52-week density curves are averaged across all top-20 bins where the edge appears, then plotted as KDE-smoothed curves with the overlap region shaded.

## Results

- **54 edges** in the original network
- **12 edges** survived the ≥10 obs filter in at least one spatial bin
- **517 edge × bin combinations** with valid overlap values
- **10 edges** retained after restricting to the top 20 bins (59 rows)

### Overlap distribution (all 517 edge × bin combinations)

| Statistic | Value |
|-----------|-------|
| Mean | 0.394 |
| Std | 0.188 |
| Min | 0.019 |
| 25% | 0.240 |
| 50% | 0.362 |
| 75% | 0.535 |
| Max | 0.865 |

## Data Limitation

The original network contained 54 plant-pollinator edges, but **42 were dropped** because no spatial bin had both species recorded with ≥10 observations. This reflects sparse co-located observation coverage in the underlying dataset rather than a true absence of interaction. The 12 surviving edges are dominated by *Asclepias syriaca* (7 of 12), a well-known pollinator generalist with dense observational coverage.

## Output Files

| File | Description |
|------|-------------|
| `overlap_coefficients.csv` | Overlap coefficient per (edge, spatial bin), 517 rows |
| `surviving_edges.csv` | The 12 edges that survived the ≥10 obs filter |
| `edge_overlap_summary.csv` | Per-edge mean, std, and bin count of overlap values |
| `phenology_grid_top20.png` | Grid plot of averaged phenology curves (raw histograms) |
| `phenology_grid_kde.png` | Grid plot of KDE-smoothed phenology curves (`bw_method=0.20`) |

## File Structure

```
phenological_overlap/
├── README.md
├── compute_overlap.py        # Steps 1–3: binning, density, overlap
├── filter_top20.py           # Step 4: restrict to top 20 bins
└── visualize_phenology.py    # Step 5: KDE-smoothed grid plot
```

## Usage

Run on the crow server inside the appropriate Python environment:

```bash
python compute_overlap.py
python filter_top20.py
python visualize_phenology.py
```

Or run cell-by-cell in Jupyter for interactive exploration.

## Dependencies

- `pandas`
- `numpy`
- `matplotlib >= 3.10` (system version incompatible with NumPy 2.x)
- `scipy >= 1.x` (rebuilt for NumPy 2.x)

Install with:
```bash
pip install --user --upgrade matplotlib scipy
```
