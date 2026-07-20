# Step 8 — Temporal Overlap for Jaccard Range Overlap Pairs

Recomputing the phenological (temporal) overlap coefficient from Step 5, but restricted to the top 100 plant-pollinator pairs identified in **Step 7 — Jaccard Range Overlap**, using all of each pair's shared spatial bins rather than Step 5's top-20-observation-density bin restriction. Also includes a standalone Jaccard scoring of the cleaned 39-edge GloBI list, for direct comparison against the top 100 searched pairs.

## Motivation

Step 5 computed temporal overlap for the original 54-edge GloBI list (later found to contain 15 invalid self-paired rows — see the Step 5 README correction — so the real list was 39 edges), restricted to Step 4's top-20 most observation-dense bins. Only 12 of those edges survived the ≥10-observation filter in any bin, and the analysis was limited by sparse co-located coverage.

Step 7 (Jaccard Range Overlap) found, across all possible plant-pollinator species combinations, which pairs have the most tightly coincident geographic ranges — independent of the GloBI edge list and independent of observation density. See the Step 7 README for the method and its motivation in full. Because Jaccard-selected pairs are chosen specifically for spatial co-occurrence, they are far more likely to have sufficient co-located data for a temporal overlap computation than an arbitrary edge list. Step 8 tests this directly: recompute temporal overlap, but for Step 7's top 100 pairs instead of the GloBI edge list.

## Data Inputs

| File | Description |
|------|-------------|
| `plant_flowering_events.parquet` | Plant flowering observations: `species, lat, lon, doy` |
| `pollinator_observations_v2.csv` | Pollinator observations: `pollinator_species, lat, lon, doy, year`. Used instead of the non-`_v2` file from Step 7 because the latter lacks a `doy` column. The two files have very similar (but not identical) row counts after filtering to the same species: 605,772 vs. 604,940. |
| `top100_jaccard_pairs.csv` | Output of Step 7 (Jaccard Range Overlap): the 100 plant-pollinator pairs with highest Jaccard score |
| `cache_plant_counts.parquet`, `cache_pollinator_counts.parquet` | Caches from Step 7, reused here to rebuild `plant_bin_sets`/`pollinator_bin_sets` for the GloBI sub-analysis without re-scanning the raw CSV |
| `plant_pollinator_edges.csv` | The original 54-row GloBI edge list (15 self-pairs, 39 valid) |

All data is stored on the crow server under `/scratch/ariana.l/`.

## Parameters

```python
BIN_SIZE_TEMPORAL = 1.5   # spatial bin size, matches Step 5's original choice
N_WEEKS = 52
MIN_OBS = 5               # see "Choosing MIN_OBS" below
```

## Pipeline

### 1. Filter to the species in Step 7's top 100 pairs
The top 100 pairs involve only 17 distinct plant species and 63 distinct pollinator species (out of 50 and 13,635 respectively in Step 7's full dataset). Both raw observation files are filtered down to these species before any further processing — a large reduction relative to Step 7's full sweep.

### 2. Spatial binning at 1.5°
Same floor-division binning as Step 5, applied to the filtered, smaller dataset.

### 3. Choosing MIN_OBS
Step 5 used `MIN_OBS=10`. Checking this threshold against the new (smaller, pre-filtered) dataset showed it would drop 38.3% of plant (species, bin) cells and 35.7% of pollinator cells — substantially more than a quick percentile read suggested (the 25th percentile of observation counts is 17 for plants and 21 for pollinators, but the distribution is heavily right-skewed, so percentile and threshold-counting diverge). `MIN_OBS=5` was chosen instead, dropping a more moderate ~24.4% (plant) / ~24.9% (pollinator) of cells, while retaining 99.1% / 99.4% of total observation rows — confirming the dropped cells were overwhelmingly the sparse, low-reliability ones.

### 4. Weekly density and overlap coefficient
Same formula as Step 5:
```
overlap = sum(min(p_density, q_density))
```
computed for every (edge, bin) combination where both species survive the `MIN_OBS=5` filter in that bin — using **all** shared bins per edge, not a top-20-density restriction. This is possible because every one of the 100 Jaccard pairs was already selected for tight spatial range overlap, so coverage is far less sparse than the GloBI list's was.

### 5. Per-edge summary
For each edge, the per-bin overlap values are summarized as: mean, an observation-weighted mean (each bin weighted by `min(n_obs_plant, n_obs_pollinator)`, consistent with the min-count logic from Step 7's Jaccard score), standard deviation, min, and max.

### 6. GloBI edge Jaccard (standalone sub-analysis)
Separately, the Jaccard score (same metric and bin sets as Step 7) was computed for the 39 cleaned GloBI edges — not searched against all combinations, just these 39 specific pairs, ranked against each other — to compare directly against Step 7's top-100 searched pairs.

### 7. Visualization
KDE-smoothed phenology grids (`bw_method=0.20`, same as Step 5), built by averaging each edge's density curve across *all* its shared bins, with panels ordered west-to-east by each edge's mean shared-bin longitude.

## Results

### Temporal overlap, Step 7's top 100 pairs
- **100 / 100 edges** had at least one valid shared bin with sufficient data on both sides (vs. 12 / 39 valid edges in Step 5's original GloBI-based analysis). This high survival rate is a direct consequence of starting from Step 7's spatially pre-filtered pairs, not evidence that the temporal method itself improved.
- **7,555 total (edge, bin) overlap rows** across the 100 edges (~75 bins/edge on average, vs. Step 5's 517 rows across 12 edges, ~43 bins/edge).
- Overlap values within a single edge can vary enormously across its shared bins — e.g. Step 7 rank 2 (*Larrea tridentata* / *Asphondylia auripila*) ranged from 0.034 to 0.665 depending on which bin was examined, so a single per-edge average can obscure a lot of real geographic variation in phenological sync.

### Spatial vs. temporal overlap are largely independent signals
Several of Step 7's top-100 pairs (selected purely for spatial range overlap) show near-zero temporal overlap — e.g. *Sanguinaria canadensis* / *Lethe anthedon* (`overlap_mean = 0.0019`), *Aquilegia canadensis* / *Pelecinus polyturator* (`0.011`). Conversely, Step 7's highest-Jaccard pair, *Larrea tridentata* / *Asphondylia auripila*, also has strong temporal overlap (`weighted mean = 0.555`) — making it the rare case where both signals agree, and consistent with why it stood out from the earliest exploration of Step 7's results.

### GloBI edge Jaccard scores vs. Step 7's top-100 searched pairs
- Of the 39 cleaned GloBI edges, only **12 could be scored** — the other 27 involve a plant or pollinator species that did not survive Step 7's `MIN_TOTAL_OBS >= 10` filter and so has no bin-set entry.
- The 12 scored GloBI edges have Jaccard scores ranging **0.053–0.625** (mean 0.274), almost entirely below the **0.505–0.654** range of Step 7's top-100 searched pairs (which is tight by construction, since it's the top 100 *by* Jaccard score).
- **Only one pair appears in both lists**: *Larrea tridentata* / *Asphondylia auripila* (GloBI rank 1 of 12, Step 7 rank 2 of 100). This is strong evidence that high spatial range overlap is **neither necessary nor sufficient** for a documented plant-pollinator interaction — e.g. *Asclepias syriaca* / *Apis mellifera* is a well-known, well-documented interaction with a comparatively low Jaccard score of 0.209, because honeybees are far more geographically widespread than milkweed.
- *Asclepias syriaca* dominates the scored GloBI list (8 of 12 rows), the same generalist-bias pattern Step 5's README already flagged for its own 12 surviving edges.

## Output Files

| File | Description |
|------|-------------|
| `cache_step8_pollinators_filtered.parquet` | Pollinator observations (v2, with `doy`) filtered to the 63 species in the top 100 pairs |
| `step8_edge_overlap_summary.csv` | Per-edge overlap statistics (mean, weighted mean, std, min, max, bin count) for Step 7's 100 pairs |
| `globi39_jaccard_ranked.csv` | Jaccard scores for the 12 (of 39) scorable cleaned GloBI edges, ranked high to low |
| `step8_phenology_grid_top10_jaccard.png` | KDE grid, top 10 pairs by Jaccard rank, west-to-east ordered |
| `step8_phenology_grid_top10_temporal.png` | KDE grid, top 10 pairs by observation-weighted temporal overlap, west-to-east ordered |
| `step8_phenology_grid_top12_temporal.png` | KDE grid, top 12 pairs by observation-weighted temporal overlap (earlier n=12 cut, kept for reference) |

## File Structure

```
step8_temporal_overlap/
├── README.md
├── 01_load_and_filter.py              # Load Jaccard pairs, filter raw data to relevant species
├── 02_bin_and_check_sufficiency.py    # 1.5° binning, MIN_OBS threshold check
├── 03_filter_and_build_densities.py   # Apply MIN_OBS=5, build 52-week density histograms
├── 04_compute_overlap.py              # Per-(edge,bin) overlap coefficient, per-edge summary
├── 05_globi_edge_jaccard.py           # Standalone: Jaccard score for the 39 cleaned GloBI edges
└── 06_visualize_phenology.py          # KDE-smoothed phenology grids (reusable plotting function)
```

## Usage

Run on the crow server, in sequence, inside the appropriate Python environment:

```bash
python 01_load_and_filter.py
python 02_bin_and_check_sufficiency.py
python 03_filter_and_build_densities.py
python 04_compute_overlap.py
python 05_globi_edge_jaccard.py
python 06_visualize_phenology.py
```

Or run cell-by-cell in Jupyter for interactive exploration — this is how the analysis was originally developed.

## Dependencies

- `pandas`
- `numpy`
- `scipy` (for `gaussian_kde`, requires the NumPy-2.x-compatible build already installed for Step 5)
- `matplotlib >= 3.10`

## Known Limitations

- The high (100/100) edge survival rate in this step reflects the fact that the input pairs were already pre-selected for spatial co-occurrence in Step 7 (Jaccard Range Overlap); it should not be read as evidence that the temporal-overlap method itself became more robust.
- `pollinator_observations_v2.csv` (used here) and `pollinator_observations.csv` (used in Step 7) are not byte-identical files, though they are very close in size after filtering to the same species. Direct row-level comparison between Step 7 and Step 8 results should account for this.
- As with the rest of this project, all overlap measures (Step 7's spatial Jaccard score and this step's temporal density overlap) describe co-occurrence, not confirmed interaction. The GloBI-vs-searched-pairs comparison in this step is itself evidence that the two are not interchangeable.
