# Step 6: Initial Evaluation — Does Timing Matter?

This step evaluates whether incorporating temporal (phenological) information
changes the picture compared to spatial co-occurrence alone. It operationalizes
five metrics from the project brief and reports the headline finding: a
substantial fraction of spatially co-occurring plant-pollinator pairs are
temporally mismatched, justifying temporal modeling.

## Research Question

> If two species share a geographic bin, do they also share a flowering/activity
> season? Or does spatial co-occurrence systematically overestimate the
> probability of actual interaction?

## Inputs

All inputs are outputs of Step 5 (Overlap Coefficient computation):

| File | Description |
|------|-------------|
| `overlap_coefficients.csv` | Overlap coefficient per (edge, spatial bin), 517 rows |
| `spatial_vs_temporal.csv` | Per-edge Jaccard spatial overlap + mean temporal overlap |
| `bin_size_sensitivity.csv` | Headline metric recomputed at 5 bin sizes (0.5°–3.0°) |
| `top20_overlap_bins.csv` | Top 20 most observation-dense 0.5° bins |

## Metrics Computed

### Metric 1 — Distribution of temporal overlap scores
Histogram of overlap coefficients across all 517 edge×bin combinations.
Mean overlap = 0.394, median = 0.362 — well below 0.6, indicating timing
discriminates meaningfully across known interactions.

### Metric 2 — Fraction of low-overlap co-occurrences (Headline Result)
Count of (plant, pollinator) pairs that share a spatial bin but have
temporal overlap < 0.3:

- **38.7% of edge×bin combinations** (200 of 517) fall below 0.3
- **41.7% of edges** (5 of 12) have mean overlap < 0.3
- This substantially exceeds the 15–20% threshold identified as meaningful in
  the project brief

### Metric 3 — Per-edge spatial vs. temporal divergence
Scatterplot of Jaccard spatial overlap vs. mean temporal overlap per edge.
Spearman ρ = 0.59 (p = 0.04), Pearson r = 0.43 (p = 0.16).
The moderate, noisy correlation confirms that spatial and temporal overlap
are related but non-redundant — knowing where two species co-occur does not
reliably predict whether they are active at the same time.

### Metric 4 — Phenology-aware vs. naive ranking
Not computed in this step. Requires candidate pollinator expansion beyond
the 54 known edges — a structural extension left for future work.

### Metric 5 — Sensitivity to spatial bin size
The headline metric (% below 0.3) was recomputed at bin sizes 0.5°, 1.0°,
1.5°, 2.0°, and 3.0°. Results range from 34.9% (3.0°) to 53.3% (0.5°),
never dropping below the 15–20% significance threshold. The temporal signal
is robust and not an artifact of the chosen bin size.

## Key Findings

1. Temporal mismatch is pervasive — over a third of spatially co-occurring
   pairs are phenologically misaligned at the primary 1.5° resolution.
2. The signal is robust across all bin sizes tested.
3. Spatial co-occurrence is a moderate predictor of temporal alignment
   (Spearman ρ = 0.59) but leaves substantial unexplained variance.
4. Five pairs with comparable spatial co-occurrence span a 0.20–0.49 range
   in temporal overlap, showing timing adds discriminative signal that
   spatial methods cannot capture.

## Data Limitations

- Only 12 of 54 network edges had sufficient co-located observations
  (≥10 per species per bin) to compute temporal overlap. The remaining 42
  edges were excluded, which may bias surviving edges toward better-sampled
  species (notably *Asclepias syriaca*, which accounts for 7 of 12).
- n = 12 edges makes correlation statistics fragile; visual patterns are
  more interpretable than p-values in this context.
- Observation effort drives both signals; per-species, per-bin normalization
  absorbs effort within a species but cross-species comparisons inherit
  observer-density bias.

## Output Files

| File | Description |
|------|-------------|
| `overlap_distribution_threshold.png` | Histogram with 0.3 threshold marked |
| `spatial_vs_temporal_scatter.png` | Scatter of spatial vs. temporal overlap per edge |
| `bin_size_sensitivity_plot.png` | Sensitivity of headline metric to bin size |
| `evaluate_overlap.py` | All analysis and plotting code for this step |

## File Structure

```
step6_initial_evaluation/
├── README.md
└── evaluate_overlap.py
```

## Usage

```bash
python evaluate_overlap.py
```

Outputs are saved to `/scratch/ariana.l/`.

## Dependencies

- pandas, numpy, matplotlib, scipy
