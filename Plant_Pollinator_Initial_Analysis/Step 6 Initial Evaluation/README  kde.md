# Step 6.2: Region-Specific Phenological Overlap KDE

Extends the overall KDE visualization (Step 6.1) by splitting the analysis
into three geographic regions, allowing comparison of plant-pollinator
phenological overlap across different ecosystems.

## Regions

| Region | Lat range | Lon range | Edges | Notes |
|--------|-----------|-----------|-------|-------|
| East | 35.5–44.5°N | 88.0–71.0°W | 10 | Dominated by *Asclepias syriaca* |
| California | 32.5–38.5°N | 123.0–117.0°W | 3 | Coastal CA bins |
| Desert Southwest | 28.0–36.0°N | 120.0–103.0°W | 3 | Core *Larrea tridentata* habitat |
| Pacific Northwest | 45.5–48.0°N | 123.0–122.0°W | 0 | Skipped — no surviving edges |

**Note:** Pacific Northwest has 2 top-20 observation-dense bins (Seattle area)
but zero surviving edges because the target plant species (*Asclepias syriaca*,
*Larrea tridentata*, etc.) are not distributed there. This is an example of
observation effort bias — high iNaturalist density does not imply coverage of
the target species.

**Note:** Desert Southwest was added as a fourth region beyond the original
three in Step 4, because *Larrea tridentata* (creosote bush) is a desert
species whose core distribution falls east of the California bounding box,
in the Mojave/Sonoran desert (~103–120°W). Step 4 will be updated accordingly.

## Color Scheme

Each region uses a distinct color palette so panels are immediately
distinguishable at a glance:

| Region | Plant line | Pollinator line | Overlap fill |
|--------|-----------|-----------------|--------------|
| East | Ruby red `#9B1B30` | Coral red `#FF6B6B` | Light red `#FFB3B3` |
| California | Dark green `#1B5E20` | Medium green `#66BB6A` | Light green `#C8E6C9` |
| Desert Southwest | Dark orange `#E65100` | Amber `#FFB300` | Light yellow `#FFE082` |

## What Each Panel Shows

Each subplot displays:
- **Plant curve** — KDE-smoothed 52-week flowering density, averaged across
  all spatial bins in the region where that edge appears
- **Pollinator curve** — KDE-smoothed 52-week activity density, same bins
- **Shaded overlap** — area of intersection between the two curves
- **Legend** — species name, observation count (n=), and mean overlap coefficient
- **Column header** — region name in the region's color

## Method

For each region:
1. Filter `overlap_coefficients.csv` to bins within the region bounding box
2. For each surviving edge in that region, retrieve all (lat_bin, lon_bin) pairs
3. Compute `weekly_density()` for plant and pollinator in each bin
4. Average the 52-week density curves across bins
5. Fit `scipy.stats.gaussian_kde` with `bw_method=0.20`
6. Plot smoothed curves and compute mean overlap coefficient from filtered rows

## Parameters

```python
BIN_SIZE = 1.5    # spatial bin size in degrees
MIN_OBS  = 10     # minimum observations per (species, bin)
N_WEEKS  = 52     # weeks per year
BW       = 0.20   # KDE bandwidth
```

## Output

| File | Description |
|------|-------------|
| `phenology_kde_regions_annotated.png` | Full 3-column annotated grid figure |

## Usage

```bash
python plot_kde_regions.py
```

## Dependencies

- pandas, numpy, matplotlib >= 3.10, scipy
