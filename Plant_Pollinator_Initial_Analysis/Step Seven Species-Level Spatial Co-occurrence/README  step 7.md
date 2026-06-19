# Step 7 — Species-Level Spatial Co-occurrence (Jaccard Range Overlap)

Identifying the plant-pollinator species pairs whose geographic ranges are most tightly coincident across CONUS, as a corrective to Step 4's aggregate observation-density overlap score.

## Motivation

Step 4's overlap score — `(plant_count_in_bin / max) × (pollinator_count_in_bin / max)` — is computed on aggregate observation density and has no notion of which plant species and which pollinator species are actually present in a bin. A bin can score high purely because it has heavy total observation traffic (e.g. a city park with dense iNaturalist/GBIF coverage), regardless of whether any real plant-pollinator pair is densely co-located there.

Step 7 instead computes co-occurrence at the **(plant species, pollinator species)** level — for every possible combination, not limited to the 54 (later corrected to 39 valid) GloBI edges, what fraction of either species' occupied 0.5° bins are shared. This finds species pairs whose *ranges* coincide, independent of observation density and independent of any pre-existing interaction list.

## Method

**Co-occurrence metric:**
```
jaccard(plant, pollinator) = |bins(plant) ∩ bins(pollinator)| / |bins(plant) ∪ bins(pollinator)|
```
Ranges from 0 (no shared bins) to 1 (identical spatial range). A pair only scores high if both species' ranges genuinely track each other — a cosmopolitan species occupying hundreds of bins is penalized by the union term, even if it is locally abundant in any single bin.

**Bin size:** 0.5° × 0.5°, matching Step 4's original resolution.

**Species scope:** all plant × pollinator combinations with at least one shared bin, not restricted to known GloBI edges — after dropping globally rare species (<10 total observations, consistent with Step 4's EDA threshold), 50 plant species × 13,635 pollinator species.

**Pollinator identity granularity:** species level (e.g. *Apis mellifera*), not taxon/order level.

## An Abandoned Approach (documented, not used)

The first method tried was a **min-count score** — `score(plant, pollinator, bin) = min(plant_count_in_bin, pollinator_count_in_bin)` — computed per bin rather than per range, streamed through a bounded min-heap (rather than materializing the full ~17.4M-row cross join, which was judged too large to build directly given the datasets already in memory).

This approach was abandoned after producing a misleading top-10: every one of the top 10 entries paired *Bellis perennis* (common daisy) with a different locally-abundant pollinator, all in the same single Bay Area bin, all at the identical score of 611. The reason: *Bellis perennis*'s count in that one bin (611) was smaller than nearly every pollinator's count there, so `min()` simply returned the plant's count regardless of which pollinator it was paired with — the ranking was being driven by one plant's local abundance, not by any genuine pair-specific relationship. This is the same density bias the method was meant to escape, just relocated from "bin density" to "the smaller side's per-species density." The Jaccard approach above replaced it.

## Pipeline

1. **Load data.** `plant_flowering_events.parquet` (208,567 records, 50 species) and `pollinator_observations.csv` (21,915,721 records, 25,484 species). The raw pollinator load is cached to parquet immediately, since it is the slow step (~30s–2min) and is never needed again in raw CSV form.
2. **Spatial binning + rare-species filter.** 0.5° bins; drop species with <10 total observations across CONUS. Plants: all 50 retained, no records lost. Pollinators: 25,484 → 13,635 species, but only ~37K of 21.9M records dropped (0.17%) — the long tail of rare species contributed negligible data.
3. **Per-species, per-bin counts**, cached to parquet (21,622 plant cells; 1,258,226 pollinator cells).
4. **Join-size estimation.** All 2,431 distinct plant bins are also pollinator bins (full spatial containment, no coverage gap). Estimated full cross-join size: ~17.4 million rows — large enough that a direct merge was avoided in favor of the per-bin streaming approach.
5. **Jaccard computation.** For each of 2,431 shared bins, cross plant species against pollinator species present in that bin, track a running top-100 by Jaccard score via a min-heap. An O(1) dictionary lookup for (species, bin) → count was precomputed before the main loop for the hotspot-bin selection step, since the initial pandas `.loc`-based version was too slow (estimated 20–40+ minutes; the dictionary version ran in roughly 2–5 minutes).
6. **Reverse geocoding.** Each hotspot bin's center is reverse-geocoded to a US state via `geopy.Nominatim`, rate-limited to 1 request/second.

## Results

- **Top pair:** *Malosma laurina* (laurel sumac) + *Paranoplium gracile*, Jaccard 0.6538, hotspot in the Los Angeles basin (34.0, -118.5).
- **Rank 2:** *Larrea tridentata* (creosote bush) + *Asphondylia auripila* (a gall-forming midge known to specialize on creosote bush), Jaccard 0.625, 130 shared bins, hotspot in the Mojave/Sonoran desert (33.0, -116.5). This is the only pair that also appears in the cleaned GloBI edge list (see Step 8), making it the rare case where both range-overlap and documented-interaction evidence agree.
- **Geographic clustering:** the top 100 pairs concentrate heavily into two regions — the Northeast/Mid-Atlantic (Virginia 25, Massachusetts 24, Vermont 19; driven by woodland spring-ephemeral plants such as *Aquilegia canadensis*, *Sanguinaria canadensis*, *Caltha palustris*, *Lysimachia borealis*) and Southern California/Mojave (California 11; driven by *Malosma laurina* and *Larrea tridentata*). A long tail of single-digit counts (Minnesota, Maryland, Illinois, Texas, North Carolina, Florida, Washington, Indiana) shows some interior/non-coastal representation that Step 4's density-based method would not have surfaced, since those regions have far less citizen-science observation traffic.
- **Known data-quality flags within the top results:** *Lymantria dispar* (rank 19, paired with *Lysimachia borealis*) is the invasive spongy moth, not a pollinator in any meaningful sense — included because the pollinator dataset captures broad insect taxa rather than a curated pollinator list. *Asphondylia auripila* (rank 2) is a gall-former, not a pollinator in the mutualistic sense.

## Output Files

| File | Description |
|------|-------------|
| `cache_pollinators_raw.parquet` | Raw pollinator observations cached to parquet (no filtering) |
| `cache_plant_counts.parquet`, `cache_pollinator_counts.parquet` | Per-species, per-0.5°-bin observation counts, after the rare-species filter |
| `top100_jaccard_pairs.csv` | Top 100 plant-pollinator pairs by Jaccard score, with hotspot bin, shared-bin count, and (after geocoding) state |

## File Structure

```
step7_jaccard_overlap/
├── README.md
├── 01_load_data.py            # Load raw plant/pollinator data, cache pollinator load
├── 02_bin_and_filter.py       # 0.5° binning, rare-species filter, per-bin counts
├── 03_estimate_join_size.py   # Join-size estimate before committing to a method
├── 04_jaccard_overlap.py      # Core method: Jaccard score, top-100 heap extraction
└── 05_geocode_hotspots.py     # Reverse-geocode hotspot bins to US states
```

## Usage

Run on the crow server, in sequence:

```bash
python 01_load_data.py
python 02_bin_and_filter.py
python 03_estimate_join_size.py
python 04_jaccard_overlap.py
python 05_geocode_hotspots.py
```

Or run cell-by-cell in Jupyter, as this was originally developed.

## Dependencies

- `pandas`
- `numpy`
- `geopy` (install via `pip install geopy --user`; do **not** use `--break-system-packages` on crow's pip version, which predates PEP 668 enforcement and does not recognize that flag — plain `--user` works)

## Known Limitations

- Jaccard score measures spatial range overlap (co-occurrence), not confirmed ecological interaction. See Step 8 for a direct, quantified comparison against documented GloBI interactions, which finds the two are largely independent signals — only 1 of the top 100 Jaccard pairs is a documented edge.
- The pollinator observation dataset is not taxonomically curated to confirmed pollinators; it includes gall-formers, parasitoids, and at least one invasive defoliator moth.
- High Jaccard scores are mechanically easier to achieve for range-restricted regional specialists than for widespread generalists — this is a property of the metric, not necessarily a signal of ecological importance, and partly explains the dominance of regionally-endemic species (*Malosma laurina*, *Larrea tridentata*) at the top of the ranking.
