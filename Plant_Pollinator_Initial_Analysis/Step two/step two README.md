# Step 2 — Plant→Pollinator Edge List

Build a unique `(plant_species, pollinator_species)` edge list from the [GloBI](https://www.globalbioticinteractions.org/) interactions dump, restricted to the top 50 plant species identified in Step 1.

## Requirements

- Python 3.10+
- `pandas`

```bash
pip install pandas
```

## Data source

GloBI interactions snapshot (tab-separated values, gzipped):

- URL: https://zenodo.org/records/20546682/files/interactions.csv.gz
- Compressed size: ~2.8 GB
- Format: CSV (despite the `.csv` extension, the file uses comma-separated values with millions of interaction records)

## Usage

### Download the GloBI dump

```bash
cd /path/to/rawpollinatordata
wget -O interactions.csv.gz "https://zenodo.org/records/20546682/files/interactions.csv.gz"
```

### Build the edge list

```bash
python3 build_edge_list.py
```

This produces:

- `plant_pollinator_edges.csv` — unique `(plant_species, pollinator_species)` pairs

## Method

1. **Filter interactions**: Keep only rows whose `interactionTypeName` is one of:
   `pollinates`, `visits`, `visitsFlowersOf`, `visitedBy`, `flowersVisitedBy`, `hasFlowerVisitor`, `pollinatedBy`.

2. **Normalize directionality**: The `*By` interaction types (`visitedBy`, `flowersVisitedBy`, `hasFlowerVisitor`, `pollinatedBy`) have the pollinator as source and the plant as target. Swap source and target for these rows so that the plant column is consistently the plant and the pollinator column is consistently the pollinator.

3. **Restrict to target plants**: Keep only rows where the plant species is in the top 50 list from Step 1 (rank 2–51, excluding `unknown`).

4. **Deduplicate**: Drop duplicate `(plant_species, pollinator_species)` pairs and any rows with missing values.

## Output files

| File | Description |
|------|-------------|
| `build_edge_list.py` | Main filtering and normalization script |
| `plant_pollinator_edges.csv` | Unique `(plant_species, pollinator_species)` pairs |

## Latest results

- **Unique plant–pollinator pairs:** 54
- **Plants with at least one recorded pollinator:** 27 of 50
- **Most-connected plant:** *Asclepias syriaca* (12 pollinators)

The relatively low pair count reflects sparse pollination coverage in GloBI for the target species set. Step 3 supplements this with pollinator occurrence records from GBIF.
