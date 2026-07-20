"""Build a unique plant→pollinator edge list from the GloBI interactions dump.

Filters the GloBI snapshot to pollination-relevant interactions, normalizes
directionality so the plant is always in the source column, restricts to the
top 50 plant species from Step 1, and writes a deduplicated CSV of
(plant_species, pollinator_species) pairs.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

INPUT_PATH = Path("interactions.csv.gz")
OUTPUT_PATH = Path("plant_pollinator_edges.csv")
CHUNK_SIZE = 100_000

INTERACTION_TYPES = {
    "pollinates",
    "visits",
    "visitsFlowersOf",
    "visitedBy",
    "flowersVisitedBy",
    "hasFlowerVisitor",
    "pollinatedBy",
}

# *By interaction types have source/target inverted (pollinator is source).
INVERTED_TYPES = {"visitedBy", "flowersVisitedBy", "hasFlowerVisitor", "pollinatedBy"}

COLUMNS_NEEDED = [
    "sourceTaxonSpeciesName",
    "targetTaxonSpeciesName",
    "interactionTypeName",
]

PLANT_SPECIES = [
    "Asimina triloba", "Sanguinaria canadensis", "Mitchella repens",
    "Cypripedium acaule", "Erodium cicutarium", "Malosma laurina",
    "Diospyros virginiana", "Dipterostemon capitatus", "Phytolacca americana",
    "Trillium grandiflorum", "Claytonia virginica", "Microstegium vimineum",
    "Trillium ovatum", "Lonicera maackii", "Asclepias syriaca",
    "Rhus glabra", "Ligustrum sinense", "Aquilegia canadensis",
    "Passiflora incarnata", "Chimaphila maculata", "Trillium erectum",
    "Alliaria petiolata", "Celastrus orbiculatus", "Amphicarpaea bracteata",
    "Arisaema triphyllum", "Bignonia capreolata", "Houstonia caerulea",
    "Dicentra cucullaria", "Impatiens capensis", "Conium maculatum",
    "Mertensia virginica", "Convolvulus arvensis", "Glechoma hederacea",
    "Erythronium americanum", "Galium aparine", "Achillea millefolium",
    "Lysimachia borealis", "Caltha palustris", "Sambucus canadensis",
    "Staphylea trifolia", "Kalmia latifolia", "Maianthemum racemosum",
    "Lamium purpureum", "Triteleia laxa", "Erigeron philadelphicus",
    "Chamaenerion angustifolium", "Bellis perennis", "Eschscholzia californica",
    "Sambucus cerulea", "Larrea tridentata",
]


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"GloBI dump not found at {INPUT_PATH}. "
            f"Download it from https://zenodo.org/records/20546682/files/interactions.csv.gz"
        )

    print(f"Reading {INPUT_PATH} in chunks of {CHUNK_SIZE:,} rows...")
    print(f"Filtering to {len(INTERACTION_TYPES)} interaction types and "
          f"{len(PLANT_SPECIES)} target plant species.")
    print("-" * 60)

    chunks: list[pd.DataFrame] = []
    for i, chunk in enumerate(
        pd.read_csv(
            INPUT_PATH,
            sep=",",
            usecols=COLUMNS_NEEDED,
            chunksize=CHUNK_SIZE,
            low_memory=False,
        )
    ):
        if i % 10 == 0:
            print(f"Processing chunk {i}...")

        # Keep only pollination-related interactions.
        chunk = chunk[chunk["interactionTypeName"].isin(INTERACTION_TYPES)].copy()

        # Normalize directionality: swap *By types so the plant is source.
        mask = chunk["interactionTypeName"].isin(INVERTED_TYPES)
        chunk.loc[mask, ["sourceTaxonSpeciesName", "targetTaxonSpeciesName"]] = (
            chunk.loc[mask, ["targetTaxonSpeciesName", "sourceTaxonSpeciesName"]].values
        )

        # Restrict to target plant species.
        chunk = chunk[chunk["sourceTaxonSpeciesName"].isin(PLANT_SPECIES)]
        chunks.append(chunk)

    df = pd.concat(chunks)
    df = df.rename(
        columns={
            "sourceTaxonSpeciesName": "plant_species",
            "targetTaxonSpeciesName": "pollinator_species",
        }
    )

    edge_list = (
        df[["plant_species", "pollinator_species"]]
        .drop_duplicates()
        .dropna()
        .sort_values(["plant_species", "pollinator_species"])
        .reset_index(drop=True)
    )

    print("-" * 60)
    print(f"Unique (plant, pollinator) pairs: {len(edge_list)}")
    print(f"Plants with at least one pollinator: {edge_list['plant_species'].nunique()}")

    edge_list.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved edge list to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
