"""Step 1 — Plant flowering events.

Stream the dcher95/phenofield train split, keep only flowering records
(label == 2) for the 50 target plant species, and write a flat table of
(species, lat, lon, doy, year) for every observed flowering event.

This replaces the earlier tally script, which only counted species
frequencies and discarded the underlying rows.
"""

from __future__ import annotations

import os

# Route all Hugging Face / datasets cache to scratch before any HF imports.
os.environ["HF_HOME"] = "/scratch/ariana.l/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "/scratch/ariana.l/hf_cache/datasets"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import io
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from datasets import Dataset
from huggingface_hub import hf_hub_download

REPO_ID = "dcher95/phenofield"
NUM_SHARDS = 107
FLOWERING_LABEL = 2  # label == 2 means "flowering" in PhenoField
PROGRESS_EVERY = 10
OUTPUT_PATH = Path("/scratch/ariana.l/plant_flowering_events.parquet")

# Columns pulled from each shard. PhenoField stores coordinates as
# latitude/longitude and has no date column — only doy + year.
PARQUET_COLUMNS = ["species", "latitude", "longitude", "doy", "year", "label"]
COLUMN_RENAME = {"latitude": "lat", "longitude": "lon"}
OUTPUT_COLUMNS = ["species", "lat", "lon", "doy", "year"]

# The 50 target species: ranks 2–51 from the species tally
# (rank 1, "unknown", is excluded).
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


def shard_filename(shard_index: int) -> str:
    return f"train/data-{shard_index:05d}-of-00107.arrow"


def read_shard_dataframe(shard_path: str) -> pd.DataFrame:
    """Read one shard, keeping only the target columns."""
    table = Dataset.from_file(shard_path).data.table.select(PARQUET_COLUMNS)
    buffer = io.BytesIO()
    pq.write_table(table, buffer)
    buffer.seek(0)
    df = pd.read_parquet(buffer, columns=PARQUET_COLUMNS)
    return df.rename(columns=COLUMN_RENAME)


def evict_cached_shard(shard_path: str) -> None:
    """Drop the shard blob after processing so the cache stays bounded."""
    path = Path(shard_path)
    if path.is_symlink():
        blob = path.resolve()
        path.unlink()
        if blob.exists():
            blob.unlink()
    elif path.exists():
        path.unlink()


def main() -> None:
    Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["HF_DATASETS_CACHE"]).mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    target_species = set(PLANT_SPECIES)

    print(f"Streaming {NUM_SHARDS} train shards from {REPO_ID}...")
    print(f"Keeping label == {FLOWERING_LABEL} (flowering) for "
          f"{len(target_species)} target species.")
    print(f"Output columns: {OUTPUT_COLUMNS}")
    print("-" * 60)

    chunks: list[pd.DataFrame] = []
    total_rows = 0
    kept_rows = 0
    start_time = time.time()

    for shard_index in range(NUM_SHARDS):
        shard_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=shard_filename(shard_index),
            repo_type="dataset",
        )
        try:
            df = read_shard_dataframe(shard_path)
            total_rows += len(df)

            mask = (df["label"] == FLOWERING_LABEL) & df["species"].isin(target_species)
            kept = df.loc[mask, OUTPUT_COLUMNS]
            if not kept.empty:
                chunks.append(kept)
                kept_rows += len(kept)
        finally:
            evict_cached_shard(shard_path)

        shard_number = shard_index + 1
        if shard_number % PROGRESS_EVERY == 0 or shard_number == NUM_SHARDS:
            elapsed = time.time() - start_time
            print(
                f"Shard {shard_number}/{NUM_SHARDS} | "
                f"rows scanned: {total_rows:,} | "
                f"flowering kept: {kept_rows:,} | "
                f"elapsed: {elapsed:.1f}s"
            )

    result = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=OUTPUT_COLUMNS)
    result.to_parquet(OUTPUT_PATH, index=False, compression="snappy")

    print("-" * 60)
    print(f"Done! {len(result):,} flowering records saved to {OUTPUT_PATH}")
    print(f"Species kept: {result['species'].nunique()} / {len(target_species)} targets")


if __name__ == "__main__":
    main()
