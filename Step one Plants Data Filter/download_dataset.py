"""Stream phenofield train shards and tally species frequencies."""

from __future__ import annotations

import os

# Route all Hugging Face / datasets cache to scratch before any HF imports.
os.environ["HF_HOME"] = "/scratch/ariana.l/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "/scratch/ariana.l/hf_cache/datasets"

import io
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from datasets import Dataset
from huggingface_hub import hf_hub_download

REPO_ID = "dcher95/phenofield"
NUM_SHARDS = 107
TARGET_COLUMNS = ["species", "lat", "lon", "label"]
PARQUET_COLUMNS = ["species", "latitude", "longitude", "label"]
COLUMN_RENAME = {"latitude": "lat", "longitude": "lon"}
PROGRESS_EVERY = 5
TOP_SPECIES = 55


def shard_filename(shard_index: int) -> str:
    return f"train/data-{shard_index:05d}-of-00107.arrow"


def read_shard_dataframe(shard_path: str) -> pd.DataFrame:
    """Read one shard through pandas.read_parquet with only the four target columns."""
    table = Dataset.from_file(shard_path).data.table.select(PARQUET_COLUMNS)
    buffer = io.BytesIO()
    pq.write_table(table, buffer)
    buffer.seek(0)
    df = pd.read_parquet(buffer, columns=PARQUET_COLUMNS)
    return df.rename(columns=COLUMN_RENAME)


def evict_cached_shard(shard_path: str) -> None:
    """Drop the shard blob after processing so cache stays bounded."""
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

    print(f"Streaming {NUM_SHARDS} train shards from {REPO_ID}...")
    print(f"HF_HOME: {os.environ['HF_HOME']}")
    print(f"HF_DATASETS_CACHE: {os.environ['HF_DATASETS_CACHE']}")
    print(f"Target columns: {TARGET_COLUMNS}")
    print(f"Progress updates every {PROGRESS_EVERY} shards. Press Ctrl+C to stop.")
    print("-" * 60)

    species_counts: Counter[str] = Counter()
    total_rows = 0
    start_time = time.time()

    for shard_index in range(NUM_SHARDS):
        shard_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=shard_filename(shard_index),
            repo_type="dataset",
        )
        try:
            df = read_shard_dataframe(shard_path)
            species_counts.update(df["species"].dropna().astype(str))
            total_rows += len(df)
        finally:
            evict_cached_shard(shard_path)

        shard_number = shard_index + 1
        if shard_number % PROGRESS_EVERY == 0 or shard_number == NUM_SHARDS:
            elapsed = time.time() - start_time
            print(
                f"Shard {shard_number}/{NUM_SHARDS} | "
                f"rows so far: {total_rows:,} | "
                f"unique species: {len(species_counts):,} | "
                f"elapsed: {elapsed:.1f}s"
            )

    print("-" * 60)
    print(f"Finished. Total rows: {total_rows:,} | Unique species: {len(species_counts):,}")

    top_species = species_counts.most_common(TOP_SPECIES)
    cumulative = sum(count for _, count in top_species)
    pct = 100 * cumulative / total_rows if total_rows else 0

    print(f"\n--- TOP {TOP_SPECIES} MOST POPULATED SPECIES ---")
    for rank, (species, count) in enumerate(top_species, 1):
        print(f"{rank:2d}. {species:<40} Count: {count:,}")

    print("-" * 60)
    print(
        f"Cumulative records in top {TOP_SPECIES} species: "
        f"{cumulative:,} ({pct:.1f}% of {total_rows:,} total rows)"
    )
    if len(top_species) >= TOP_SPECIES:
        name, count = top_species[TOP_SPECIES - 1]
        print(f"55th most populated species: {name} ({count:,} records)")


if __name__ == "__main__":
    main()
