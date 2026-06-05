# PhenoField Species Tally

Stream the [dcher95/phenofield](https://huggingface.co/datasets/dcher95/phenofield) train split from Hugging Face, tally species frequencies with a `collections.Counter`, and report the top 55 most populated species.

## Requirements

- Python 3.10+
- `pandas`
- `pyarrow`
- `datasets`
- `huggingface_hub`
- `python-docx` (for Word export only)

```bash
pip install pandas pyarrow datasets huggingface_hub python-docx
```

## Cache location

All Hugging Face and datasets cache is directed to scratch (not home):

- `HF_HOME=/scratch/ariana.l/hf_cache`
- `HF_DATASETS_CACHE=/scratch/ariana.l/hf_cache/datasets`

## Usage

### Run the species tally

```bash
cd /home/ariana.l/CfE2026CVforEcology/rawplantsdata
nohup python3 -u download_dataset.py 2>&1 | tee species_tally_top55.log &
```

### Export results to Parquet and Word

After the tally completes, export the ranked list:

```bash
python3 export_top_species.py
```

This produces:

- `top55_species.parquet` — ranked table (`rank`, `species`, `count`)
- `top55_species.docx` — formatted report with summary and table

## Output files

| File | Description |
|------|-------------|
| `species_tally_top55.log` | Full console log from the tally run |
| `top55_species.parquet` | Top 55 species in Parquet format |
| `top55_species.docx` | Top 55 species in Word format |
| `download_dataset.py` | Main streaming / tally script |
| `export_top_species.py` | Log parser and exporter |

## Latest results (train split)

- **Total rows:** 2,650,448
- **Unique species:** 6,466
- **Top 55 cumulative records:** 463,105 (17.5% of dataset)
- **55th most populated species:** Acer rubrum (5,650 records)

## `download_dataset.py`

```python
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
```

## `export_top_species.py`

```python
"""Export top species rankings to Parquet and Word formats."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from docx import Document
from docx.shared import Inches

LOG_PATH = Path("species_tally_top55.log")
PARQUET_PATH = Path("top55_species.parquet")
DOCX_PATH = Path("top55_species.docx")


def parse_log(log_path: Path) -> tuple[pd.DataFrame, dict[str, str | int | float]]:
    text = log_path.read_text()
    summary: dict[str, str | int | float] = {}

    finished = re.search(
        r"Finished\. Total rows: ([\d,]+) \| Unique species: ([\d,]+)",
        text,
    )
    if finished:
        summary["total_rows"] = int(finished.group(1).replace(",", ""))
        summary["unique_species"] = int(finished.group(2).replace(",", ""))

    cumulative = re.search(
        r"Cumulative records in top 55 species: ([\d,]+) \(([\d.]+)% of ([\d,]+) total rows\)",
        text,
    )
    if cumulative:
        summary["cumulative_top55_records"] = int(cumulative.group(1).replace(",", ""))
        summary["cumulative_top55_pct"] = float(cumulative.group(2))
        summary["cumulative_top55_total_rows"] = int(cumulative.group(3).replace(",", ""))

    rank55 = re.search(
        r"55th most populated species: (.+) \(([\d,]+) records\)",
        text,
    )
    if rank55:
        summary["rank_55_species"] = rank55.group(1).strip()
        summary["rank_55_count"] = int(rank55.group(2).replace(",", ""))

    rows: list[dict[str, object]] = []
    for match in re.finditer(
        r"^\s*(\d+)\.\s+(.+?)\s+Count:\s+([\d,]+)\s*$",
        text,
        re.MULTILINE,
    ):
        rows.append(
            {
                "rank": int(match.group(1)),
                "species": match.group(2).strip(),
                "count": int(match.group(3).replace(",", "")),
            }
        )

    if not rows:
        raise ValueError(f"No ranked species found in {log_path}")

    df = pd.DataFrame(rows).sort_values("rank").reset_index(drop=True)
    return df, summary


def export_parquet(df: pd.DataFrame, summary: dict[str, str | int | float], path: Path) -> None:
    metadata = {k: str(v) for k, v in summary.items()}
    df.to_parquet(path, index=False, compression="snappy")
    print(f"Wrote {path} ({len(df)} rows, metadata: {metadata})")


def export_docx(df: pd.DataFrame, summary: dict[str, str | int | float], path: Path) -> None:
    doc = Document()
    doc.add_heading("Top 55 Most Populated Species — PhenoField Train Set", level=1)
    doc.add_paragraph(
        "Dataset: dcher95/phenofield (train split). "
        "Species frequencies tallied by streaming 107 shards."
    )

    if summary:
        doc.add_heading("Summary", level=2)
        if "total_rows" in summary:
            doc.add_paragraph(f"Total rows processed: {summary['total_rows']:,}")
        if "unique_species" in summary:
            doc.add_paragraph(f"Unique species: {summary['unique_species']:,}")
        if "cumulative_top55_records" in summary:
            doc.add_paragraph(
                f"Cumulative records in top 55 species: "
                f"{summary['cumulative_top55_records']:,} "
                f"({summary.get('cumulative_top55_pct', 0):.1f}% of dataset)"
            )
        if "rank_55_species" in summary:
            doc.add_paragraph(
                f"55th most populated species: {summary['rank_55_species']} "
                f"({summary.get('rank_55_count', 0):,} records)"
            )

    doc.add_heading("Ranked Species List", level=2)
    table = doc.add_table(rows=1, cols=3)
    table.style = "Table Grid"
    headers = table.rows[0].cells
    headers[0].text = "Rank"
    headers[1].text = "Species"
    headers[2].text = "Count"

    for _, row in df.iterrows():
        cells = table.add_row().cells
        cells[0].text = str(int(row["rank"]))
        cells[1].text = str(row["species"])
        cells[2].text = f"{int(row['count']):,}"

    for column in table.columns:
        for cell in column.cells:
            cell.width = Inches(1.5)

    doc.save(path)
    print(f"Wrote {path}")


def main() -> None:
    if not LOG_PATH.exists():
        raise FileNotFoundError(f"Log file not found: {LOG_PATH}")

    df, summary = parse_log(LOG_PATH)
    export_parquet(df, summary, PARQUET_PATH)
    export_docx(df, summary, DOCX_PATH)


if __name__ == "__main__":
    main()
```
