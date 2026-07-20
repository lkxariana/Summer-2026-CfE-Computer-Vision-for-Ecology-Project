"""Build a flat pollinator observations table from the GBIF occurrence dump.

Streams the GBIF TSV in chunks, keeps only rows with complete species,
coordinates, and date fields, computes day-of-year from (year, month, day),
and writes (pollinator_species, lat, lon, doy, year) per observation.

The taxonomic, geographic, temporal, and coordinate-accuracy filters are
applied at GBIF download time and do not need to be re-applied here.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Path to the unzipped GBIF TSV file. The filename comes from the GBIF
# download request and will look like 0047411-260519110011954.csv.
INPUT_PATH = Path("/scratch/ariana.l/0047411-260519110011954.csv")
OUTPUT_PATH = Path("/scratch/ariana.l/pollinator_observations.csv")
CHUNK_SIZE = 100_000

COLUMNS_NEEDED = [
    "species",
    "decimalLatitude",
    "decimalLongitude",
    "day",
    "month",
    "year",
]


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"GBIF dump not found at {INPUT_PATH}. "
            f"Request a download at https://www.gbif.org/occurrence/search "
            f"with the filters described in the README, then unzip the result."
        )

    print(f"Reading {INPUT_PATH} in chunks of {CHUNK_SIZE:,} rows...")
    print("-" * 60)

    chunks: list[pd.DataFrame] = []
    for i, chunk in enumerate(
        pd.read_csv(
            INPUT_PATH,
            sep="\t",
            usecols=COLUMNS_NEEDED,
            chunksize=CHUNK_SIZE,
            low_memory=False,
        )
    ):
        if i % 10 == 0:
            print(f"Processing chunk {i}...")

        # Drop rows with any missing field; GBIF stores partial dates as
        # year-only eventDate in some records, so day/month presence is the
        # signal that the date is complete.
        chunk = chunk.dropna(
            subset=[
                "species",
                "decimalLatitude",
                "decimalLongitude",
                "day",
                "month",
                "year",
            ]
        )
        chunks.append(chunk)

    df = pd.concat(chunks)

    df["day"] = df["day"].astype(int)
    df["month"] = df["month"].astype(int)
    df["year"] = df["year"].astype(int)
    df["doy"] = pd.to_datetime(df[["year", "month", "day"]]).dt.dayofyear

    df = df.rename(
        columns={
            "species": "pollinator_species",
            "decimalLatitude": "lat",
            "decimalLongitude": "lon",
        }
    )

    output = df[["pollinator_species", "lat", "lon", "doy", "year"]]

    print("-" * 60)
    print(f"Records with complete dates: {len(output):,}")
    print(f"Unique pollinator species: {output['pollinator_species'].nunique():,}")

    output.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved pollinator observations to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
