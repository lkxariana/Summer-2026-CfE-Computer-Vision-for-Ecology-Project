#!/usr/bin/env python
"""Submit a GBIF predicate download for CONUS, post-2013, Research-Grade observations
across five orders (Hemiptera, Passeriformes, Thysanoptera, Neuroptera, Chiroptera).

GBIF has no "research grade" field -- that is an iNaturalist concept. iNaturalist only
publishes Research Grade records to GBIF, so we restrict to its datasetKey. Comment that
predicate out (and uncomment the basisOfRecord one) to broaden to all human observations.

Requires a free GBIF.org account. Set creds via env:
    export GBIF_USER=... GBIF_PWD=... GBIF_EMAIL=...
    python gbif_download_pollinators.py
Then poll the returned download key until status==SUCCEEDED and grab the .zip URL.
"""
import os
import sys
import json
import requests

GBIF_USER  = os.environ["GBIF_USER"]
GBIF_PWD   = os.environ["GBIF_PWD"]
GBIF_EMAIL = os.environ["GBIF_EMAIL"]

TAXON_KEYS = ["809", "729", "1228", "1501", "734"]  # Hemiptera, Passeriformes, Thysanoptera, Neuroptera, Chiroptera
INAT_DATASET_KEY = "50c9509d-22c7-4a22-a47d-8c48425ef4a7"  # iNaturalist Research-grade Observations
YEAR_MIN = "2013"  # 2013 inclusive

predicate = {
    "type": "and",
    "predicates": [
        {"type": "in", "key": "TAXON_KEY", "values": TAXON_KEYS},
        # Research-Grade equivalent = HUMAN_OBSERVATION across ALL GBIF sources, matching the
        # Part 1 Step-3 methodology (NOT iNat-only -- that would discard eBird birds and worsen
        # the iNaturalist leakage flagged in 04-data.md). To restrict to iNat only instead,
        # swap this for: {"type": "equals", "key": "DATASET_KEY", "value": INAT_DATASET_KEY}
        {"type": "equals", "key": "BASIS_OF_RECORD", "value": "HUMAN_OBSERVATION"},
        {"type": "equals", "key": "COUNTRY", "value": "US"},
        {"type": "equals", "key": "HAS_COORDINATE", "value": "true"},
        {"type": "equals", "key": "HAS_GEOSPATIAL_ISSUE", "value": "false"},
        {"type": "greaterThanOrEquals", "key": "YEAR", "value": YEAR_MIN},
        # Coordinate uncertainty < 1000 m, matching Part 1 Step 3. NOTE: this drops records
        # with a NULL uncertainty value (a large share on GBIF) -- comment out to keep them.
        {"type": "lessThanOrEquals", "key": "COORDINATE_UNCERTAINTY_IN_METERS", "value": "1000"},
        # CONUS bounding box (excludes AK/HI/territories)
        {"type": "greaterThanOrEquals", "key": "DECIMAL_LATITUDE",  "value": "24.5"},
        {"type": "lessThanOrEquals",    "key": "DECIMAL_LATITUDE",  "value": "49.5"},
        {"type": "greaterThanOrEquals", "key": "DECIMAL_LONGITUDE", "value": "-125"},
        {"type": "lessThanOrEquals",    "key": "DECIMAL_LONGITUDE", "value": "-66.5"},
    ],
}

body = {
    "creator": GBIF_USER,
    "notificationAddresses": [GBIF_EMAIL],
    "sendNotification": True,
    "format": "SIMPLE_CSV",  # or "DWCA" for the full Darwin Core Archive
    "predicate": predicate,
}

resp = requests.post(
    "https://api.gbif.org/v1/occurrence/download/request",
    auth=(GBIF_USER, GBIF_PWD),
    json=body,
    headers={"Content-Type": "application/json"},
)
if resp.status_code in (200, 201):
    key = resp.text.strip()
    print(f"Download submitted. key={key}")
    print(f"Status:  https://api.gbif.org/v1/occurrence/download/{key}")
    print(f"Portal:  https://www.gbif.org/occurrence/download/{key}")
else:
    print(f"FAILED {resp.status_code}: {resp.text}", file=sys.stderr)
    sys.exit(1)
