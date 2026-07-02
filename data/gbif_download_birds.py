#!/usr/bin/env python
"""Submit a GBIF predicate download for CONUS, post-2013, human-observation BIRD records.

Separate from gbif_download_pollinators.py because birds behave oppositely to the
invertebrates+bats: eBird dominates GBIF birds and leaves coordinateUncertaintyInMeters
NULL, so the <1000 m uncertainty filter used for the inverts would silently drop ~99% of
records. We therefore OMIT that filter here -- at the 0.5deg analysis bin (~55 km) sub-km
uncertainty is irrelevant, and hasGeospatialIssue=false already removes broken coordinates.

SIZE WARNING: Passeriformes alone is ~555M CONUS human-observation records (mostly eBird).
This is a tens-of-GB SIMPLE_CSV archive that GBIF may take a while to prepare. If you only
need week-of-year activity curves, far fewer records would suffice -- but per the decision to
recover eBird coverage, this pulls the full set.

Requires a free GBIF.org account. Set creds via env:
    export GBIF_USER=... GBIF_PWD=... GBIF_EMAIL=...
    python gbif_download_birds.py
"""
import os
import sys
import requests

GBIF_USER  = os.environ["GBIF_USER"]
GBIF_PWD   = os.environ["GBIF_PWD"]
GBIF_EMAIL = os.environ["GBIF_EMAIL"]

# Passeriformes = the 173 flower-visiting perching taxa from the GloBI edge list.
# NOTE: the principal CONUS bird pollinators are hummingbirds = Trochilidae (key 5289,
# order Apodiformes, NOT Passeriformes). Part 1 already covered Trochilidae; uncomment
# below to fold it into this bird download too.
TAXON_KEYS = ["729"]            # Passeriformes
# TAXON_KEYS = ["729", "5289"]  # + Trochilidae (hummingbirds)

YEAR_MIN = "2013"  # 2013 inclusive

predicate = {
    "type": "and",
    "predicates": [
        {"type": "in", "key": "TAXON_KEY", "values": TAXON_KEYS},
        {"type": "equals", "key": "BASIS_OF_RECORD", "value": "HUMAN_OBSERVATION"},
        {"type": "equals", "key": "COUNTRY", "value": "US"},
        {"type": "equals", "key": "HAS_COORDINATE", "value": "true"},
        {"type": "equals", "key": "HAS_GEOSPATIAL_ISSUE", "value": "false"},
        {"type": "greaterThanOrEquals", "key": "YEAR", "value": YEAR_MIN},
        # NO coordinate-uncertainty filter (would drop ~99% of eBird; see module docstring).
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
    "format": "SIMPLE_CSV",  # keep SIMPLE_CSV -- DWCA at this volume would be enormous
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
    print(f"Bird download submitted. key={key}")
    print(f"Status:  https://api.gbif.org/v1/occurrence/download/{key}")
    print(f"Portal:  https://www.gbif.org/occurrence/download/{key}")
else:
    print(f"FAILED {resp.status_code}: {resp.text}", file=sys.stderr)
    sys.exit(1)
