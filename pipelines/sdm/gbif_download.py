import argparse
import csv
import os
import sys
import zipfile
from pathlib import Path
import pandas as pd
import requests

API = "https://api.gbif.org/v1/occurrence/download"


def predicate(keys):
    """CONUS, post-2013, human observations, coords within 1 km — matches Part 1 Step 3."""
    return {"type": "and", "predicates": [
        {"type": "in", "key": "TAXON_KEY", "values": [str(k) for k in keys]},
        {"type": "equals", "key": "BASIS_OF_RECORD", "value": "HUMAN_OBSERVATION"},
        {"type": "equals", "key": "COUNTRY", "value": "US"},
        {"type": "equals", "key": "HAS_COORDINATE", "value": "true"},
        {"type": "equals", "key": "HAS_GEOSPATIAL_ISSUE", "value": "false"},
        {"type": "greaterThanOrEquals", "key": "YEAR", "value": "2013"},
        {"type": "lessThanOrEquals", "key": "COORDINATE_UNCERTAINTY_IN_METERS", "value": "1000"},
        {"type": "greaterThanOrEquals", "key": "DECIMAL_LATITUDE", "value": "24.5"},
        {"type": "lessThanOrEquals", "key": "DECIMAL_LATITUDE", "value": "49.5"},
        {"type": "greaterThanOrEquals", "key": "DECIMAL_LONGITUDE", "value": "-125"},
        {"type": "lessThanOrEquals", "key": "DECIMAL_LONGITUDE", "value": "-66.5"},
    ]}


def submit(args):
    user, pwd, email = os.environ["GBIF_USER"], os.environ["GBIF_PWD"], os.environ["GBIF_EMAIL"]
    m = pd.read_csv(args.keys)
    keys = m.acceptedUsageKey.fillna(m.usageKey).dropna().astype(int).unique()
    print(f"{len(keys)} taxon keys from {args.keys}")
    body = {"creator": user, "notificationAddresses": [email], "sendNotification": True,
            "format": "SIMPLE_CSV", "predicate": predicate(keys)}
    r = requests.post(f"{API}/request", auth=(user, pwd), json=body,
                      headers={"Content-Type": "application/json"})
    if r.status_code not in (200, 201):
        sys.exit(f"FAILED {r.status_code}: {r.text}")
    print(f"submitted: key={r.text.strip()}")
    print(f"status: python pipelines/sdm/gbif_download.py status {r.text.strip()}")


def status(args):
    d = requests.get(f"{API}/{args.key}").json()
    print(d.get("status"), f"records={d.get('totalRecords')}", d.get("downloadLink") or "")


def fetch(args):
    d = requests.get(f"{API}/{args.key}").json()
    assert d["status"] == "SUCCEEDED", d["status"]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(d["downloadLink"], stream=True) as r, open(out, "wb") as f:
        for chunk in r.iter_content(1 << 22):
            f.write(chunk)
    print(f"fetched {out} ({out.stat().st_size/1e9:.2f} GB)")


def convert(args):
    """SIMPLE_CSV zip -> the extract schema (pollinator_species, lat, lon, doy, year)."""
    zf = zipfile.ZipFile(args.zip)
    member = [n for n in zf.namelist() if n.endswith(".csv")][0]
    with zf.open(member) as f, open(args.out, "w", newline="") as g:
        w = csv.writer(g)
        w.writerow(["pollinator_species", "lat", "lon", "doy", "year"])
        rd = pd.read_csv(f, sep="\t", chunksize=1_000_000, on_bad_lines="skip",
                         usecols=["species", "decimalLatitude", "decimalLongitude", "eventDate"])
        n = 0
        for ch in rd:
            ch = ch.dropna(subset=["species", "decimalLatitude", "decimalLongitude", "eventDate"])
            dt = pd.to_datetime(ch.eventDate, errors="coerce", format="mixed", utc=True)
            ch = ch[dt.notna()]
            dt = dt[dt.notna()]
            for row in zip(ch.species, ch.decimalLatitude, ch.decimalLongitude, dt.dt.dayofyear, dt.dt.year):
                w.writerow(row)
            n += len(ch)
            print(f"  {n:,}", flush=True)
    print(f"wrote {args.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("submit", help="needs GBIF_USER / GBIF_PWD / GBIF_EMAIL env vars")
    s.add_argument("--keys", required=True, help="gbif_key_match.csv (usageKey / acceptedUsageKey)")
    s.set_defaults(fn=submit)
    s = sub.add_parser("status")
    s.add_argument("key")
    s.set_defaults(fn=status)
    s = sub.add_parser("fetch")
    s.add_argument("key")
    s.add_argument("--out", required=True)
    s.set_defaults(fn=fetch)
    s = sub.add_parser("convert")
    s.add_argument("zip")
    s.add_argument("--out", required=True)
    s.set_defaults(fn=convert)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
