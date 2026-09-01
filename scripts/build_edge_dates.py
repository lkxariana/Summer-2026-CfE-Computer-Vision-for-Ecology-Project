import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from antheia.config import load_config
from antheia.globi import binomial
from antheia.store import FeatureStore

RAW = "/scratch/ariana.l/CfE2026CVforEcology/rawpollinatordata/interactions.csv.gz"
TYPES = ["visitsFlowersOf", "visits", "pollinates"]
LAT, LON = (24.0, 49.5), (-125.0, -66.0)


def main():
    ap = argparse.ArgumentParser(description="First-documentation year per oriented edge (for temporal holdout)")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    store = FeatureStore(cfg)
    plants, polls = set(store.plants), set(store.polls)
    pl_bin = {binomial(x): x for x in plants}
    po_bin = {binomial(x): x for x in polls}
    pl_bin.pop(None, None)
    po_bin.pop(None, None)

    parts = []
    for chunk in pd.read_csv(RAW, usecols=["sourceTaxonName", "targetTaxonName", "interactionTypeName",
                                           "decimalLatitude", "decimalLongitude", "eventDate"],
                             chunksize=1_000_000, low_memory=False, on_bad_lines="skip"):
        d = chunk[(chunk["decimalLatitude"].between(*LAT)) & (chunk["decimalLongitude"].between(*LON))]
        d = d[d["interactionTypeName"].isin(TYPES)].dropna(subset=["sourceTaxonName", "targetTaxonName"])
        d = d[d["sourceTaxonName"] != d["targetTaxonName"]]
        if not len(d):
            continue
        year = pd.to_numeric(d["eventDate"].astype(str).str.slice(0, 4), errors="coerce")
        src, tgt = d["sourceTaxonName"], d["targetTaxonName"]
        src_pl = src.where(src.isin(plants)).fillna(src.map(binomial).map(pl_bin))
        src_po = src.where(src.isin(polls)).fillna(src.map(binomial).map(po_bin))
        tgt_pl = tgt.where(tgt.isin(plants)).fillna(tgt.map(binomial).map(pl_bin))
        tgt_po = tgt.where(tgt.isin(polls)).fillna(tgt.map(binomial).map(po_bin))
        asis = src_pl.notna() & tgt_po.notna()
        swap = src_po.notna() & tgt_pl.notna() & ~asis
        parts.append(pd.concat([
            pd.DataFrame({"plant": src_pl[asis], "pollinator": tgt_po[asis], "year": year[asis]}),
            pd.DataFrame({"plant": tgt_pl[swap], "pollinator": src_po[swap], "year": year[swap]}),
        ]))

    df = pd.concat(parts)
    out = df.groupby(["plant", "pollinator"])["year"].agg(
        first_year="min", last_year="max", n_dated="count").reset_index()
    path = cfg["edges"].parent / "edge_years_v1.csv"
    out.to_csv(path, index=False)
    print(f"edges with >=1 dated record: {len(out):,}")
    print(out["first_year"].describe().to_string())
    for y in (2015, 2018, 2020, 2022, 2023, 2024):
        print(f"  first documented <= {y}: {(out['first_year'] <= y).sum():,} | after: {(out['first_year'] > y).sum():,}")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
