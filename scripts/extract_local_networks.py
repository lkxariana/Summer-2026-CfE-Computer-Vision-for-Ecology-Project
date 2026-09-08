"""Per-network membership for the local-network completion evaluation (plan §1.5).

`edges.parquet` keeps only the source *dataset* per edge, so it cannot say which plants and
pollinators were surveyed together in one field network. The raw GloBI export can: for
network-style datasets (Web of Life and similar), `referenceCitation` identifies the study and
`localityName` / coordinates the site, and every record of a network carries the same citation.

Streams the 3 GB export once, keeps rows from network-style datasets whose plant and pollinator both
resolve to the modelled universe, and writes one row per (network, plant, pollinator) with the
network's coordinates and year range. A "network" is (sourceCitation dataset, referenceCitation,
localityName). Networks with fewer than `--min-plants` plants or `--min-polls` pollinators are dropped.

Output: data/network/local_networks.parquet and a summary printed to stdout.
"""
import argparse
import gzip
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NETWORK_DATASETS = ("web-of-life", "Plant-Pollinator-Web", "bee-interaction-database", "Pollination-Collection",
                    "guzman2022", "CropPol", "lamanna2020", "dorey2023")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--globi", default=ROOT / "artifacts/external/globi_2026-08-26.csv.gz")
    ap.add_argument("--min-plants", type=int, default=10)
    ap.add_argument("--min-polls", type=int, default=10)
    ap.add_argument("--chunk", type=int, default=500_000)
    args = ap.parse_args()

    u = json.load(open(ROOT / "data/network/modelled_universe.json"))
    plants = {p["label"] for p in u["plants"]}; polls = {p["label"] for p in u["pollinators"]}
    cols = ["sourceTaxonName", "targetTaxonName", "interactionTypeName", "decimalLatitude", "decimalLongitude",
            "localityName", "eventDate", "referenceCitation", "sourceCitation"]
    keep = []
    n_rows = 0
    for ch in pd.read_csv(args.globi, usecols=cols, chunksize=args.chunk, dtype=str, compression="gzip", on_bad_lines="skip"):
        n_rows += len(ch)
        src = ch["sourceCitation"].fillna("") + " " + ch["referenceCitation"].fillna("")
        m = src.str.contains("|".join(NETWORK_DATASETS), regex=True, na=False)
        ch = ch[m]
        if ch.empty:
            continue
        # orient by membership: the plant is whichever side is in the plant universe
        a_pl = ch.sourceTaxonName.isin(plants) & ch.targetTaxonName.isin(polls)
        b_pl = ch.targetTaxonName.isin(plants) & ch.sourceTaxonName.isin(polls)
        pa = ch[a_pl].rename(columns={"sourceTaxonName": "plant", "targetTaxonName": "pollinator"})
        pb = ch[b_pl].rename(columns={"targetTaxonName": "plant", "sourceTaxonName": "pollinator"})
        keep.append(pd.concat([pa, pb])[["plant", "pollinator", "decimalLatitude", "decimalLongitude", "localityName",
                                          "eventDate", "referenceCitation", "sourceCitation"]])
        if n_rows % (args.chunk * 10) == 0:
            print(f"  {n_rows:,} rows scanned, {sum(len(k) for k in keep):,} kept", flush=True)
    d = pd.concat(keep, ignore_index=True)
    print(f"[scan] {n_rows:,} rows; {len(d):,} network-dataset records with both taxa in the universe", flush=True)

    d["dataset"] = d.sourceCitation.fillna("").str.extract(r"(web-of-life|Plant-Pollinator-Web|bee-interaction-database|Pollination-Collection|guzman2022|CropPol|lamanna2020|dorey2023)")[0]
    d["network"] = d["dataset"].fillna("?") + " | " + d.referenceCitation.fillna("?").str.slice(0, 120) + " | " + d.localityName.fillna("?")
    d["lat"] = pd.to_numeric(d.decimalLatitude, errors="coerce"); d["lon"] = pd.to_numeric(d.decimalLongitude, errors="coerce")
    d["year"] = pd.to_datetime(d.eventDate, errors="coerce").dt.year
    g = d.groupby("network")
    net = pd.DataFrame({"n_plants": g.plant.nunique(), "n_polls": g.pollinator.nunique(), "n_pairs": g.size(),
                        "lat": g.lat.median(), "lon": g.lon.median(), "has_coords": g.lat.apply(lambda s: s.notna().any()),
                        "year_min": g.year.min(), "year_max": g.year.max(), "dataset": g.dataset.first()})
    ok = net[(net.n_plants >= args.min_plants) & (net.n_polls >= args.min_polls)]
    print(f"[networks] {len(net)} (dataset, citation, locality) groups; {len(ok)} with >= {args.min_plants} plants and "
          f">= {args.min_polls} pollinators; {int(ok.has_coords.sum())} of those carry coordinates", flush=True)
    print(ok.sort_values("n_pairs", ascending=False).head(25).to_string(), flush=True)
    d = d[d.network.isin(ok.index)]
    d[["network", "dataset", "plant", "pollinator", "lat", "lon", "year", "referenceCitation", "localityName"]].to_parquet(
        ROOT / "data/network/local_networks.parquet", index=False)
    ok.to_csv(ROOT / "data/network/local_networks_summary.csv")
    print(f"[wrote] data/network/local_networks.parquet ({len(d):,} rows), local_networks_summary.csv", flush=True)


if __name__ == "__main__":
    main()
