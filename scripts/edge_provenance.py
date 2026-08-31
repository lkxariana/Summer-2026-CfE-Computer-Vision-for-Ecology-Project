import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, "/scratch/cher/Summer-2026-CfE-Computer-Vision-for-Ecology-Project/src")
from antheia.config import load_config
from antheia.globi import binomial
from antheia.store import FeatureStore

RAW = "/scratch/ariana.l/CfE2026CVforEcology/rawpollinatordata/interactions.csv.gz"
TYPES = ["visitsFlowersOf", "visits", "pollinates"]
LAT = (24.0, 49.5)
LON = (-125.0, -66.0)

cfg = load_config()
store = FeatureStore(cfg)
plants, polls = set(store.plants), set(store.polls)
pl_bin = {binomial(x): x for x in plants}
po_bin = {binomial(x): x for x in polls}
pl_bin.pop(None, None)
po_bin.pop(None, None)

usecols = ["sourceTaxonName", "targetTaxonName", "interactionTypeName",
           "decimalLatitude", "decimalLongitude",
           "sourceNamespace", "sourceCitation", "sourceBasisOfRecordName"]
type_counts_all = {}
ns_counts = {}
basis_counts = {}
edge_rows = []
n_records = n_inat = 0

for chunk in pd.read_csv(RAW, usecols=usecols, chunksize=1_000_000, low_memory=False, on_bad_lines="skip"):
    conus = chunk[(chunk["decimalLatitude"].between(*LAT)) & (chunk["decimalLongitude"].between(*LON))]
    for t, c in conus["interactionTypeName"].value_counts().items():
        type_counts_all[t] = type_counts_all.get(t, 0) + c
    d = conus[conus["interactionTypeName"].isin(TYPES)].dropna(subset=["sourceTaxonName", "targetTaxonName"])
    d = d[d["sourceTaxonName"] != d["targetTaxonName"]]
    if not len(d):
        continue
    inat = (d["sourceNamespace"].fillna("").str.contains("inat", case=False) |
            d["sourceCitation"].fillna("").str.contains("inaturalist", case=False))
    n_records += len(d)
    n_inat += int(inat.sum())
    for ns, c in d["sourceNamespace"].fillna("?").value_counts().items():
        ns_counts[ns] = ns_counts.get(ns, 0) + c
    for b, c in d["sourceBasisOfRecordName"].fillna("?").value_counts().items():
        basis_counts[b] = basis_counts.get(b, 0) + c

    src, tgt = d["sourceTaxonName"], d["targetTaxonName"]
    src_pl = src.where(src.isin(plants)).fillna(src.map(binomial).map(pl_bin))
    src_po = src.where(src.isin(polls)).fillna(src.map(binomial).map(po_bin))
    tgt_pl = tgt.where(tgt.isin(plants)).fillna(tgt.map(binomial).map(pl_bin))
    tgt_po = tgt.where(tgt.isin(polls)).fillna(tgt.map(binomial).map(po_bin))
    asis = src_pl.notna() & tgt_po.notna()
    swap = src_po.notna() & tgt_pl.notna() & ~asis
    part = pd.concat([
        pd.DataFrame({"plant": src_pl[asis], "pollinator": tgt_po[asis], "inat": inat[asis]}),
        pd.DataFrame({"plant": tgt_pl[swap], "pollinator": src_po[swap], "inat": inat[swap]}),
    ], ignore_index=True)
    if len(part):
        edge_rows.append(part.groupby(["plant", "pollinator"])["inat"].agg(n="size", n_inat="sum").reset_index())

agg = pd.concat(edge_rows).groupby(["plant", "pollinator"])[["n", "n_inat"]].sum().reset_index()
print(f"CONUS pollination records (3 types): {n_records:,}")
print(f"  iNaturalist-sourced: {n_inat:,} ({n_inat / n_records:.1%})")
print("\nTop sourceNamespaces:")
for ns, c in sorted(ns_counts.items(), key=lambda x: -x[1])[:12]:
    print(f"  {c:>9,}  {ns}")
print("\nBasisOfRecord:")
for b, c in sorted(basis_counts.items(), key=lambda x: -x[1])[:8]:
    print(f"  {c:>9,}  {b}")
print("\nAll CONUS interaction types (top 15, pre-filter):")
for t, c in sorted(type_counts_all.items(), key=lambda x: -x[1])[:15]:
    print(f"  {c:>9,}  {t}")

print(f"\nOriented edges: {len(agg):,}")
only_inat = (agg["n_inat"] == agg["n"]).mean()
has_noninat = (agg["n_inat"] < agg["n"]).mean()
print(f"  edges supported ONLY by iNat records:   {only_inat:.1%}")
print(f"  edges with >=1 non-iNat record:         {has_noninat:.1%}")
print(f"  edges with >=2 records:                 {(agg['n'] >= 2).mean():.1%}")
print(f"  edges with >=2 records incl >=1 noniNat: {((agg['n'] >= 2) & (agg['n_inat'] < agg['n'])).mean():.1%}")
agg.to_csv("/scratch/cher/Summer-2026-CfE-Computer-Vision-for-Ecology-Project/artifacts/edge_provenance_v1.csv", index=False)
