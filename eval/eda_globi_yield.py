import sys
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from antheia.config import load_config
from antheia.globi import binomial
from antheia.store import FeatureStore

RAW = "/scratch/ariana.l/CfE2026CVforEcology/rawpollinatordata/interactions.csv.gz"
LAT, LON = (24.0, 49.5), (-125.0, -66.0)
FLOWER = ["visitsFlowersOf", "visits", "pollinates"]
COLS = ["sourceTaxonName", "sourceTaxonRank", "sourceTaxonIds", "sourceOccurrenceId",
        "targetTaxonName", "targetTaxonRank", "targetTaxonIds",
        "interactionTypeName", "argumentTypeId", "sourceBasisOfRecordName",
        "decimalLatitude", "decimalLongitude", "sourceCitation", "referenceDoi"]

cfg = load_config()
store = FeatureStore(cfg)
plants, polls = set(store.plants), set(store.polls)
pl_bin = {binomial(x): x for x in plants}; pl_bin.pop(None, None)
po_bin = {binomial(x): x for x in polls}; po_bin.pop(None, None)
pl_gen = {s.split()[0] for s in plants}
po_gen = {s.split()[0] for s in polls}

tot = Counter(); arg = Counter(); rank_pairs = Counter(); types_all = Counter()
drop_reason = Counter(); id_ns = Counter(); occ_dupe = Counter()
recover_genus = 0; n_flower = 0

for chunk in pd.read_csv(RAW, usecols=COLS, chunksize=1_000_000, low_memory=False, on_bad_lines="skip"):
    c = chunk[(chunk["decimalLatitude"].between(*LAT)) & (chunk["decimalLongitude"].between(*LON))]
    tot["conus_records"] += len(c)
    for t, n in c["interactionTypeName"].value_counts().items():
        types_all[t] += n
    f = c[c["interactionTypeName"].isin(FLOWER)].dropna(subset=["sourceTaxonName", "targetTaxonName"])
    f = f[f["sourceTaxonName"] != f["targetTaxonName"]]
    n_flower += len(f)
    for a, n in f["argumentTypeId"].fillna("(none)").value_counts().items():
        arg[a] += n
    for r, n in f[["sourceTaxonRank", "targetTaxonRank"]].fillna("(none)").apply(
            lambda r: f"{r.sourceTaxonRank}|{r.targetTaxonRank}", axis=1).value_counts().items():
        rank_pairs[r] += n
    for ids in f["targetTaxonIds"].dropna().head(2000):
        for tok in str(ids).split("|"):
            if ":" in tok:
                id_ns[tok.split(":")[0]] += 1
    for o, n in f["sourceOccurrenceId"].dropna().value_counts().items():
        occ_dupe[o] += n

    s, t = f["sourceTaxonName"], f["targetTaxonName"]
    s_pl = s.isin(plants) | s.map(binomial).isin(pl_bin)
    s_po = s.isin(polls) | s.map(binomial).isin(po_bin)
    t_pl = t.isin(plants) | t.map(binomial).isin(pl_bin)
    t_po = t.isin(polls) | t.map(binomial).isin(po_bin)
    usable = (s_pl & t_po) | (s_po & t_pl)
    tot["usable"] += int(usable.sum())
    d = f[~usable]
    drop_reason["dropped_total"] += len(d)
    ds, dt = d["sourceTaxonName"], d["targetTaxonName"]
    dt_genus_only = dt.map(binomial).isna()
    drop_reason["target_genus_only"] += int(dt_genus_only.sum())
    drop_reason["source_genus_only"] += int(ds.map(binomial).isna().sum())
    # Could a genus-level match rescue it? (pollinator side known, plant side only to genus)
    rescue = (ds.isin(polls) | ds.map(binomial).isin(po_bin)) & dt.str.split().str[0].isin(pl_gen)
    recover_genus += int(rescue.sum())

print("=== CONUS records:", f"{tot['conus_records']:,}")
print("\n=== all interaction types (top 12):")
for t, n in types_all.most_common(12):
    print(f"  {n:>10,}  {t}")
print(f"\n=== flower-visitation records (3 types): {n_flower:,}")
print("=== argumentTypeId (SUPPORTS vs REFUTES):")
for a, n in arg.most_common():
    print(f"  {n:>10,}  {a}")
print("\n=== source|target taxon rank pairs (top 8):")
for r, n in rank_pairs.most_common(8):
    print(f"  {n:>10,}  {r}")
print("\n=== id namespaces on targetTaxonIds (sampled):")
for k, n in id_ns.most_common(8):
    print(f"  {n:>8,}  {k}")
print(f"\n=== usable under current logic: {tot['usable']:,}")
for k, v in drop_reason.items():
    print(f"  {k}: {v:,}")
print(f"  rescuable by GENUS-level plant match: {recover_genus:,}")
dupes = sum(v - 1 for v in occ_dupe.values() if v > 1)
print(f"\n=== duplicate sourceOccurrenceId rows: {dupes:,} extra rows over {len(occ_dupe):,} unique ids")
