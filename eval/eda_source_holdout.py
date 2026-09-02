import sys
from collections import defaultdict
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

cfg = load_config()
store = FeatureStore(cfg)
plants, polls = set(store.plants), set(store.polls)
pl_b = {binomial(x): x for x in plants}; pl_b.pop(None, None)
po_b = {binomial(x): x for x in polls}; po_b.pop(None, None)

edge_src = defaultdict(set)
for chunk in pd.read_csv(RAW, usecols=["sourceTaxonName", "targetTaxonName", "interactionTypeName",
                                       "decimalLatitude", "decimalLongitude", "sourceNamespace"],
                         chunksize=1_000_000, low_memory=False, on_bad_lines="skip"):
    c = chunk[(chunk["decimalLatitude"].between(*LAT)) & (chunk["decimalLongitude"].between(*LON))]
    f = c[c["interactionTypeName"].isin(FLOWER)].dropna(subset=["sourceTaxonName", "targetTaxonName"])
    f = f[f["sourceTaxonName"] != f["targetTaxonName"]]
    s, t = f["sourceTaxonName"], f["targetTaxonName"]
    s_pl = s.where(s.isin(plants)).fillna(s.map(binomial).map(pl_b))
    s_po = s.where(s.isin(polls)).fillna(s.map(binomial).map(po_b))
    t_pl = t.where(t.isin(plants)).fillna(t.map(binomial).map(pl_b))
    t_po = t.where(t.isin(polls)).fillna(t.map(binomial).map(po_b))
    asis = s_pl.notna() & t_po.notna()
    swap = s_po.notna() & t_pl.notna() & ~asis
    for m, pcol, qcol in ((asis, s_pl, t_po), (swap, t_pl, s_po)):
        sub = f[m]
        for p, q, ns in zip(pcol[m], qcol[m], sub["sourceNamespace"].fillna("(none)")):
            edge_src[(p, q)].add(ns)

print(f"total edges: {len(edge_src):,}")
rows = []
allsrc = defaultdict(int); uniq = defaultdict(int)
for e, srcs in edge_src.items():
    for s in srcs:
        allsrc[s] += 1
        if len(srcs) == 1:
            uniq[s] += 1
for s in sorted(allsrc, key=lambda x: -allsrc[x])[:12]:
    rows.append({"source": s, "edges_touched": allsrc[s], "edges_UNIQUE_to_source": uniq[s],
                 "pct_of_all_edges": allsrc[s] / len(edge_src)})
df = pd.DataFrame(rows)
print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print("\nLeave-source-out feasibility: 'edges_UNIQUE_to_source' is the size of the held-out")
print("test set if that source is dropped from training (edges seen ONLY via that source).")
df.to_csv(cfg["edges"].parent / "source_holdout_feasibility.csv", index=False)
