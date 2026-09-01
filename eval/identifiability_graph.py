import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from antheia.config import load_config
from antheia.globi import binomial
from antheia.store import FeatureStore

# Chen et al. (ICML 2024): under the examination hypothesis, relevance is identifiable up to scale
# iff the identifiability graph is connected — nodes are bias factors (here: source datasets),
# with an edge between s,t whenever some item is observed under both.
RAW = "/scratch/ariana.l/CfE2026CVforEcology/rawpollinatordata/interactions.csv.gz"
TYPES = ["visitsFlowersOf", "visits", "pollinates"]
LAT, LON = (24.0, 49.5), (-125.0, -66.0)

cfg = load_config()
store = FeatureStore(cfg)
plants, polls = set(store.plants), set(store.polls)
pl_bin = {binomial(x): x for x in plants}
po_bin = {binomial(x): x for x in polls}
pl_bin.pop(None, None)
po_bin.pop(None, None)

edge_sources = defaultdict(set)
for chunk in pd.read_csv(RAW, usecols=["sourceTaxonName", "targetTaxonName", "interactionTypeName",
                                       "decimalLatitude", "decimalLongitude", "sourceNamespace"],
                         chunksize=1_000_000, low_memory=False, on_bad_lines="skip"):
    d = chunk[(chunk["decimalLatitude"].between(*LAT)) & (chunk["decimalLongitude"].between(*LON))]
    d = d[d["interactionTypeName"].isin(TYPES)].dropna(subset=["sourceTaxonName", "targetTaxonName"])
    d = d[d["sourceTaxonName"] != d["targetTaxonName"]]
    if not len(d):
        continue
    src, tgt = d["sourceTaxonName"], d["targetTaxonName"]
    s_pl = src.where(src.isin(plants)).fillna(src.map(binomial).map(pl_bin))
    s_po = src.where(src.isin(polls)).fillna(src.map(binomial).map(po_bin))
    t_pl = tgt.where(tgt.isin(plants)).fillna(tgt.map(binomial).map(pl_bin))
    t_po = tgt.where(tgt.isin(polls)).fillna(tgt.map(binomial).map(po_bin))
    asis = s_pl.notna() & t_po.notna()
    swap = s_po.notna() & t_pl.notna() & ~asis
    part = pd.concat([
        pd.DataFrame({"p": s_pl[asis], "q": t_po[asis], "ns": d.loc[asis, "sourceNamespace"]}),
        pd.DataFrame({"p": t_pl[swap], "q": s_po[swap], "ns": d.loc[swap, "sourceNamespace"]}),
    ]).dropna()
    for p, q, ns in part.itertuples(index=False):
        edge_sources[(p, q)].add(ns)

print(f"edges with source metadata: {len(edge_sources):,}")
per_src = defaultdict(int)
for s in edge_sources.values():
    for x in s:
        per_src[x] += 1
srcs = sorted(per_src, key=per_src.get, reverse=True)
print(f"\n{'source dataset':<48}{'edges':>10}")
for s in srcs[:12]:
    print(f"{s:<48}{per_src[s]:>10,}")

n_multi = sum(1 for s in edge_sources.values() if len(s) > 1)
print(f"\nedges documented by >1 source: {n_multi:,} ({n_multi / len(edge_sources):.1%})")

co = defaultdict(int)
for s in edge_sources.values():
    for a, b in combinations(sorted(s), 2):
        co[(a, b)] += 1
print(f"co-documented source PAIRS with >=1 shared edge: {len(co)} of {len(srcs) * (len(srcs) - 1) // 2} possible")

# Connectivity of the identifiability graph over the major sources.
major = [s for s in srcs if per_src[s] >= 100]
idx = {s: i for i, s in enumerate(major)}
parent = list(range(len(major)))


def find(a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a


n_edges_ig = 0
for (a, b), c in co.items():
    if a in idx and b in idx:
        n_edges_ig += 1
        ra, rb = find(idx[a]), find(idx[b])
        if ra != rb:
            parent[ra] = rb
comps = defaultdict(list)
for s, i in idx.items():
    comps[find(i)].append(s)
print(f"\nIdentifiability graph over {len(major)} sources with >=100 edges: "
      f"{n_edges_ig} edges, {len(comps)} connected component(s)")
for c in comps.values():
    print(f"  component ({len(c)}): {', '.join(sorted(c)[:6])}{' ...' if len(c) > 6 else ''}")

print("\nWeakest links (source pairs sharing fewest edges, among majors):")
weak = sorted(((c, a, b) for (a, b), c in co.items() if a in idx and b in idx))[:6]
for c, a, b in weak:
    print(f"  {c:>5} shared edges: {a.split('/')[-1]} <-> {b.split('/')[-1]}")
print("\nConnected => relevance identifiable UP TO SCALE, but only if detection depends on the")
print("source alone (rank-1). Pair-dependent detection breaks that regardless of connectivity.")
