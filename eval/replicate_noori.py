import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd

# Robustness check of the identifiability-graph and overlap diagnostics on an independent
# curation of bee-plant interactions (Noori et al. 2026, Sci Data; 186 sources, global, CC BY).
# NOTE: GloBI-derived, so this is a differently-curated slice, not a fully independent corpus.
PATH = Path("artifacts/external/noori_globi_curated.csv")
COLS = ["bee_species", "plant_species", "sourceCitation", "sourceInstitutionCode", "basisOfRecord"]

use = [c for c in COLS if c in pd.read_csv(PATH, nrows=0, encoding="latin-1").columns]
print(f"columns used: {use}")
df = pd.concat([c for c in pd.read_csv(PATH, usecols=use, chunksize=500_000, low_memory=False, encoding="latin-1", on_bad_lines="skip")])
print(f"records: {len(df):,}")
df = df.dropna(subset=["bee_species", "plant_species"])
df["src"] = df["sourceCitation"].fillna("unknown").str.slice(0, 60)
print(f"records with both partners named: {len(df):,} | distinct sources: {df['src'].nunique():,}")

edge_src = df.groupby(["plant_species", "bee_species"])["src"].agg(set)
print(f"unique bee-plant edges: {len(edge_src):,}")
multi = edge_src.map(len) > 1
print(f"\nedges documented by >1 source: {int(multi.sum()):,} ({multi.mean():.1%})"
      "   [ANTHEIA CONUS: 4.8%]")

per_src = defaultdict(int)
for ss in edge_src:
    for x in ss:
        per_src[x] += 1
majors = [s for s, c in per_src.items() if c >= 100]
idx = {s: i for i, s in enumerate(majors)}
co = defaultdict(int)
for ss in edge_src:
    m = sorted(x for x in ss if x in idx)
    for a, b in combinations(m, 2):
        co[(a, b)] += 1
parent = list(range(len(majors)))
def find(a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]; a = parent[a]
    return a
for (a, b) in co:
    ra, rb = find(idx[a]), find(idx[b])
    if ra != rb:
        parent[ra] = rb
comps = defaultdict(list)
for s_, i in idx.items():
    comps[find(i)].append(s_)
print(f"\nidentifiability graph over {len(majors)} sources with >=100 edges:")
print(f"  edges in graph: {len(co)} | connected components: {len(comps)}")
sizes = sorted((len(v) for v in comps.values()), reverse=True)
print(f"  component sizes: {sizes[:8]}")
if co:
    w = sorted(co.values())
    print(f"  bridge strength: min={w[0]}, 10th pct={int(np.percentile(w,10))}, median={int(np.median(w))}"
          f"   [ANTHEIA CONUS: min 4, weakest bridges 4-13]")
    print(f"  bridges with <=13 shared edges: {sum(1 for x in w if x <= 13)}/{len(w)} ({sum(1 for x in w if x<=13)/len(w):.0%})")
print("\nIf overlap is again a few percent and bridges are thin, the vacuous-identifiability")
print("diagnostic is a property of aggregated interaction databases, not of our extraction.")
