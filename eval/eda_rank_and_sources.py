import sys
from collections import Counter, defaultdict
from pathlib import Path
import pandas as pd

RAW = "/scratch/ariana.l/CfE2026CVforEcology/rawpollinatordata/interactions.csv.gz"
LAT, LON = (24.0, 49.5), (-125.0, -66.0)
TYPES = ["visitsFlowersOf", "pollinates", "visits", "interactsWith"]   # Tier A+B
PLANT_K = {"Plantae", "Archaeplastida", "Viridiplantae"}
ANIMAL_K = {"Animalia", "Metazoa"}
COLS = ["interactionTypeName", "decimalLatitude", "decimalLongitude", "sourceNamespace",
        "sourceTaxonName", "targetTaxonName", "sourceTaxonRank", "targetTaxonRank",
        "sourceTaxonSpeciesName", "targetTaxonSpeciesName",
        "sourceTaxonGenusName", "targetTaxonGenusName",
        "sourceTaxonKingdomName", "targetTaxonKingdomName",
        "sourceTaxonOrderName", "sourceTaxonClassName"]

rank = Counter(); ns_rec = Counter()
ns_edges = defaultdict(set); ns_order = defaultdict(Counter)
order_rec = Counter(); edges_sp = set(); edges_any = set()
for chunk in pd.read_csv(RAW, usecols=COLS, chunksize=1_500_000, low_memory=False, on_bad_lines="skip"):
    c = chunk[(chunk["decimalLatitude"].between(*LAT)) & (chunk["decimalLongitude"].between(*LON))]
    f = c[c["interactionTypeName"].isin(TYPES)]
    f = f[f["targetTaxonKingdomName"].isin(PLANT_K) & f["sourceTaxonKingdomName"].isin(ANIMAL_K)]
    if not len(f):
        continue
    pr = f["targetTaxonRank"].fillna("(none)")
    qr = f["sourceTaxonRank"].fillna("(none)")
    for a, b in zip(pr, qr):
        rank[(a, b)] += 1
    ns = f["sourceNamespace"].fillna("(none)")
    for n, k in ns.value_counts().items():
        ns_rec[n] += k
    for o, k in f["sourceTaxonOrderName"].fillna("(none)").value_counts().items():
        order_rec[o] += k
    # edge identity at species rank vs lowest-available rank
    p_sp, q_sp = f["targetTaxonSpeciesName"], f["sourceTaxonSpeciesName"]
    p_lo = p_sp.fillna(f["targetTaxonGenusName"])
    q_lo = q_sp.fillna(f["sourceTaxonGenusName"])
    for p, q, n, o in zip(p_lo, q_lo, ns, f["sourceTaxonOrderName"].fillna("(none)")):
        if pd.notna(p) and pd.notna(q):
            edges_any.add((p, q)); ns_edges[n].add((p, q)); ns_order[n][o] += 1
    for p, q in zip(p_sp, q_sp):
        if pd.notna(p) and pd.notna(q):
            edges_sp.add((p, q))

print(f"=== EDGES (Tier A+B, CONUS, role-valid) ===")
print(f"species-rank both sides : {len(edges_sp):,}")
print(f"lowest-available rank   : {len(edges_any):,}   (+{len(edges_any)-len(edges_sp):,} = "
      f"{(len(edges_any)-len(edges_sp))/max(len(edges_any),1):.1%} gained by allowing genus)")
print("\n=== RANK PAIRS (plant|pollinator), top 8 ===")
tot = sum(rank.values())
for (a, b), n in rank.most_common(8):
    print(f"  {n:>10,}  {n/tot:6.1%}  plant={a:<12} pollinator={b}")
print("\n=== SOURCE NAMESPACES: records, unique edges, dominant order ===")
print(f"{'namespace':<48}{'records':>11}{'edges':>10}  dominant pollinator orders")
for n, k in ns_rec.most_common(10):
    tops = ", ".join(f"{o} {c*100//max(sum(ns_order[n].values()),1)}%" for o, c in ns_order[n].most_common(3))
    print(f"  {n[:46]:<46}{k:>11,}{len(ns_edges[n]):>10,}  {tops}")
print("\n=== POLLINATOR ORDER composition overall ===")
t = sum(order_rec.values())
for o, n in order_rec.most_common(8):
    print(f"  {n:>10,}  {n/t:6.1%}  {o}")
