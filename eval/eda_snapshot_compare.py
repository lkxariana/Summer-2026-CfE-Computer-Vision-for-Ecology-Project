import hashlib
import sys
from pathlib import Path
import pandas as pd

FILES = {
    "june_local": "/scratch/ariana.l/CfE2026CVforEcology/rawpollinatordata/interactions.csv.gz",
    "aug26_pinned": "artifacts/external/globi_2026-08-26.csv.gz",
}
LAT, LON = (24.0, 49.5), (-125.0, -66.0)
TIER_A = ["visitsFlowersOf", "pollinates"]
TIER_B = ["visits", "interactsWith"]
COLS = ["interactionTypeName", "decimalLatitude", "decimalLongitude", "sourceNamespace"]


def sha256(path, limit=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        n = 0
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
            n += len(b)
            if limit and n >= limit:
                break
    return h.hexdigest()


out = {}
for name, path in FILES.items():
    if not Path(path).exists():
        print(f"{name}: MISSING")
        continue
    rows = conus = 0
    types = {}
    ns = {}
    for chunk in pd.read_csv(path, usecols=COLS, chunksize=2_000_000, low_memory=False,
                             on_bad_lines="skip"):
        rows += len(chunk)
        c = chunk[(chunk["decimalLatitude"].between(*LAT)) & (chunk["decimalLongitude"].between(*LON))]
        conus += len(c)
        for t, n in c["interactionTypeName"].value_counts().items():
            types[t] = types.get(t, 0) + n
        f = c[c["interactionTypeName"].isin(TIER_A + TIER_B)]
        for s, n in f["sourceNamespace"].fillna("(none)").value_counts().items():
            ns[s] = ns.get(s, 0) + n
    out[name] = {"rows": rows, "conus": conus, "types": types, "ns": ns,
                 "sha256_head256MB": sha256(path, 1 << 28), "bytes": Path(path).stat().st_size}
    print(f"{name}: rows {rows:,} | CONUS {conus:,} | bytes {out[name]['bytes']:,}", flush=True)

if len(out) == 2:
    a, b = out["june_local"], out["aug26_pinned"]
    print("\n=== CONUS records by interaction type ===")
    print(f"{'type':<34}{'june':>12}{'aug26':>12}{'delta':>12}{'pct':>9}")
    for t in sorted(set(a["types"]) | set(b["types"]), key=lambda x: -(a["types"].get(x, 0))):
        x, y = a["types"].get(t, 0), b["types"].get(t, 0)
        if max(x, y) < 5000:
            continue
        tag = "  <-- TIER A" if t in TIER_A else ("  <-- tier B" if t in TIER_B else "")
        print(f"{t:<34}{x:>12,}{y:>12,}{y-x:>+12,}{(y-x)/max(x,1):>+8.1%}{tag}")
    ta_a = sum(a["types"].get(t, 0) for t in TIER_A)
    ta_b = sum(b["types"].get(t, 0) for t in TIER_A)
    print(f"\nTIER A total: june {ta_a:,} -> aug26 {ta_b:,}  ({(ta_b-ta_a)/max(ta_a,1):+.1%})")
    print("\n=== Tier A+B CONUS records by source namespace (top 10 by june) ===")
    print(f"{'namespace':<52}{'june':>11}{'aug26':>11}{'delta':>11}")
    for s in sorted(a["ns"], key=lambda x: -a["ns"][x])[:10]:
        x, y = a["ns"][s], b["ns"].get(s, 0)
        print(f"{s[:50]:<52}{x:>11,}{y:>11,}{y-x:>+11,}")
    gone = [s for s in a["ns"] if s not in b["ns"] and a["ns"][s] > 100]
    new = [s for s in b["ns"] if s not in a["ns"] and b["ns"][s] > 100]
    print(f"\nnamespaces REMOVED since june (>100 recs): {gone if gone else 'none'}")
    print(f"namespaces ADDED since june (>100 recs):   {new if new else 'none'}")
    print(f"\nsha256(first 256MB) june : {a['sha256_head256MB']}")
    print(f"sha256(first 256MB) aug26: {b['sha256_head256MB']}")
