import sys
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/v2")
e = pd.read_parquet(OUT / "edges.parquet")
pl = pd.read_parquet(OUT / "nodes_plants.parquet")
po = pd.read_parquet(OUT / "nodes_pollinators.parquet")
led = pd.read_csv(OUT / "yield_ledger.csv")

def hdr(s): print(f"\n{'='*72}\n{s}\n{'='*72}")

hdr("YIELD LEDGER")
print(led.to_string(index=False))

hdr("1. NETWORK SIZE")
print(f"edges {len(e):,} | plants {e.plant_id.nunique():,} | pollinators {e.pollinator_id.nunique():,}")
print(f"connectance {len(e)/(e.plant_id.nunique()*e.pollinator_id.nunique()):.4%}")
for t in ("A", "B"):
    s = e[e.tier == t]
    print(f"  tier {t}: {len(s):,} edges | {s.plant_id.nunique():,} plants | {s.pollinator_id.nunique():,} pollinators")

hdr("2. RANK COMPOSITION (the 'lowest available' policy)")
print(pd.crosstab(e.plant_rank, e.pollinator_rank, margins=True).to_string())
sp = e[(e.plant_rank == "species") & (e.pollinator_rank == "species")]
print(f"\nspecies-only subgraph: {len(sp):,} edges ({len(sp)/len(e):.1%} of total)")
print(f"edges gained by allowing genus: {len(e)-len(sp):,}")

hdr("3. DEGREE DISTRIBUTIONS")
for nm, d in (("plant", e.groupby('plant_id').size()), ("pollinator", e.groupby('pollinator_id').size())):
    print(f"{nm:<11} median {d.median():>4.0f}  mean {d.mean():>6.1f}  max {d.max():>5}  "
          f"deg<=3 {100*(d<=3).mean():>4.0f}%  deg>20 {100*(d>20).mean():>4.0f}%")

hdr("4. EVIDENTIAL SUPPORT")
print(f"n_records:      median {e.n_records.median():.0f}  >=2 {100*(e.n_records>=2).mean():.0f}%  max {e.n_records.max():,}")
print(f"n_observations: median {e.n_observations.median():.0f}  >=2 {100*(e.n_observations>=2).mean():.0f}%")
print(f"n_sources:      1 source {100*(e.n_sources==1).mean():.0f}%  >=2 {100*(e.n_sources>=2).mean():.0f}%")
print(f"edges with any non-iNat record: {100*(e.n_inat < e.n_records).mean():.0f}%")

hdr("5. SOURCE HOLDOUT FEASIBILITY (edges unique to each source)")
rows = []
for s in ["web-of-life", "gbif-us-bees", "usgs-pollinator-library", "CropPol", "guzman2022", "inaturalist"]:
    has = e.sources.fillna("").str.contains(s, regex=False)
    only = has & (e.n_sources == 1)
    rows.append({"source": s, "edges_touched": int(has.sum()), "edges_unique": int(only.sum())})
print(pd.DataFrame(rows).to_string(index=False))

hdr("6. POLLINATOR TAXONOMY — what is in the network")
print("by order (edges):")
print((100*e.pollinator_order.value_counts(normalize=True).head(10)).round(1).to_string())
print("\nTIER A only:")
print((100*e[e.tier=='A'].pollinator_order.value_counts(normalize=True).head(8)).round(1).to_string())

hdr("7. TEMPORAL COVERAGE")
print(f"first_year: min {e.first_year.min():.0f} median {e.first_year.median():.0f} max {e.first_year.max():.0f}")
print(f"edges with a usable year: {100*e.first_year.notna().mean():.0f}%")
for y in (2015, 2018, 2020, 2022):
    print(f"  first documented <= {y}: {int((e.first_year<=y).sum()):>7,}   after: {int((e.first_year>y).sum()):>7,}")

hdr("8. ROLE ASSIGNMENT")
print(e.role_source.value_counts().to_string())

hdr("9. TOP TAXA (sanity)")
print("top plants:", ", ".join(e.plant.value_counts().head(5).index))
print("top pollinators:", ", ".join(e.pollinator.value_counts().head(5).index))
print(f"\nplants that are also pollinator nodes (should be 0): "
      f"{len(set(e.plant_id) & set(e.pollinator_id))}")
