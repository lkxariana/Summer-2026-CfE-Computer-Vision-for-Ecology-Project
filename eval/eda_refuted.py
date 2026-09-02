import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from antheia.config import load_config
from antheia.globi import binomial
from antheia.store import FeatureStore

REF = "artifacts/external/refuted-interp.tsv.gz"
cfg = load_config()
store = FeatureStore(cfg)
plants, polls = set(store.plants), set(store.polls)
pl_b = {binomial(x): x for x in plants}; pl_b.pop(None, None)
po_b = {binomial(x): x for x in polls}; po_b.pop(None, None)

r = pd.read_csv(REF, sep="\t", low_memory=False,
                usecols=lambda c: c in {"sourceTaxonName", "targetTaxonName", "interactionTypeName",
                                        "argumentTypeId", "referenceCitation", "sourceCitation"})
print(f"refuted rows: {len(r):,}")
print("\ninteraction types:")
print(r["interactionTypeName"].value_counts().head(6).to_string())
print("\nargumentTypeId:")
print(r["argumentTypeId"].value_counts().head(3).to_string())
print("\nreferenceCitation (top 3 reasons):")
for c, n in r["referenceCitation"].value_counts().head(3).items():
    print(f"  {n:>7,}  {str(c)[:100]}")

f = r[r["interactionTypeName"].isin(["visitsFlowersOf", "visits", "pollinates"])].dropna(
    subset=["sourceTaxonName", "targetTaxonName"])
print(f"\nflower-visitation refutations: {len(f):,} rows, {len(f[['sourceTaxonName','targetTaxonName']].drop_duplicates()):,} distinct name pairs")

# Map to our species universe in BOTH orientations, since refuted rows are error records.
s, t = f["sourceTaxonName"], f["targetTaxonName"]
s_pl = s.where(s.isin(plants)).fillna(s.map(binomial).map(pl_b))
s_po = s.where(s.isin(polls)).fillna(s.map(binomial).map(po_b))
t_pl = t.where(t.isin(plants)).fillna(t.map(binomial).map(pl_b))
t_po = t.where(t.isin(polls)).fillna(t.map(binomial).map(po_b))
asis = s_pl.notna() & t_po.notna()
swap = s_po.notna() & t_pl.notna() & ~asis
neg = pd.concat([
    pd.DataFrame({"plant": s_pl[asis], "pollinator": t_po[asis]}),
    pd.DataFrame({"plant": t_pl[swap], "pollinator": s_po[swap]}),
]).drop_duplicates()
print(f"refuted pairs mapping into OUR universe: {len(neg):,}  (as-is {int(asis.sum()):,}, swapped {int(swap.sum()):,})")

edges = pd.read_parquet(cfg["edges"])
pos = set(zip(edges["plant"], edges["pollinator"]))
negset = set(zip(neg["plant"], neg["pollinator"]))
overlap = pos & negset
print(f"\n*** CONTAMINATION CHECK ***")
print(f"our positive edges: {len(pos):,}")
print(f"refuted pairs in universe: {len(negset):,}")
print(f"pairs BOTH asserted and refuted: {len(overlap):,} ({len(overlap)/len(pos):.3%} of our edges)")
print(f"refuted pairs usable as TRUE NEGATIVES (not in our positives): {len(negset - pos):,}")
if overlap:
    print("examples of contested pairs:", list(overlap)[:5])
pd.DataFrame(sorted(negset - pos), columns=["plant", "pollinator"]).to_csv(
    cfg["edges"].parent / "refuted_negatives_interp_v1.csv", index=False)
print(f"\nsaved -> refuted_negatives_interp_v1.csv")
