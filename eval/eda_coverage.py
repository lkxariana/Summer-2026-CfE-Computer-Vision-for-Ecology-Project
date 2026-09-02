import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from antheia.config import load_config

cfg = load_config()
e = pd.read_parquet("data/network/edges.parquet")
P = cfg["paths"]

# Feature universes as currently available.
plant_feat = set(pd.read_csv(P["vf"], usecols=[0]).iloc[:, 0])
poll_feat = set(pd.read_csv(P["vp"], usecols=[0]).iloc[:, 0])
print(f"feature universes: plants {len(plant_feat):,} | pollinators {len(poll_feat):,}")
print(f"network taxa:      plants {e.plant.nunique():,} | pollinators {e.pollinator.nunique():,}\n")

e["p_cov"] = e.plant.isin(plant_feat)
e["q_cov"] = e.pollinator.isin(poll_feat)
both = e.p_cov & e.q_cov
print(f"interactions with BOTH sides covered: {both.sum():,} / {len(e):,} ({both.mean():.1%})")
print(f"  plant side covered:      {e.p_cov.mean():.1%}")
print(f"  pollinator side covered: {e.q_cov.mean():.1%}\n")

def profile(mask, label):
    s = e[mask]
    print(f"--- {label}: {len(s):,} interactions, {s.plant.nunique():,} plants, {s.pollinator.nunique():,} pollinators")
    print(f"    tier A {100*(s.tier=='A').mean():.0f}%  genus-rank plant {100*(s.plant_rank=='genus').mean():.0f}%  "
          f"genus-rank pollinator {100*(s.pollinator_rank=='genus').mean():.0f}%")
    print(f"    top pollinator orders: " + ", ".join(
        f"{o} {v:.0f}%" for o, v in (100*s.pollinator_order.value_counts(normalize=True).head(4)).items()))

profile(both, "COVERED (modelled subgraph)")
profile(~both, "EXCLUDED at feature intersection")

print("\n=== WHY are pollinators excluded? (uncovered pollinator taxa) ===")
unc = e[~e.q_cov].drop_duplicates("pollinator")
print(f"uncovered pollinator taxa: {len(unc):,}")
print("by rank:"); print(unc.pollinator_rank.value_counts().to_string())
print("\nby order (top 10) — these are the taxa the current pool excludes:")
o = unc.pollinator_order.value_counts().head(10)
tot = e.drop_duplicates("pollinator").pollinator_order.value_counts()
for k, v in o.items():
    print(f"  {k:<18} {v:>6,} uncovered of {tot.get(k,0):>6,} in network ({100*v/max(tot.get(k,1),1):.0f}%)")

print("\n=== edges recoverable by widening the pollinator pool ===")
for order in ["Hymenoptera", "Lepidoptera", "Diptera", "Coleoptera", "Hemiptera", "Passeriformes", "Araneae"]:
    m = (~e.q_cov) & (e.pollinator_order == order) & e.p_cov
    print(f"  {order:<16} +{m.sum():>7,} interactions would be added if this order were fully covered")

print("\n=== WHY are plants excluded? ===")
uncp = e[~e.p_cov].drop_duplicates("plant")
print(f"uncovered plant taxa: {len(uncp):,}")
print("by rank:"); print(uncp.plant_rank.value_counts().to_string())
print("\ntop uncovered plant families:")
print(uncp.plant_family.value_counts().head(8).to_string())
