import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from antheia.config import load_config

cfg = load_config(); P = cfg["paths"]
e = pd.read_parquet("data/network/edges.parquet")
plant_feat = set(pd.read_csv(P["vf"], usecols=[0]).iloc[:, 0])
poll_feat = set(pd.read_csv(P["vp"], usecols=[0]).iloc[:, 0])

print("=" * 72)
print("Q1. Are the missing pollinators present in the raw occurrence source?")
print("=" * 72)
raw = P["data_root"] / "Plant Pollinator Initial Analysis" / "pollinator_observations_v2.csv" \
      if isinstance(P.get("data_root"), Path) else Path(
      "/scratch/ariana.l/Plant Pollinator Initial Analysis/pollinator_observations_v2.csv")
cnt = {}
for c in pd.read_csv(raw, usecols=["pollinator_species"], chunksize=2_000_000, low_memory=False):
    for s, n in c["pollinator_species"].value_counts().items():
        cnt[s] = cnt.get(s, 0) + int(n)
raw_species = set(cnt)
print(f"raw occurrence file: {len(raw_species):,} species | feature matrix: {len(poll_feat):,}")

miss = e[~e.pollinator.isin(poll_feat)].drop_duplicates("pollinator")
miss_sp = miss[miss.pollinator_rank == "species"]
in_raw = miss_sp.pollinator.isin(raw_species)
print(f"\nmissing species-rank pollinator taxa: {len(miss_sp):,}")
print(f"  present in raw occurrence file : {in_raw.sum():,}  <- recoverable by rebuilding features")
print(f"  absent from raw occurrence file: {(~in_raw).sum():,}  <- would need a new GBIF download")
rec = miss_sp[in_raw]
if len(rec):
    rec = rec.assign(n_occ=rec.pollinator.map(cnt))
    print(f"\n  of the recoverable, occurrence counts: median {rec.n_occ.median():.0f}, "
          f">=10 recs {100*(rec.n_occ>=10).mean():.0f}%, >=50 recs {100*(rec.n_occ>=50).mean():.0f}%")
    print("  by order:")
    print("   " + rec.pollinator_order.value_counts().head(5).to_string().replace("\n", "\n   "))
    gain = e[(~e.pollinator.isin(poll_feat)) & e.pollinator.isin(set(rec.pollinator)) &
             e.plant.isin(plant_feat)]
    print(f"\n  interactions unlocked by rebuilding pollinator features: {len(gain):,}")

print()
print("=" * 72)
print("Q2. Is genus-level feature aggregation feasible?")
print("=" * 72)
gp = e[e.plant_rank == "genus"].drop_duplicates("plant")
gq = e[e.pollinator_rank == "genus"].drop_duplicates("pollinator")
plant_gen = {}
for s in plant_feat:
    plant_gen.setdefault(str(s).split()[0], []).append(s)
poll_gen = {}
for s in poll_feat:
    poll_gen.setdefault(str(s).split()[0], []).append(s)
gp_n = gp.plant.map(lambda g: len(plant_gen.get(g, [])))
gq_n = gq.pollinator.map(lambda g: len(poll_gen.get(g, [])))
print(f"genus-rank plant nodes: {len(gp):,} | with >=1 feature-covered congener: {(gp_n>0).sum():,} ({100*(gp_n>0).mean():.0f}%)")
print(f"  median covered species per genus: {gp_n[gp_n>0].median():.0f}")
print(f"genus-rank pollinator nodes: {len(gq):,} | with >=1 feature-covered congener: {(gq_n>0).sum():,} ({100*(gq_n>0).mean():.0f}%)")
print(f"  median covered species per genus: {gq_n[gq_n>0].median():.0f}")

ok_p = set(gp[gp_n > 0].plant); ok_q = set(gq[gq_n > 0].pollinator)
pc = e.plant.isin(plant_feat) | e.plant.isin(ok_p)
qc = e.pollinator.isin(poll_feat) | e.pollinator.isin(ok_q)
print(f"\ninteractions modellable with genus aggregation: {(pc & qc).sum():,} "
      f"(vs {(e.plant.isin(plant_feat) & e.pollinator.isin(poll_feat)).sum():,} without) "
      f"= +{(pc & qc).sum() - (e.plant.isin(plant_feat) & e.pollinator.isin(poll_feat)).sum():,}")
