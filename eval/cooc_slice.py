import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "/scratch/cher/Summer-2026-CfE-Computer-Vision-for-Ecology-Project/src")
from antheia import models
from antheia.config import load_config
from antheia.pairs import load_split
from antheia.store import FeatureStore

cfg = load_config()
store = FeatureStore(cfg)
edges = pd.read_parquet(cfg["edges"])
known_pos = set(zip(edges["plant"], edges["pollinator"]))
seed = cfg["eval"]["seed"]
split = load_split(cfg["edges"].parent / "split_v1.json")
train_pos = edges[edges["plant"].isin(split["train"])]
test_pos = edges[edges["plant"].isin(split["test"])]

pi, qp, qn = models.rank_contrast_sets(train_pos, store, 10, seed, known_pos)
pipes = {name: models.train_pairwise(store, spec, pi, qp, qn, seed)
         for name, spec in models.RANK_SPECS.items() if name != "rank_n"}

partners = {sp: set(store.idx_polls(g["pollinator"])) for sp, g in test_pos.groupby("plant")}
rng = np.random.default_rng(0)
rows = []
frac_partners_cooc = []
for sp, part in partners.items():
    p = store.p2i[sp]
    cand = np.flatnonzero(store.N_full[p] > 0)
    cand_set = set(cand.tolist())
    part_c = part & cand_set
    frac_partners_cooc.append(len(part_c) / len(part))
    if not part_c:
        continue
    jitter = rng.uniform(0, 1e-6, size=len(cand))
    scores = {"rank_n": store.N_full[p][cand].astype(float) + jitter}
    for name, pipe in pipes.items():
        spec = models.RANK_SPECS[name]
        X = store.assemble_plant(p, spec)[cand]
        scores[name] = pipe.decision_function(X) + jitter
    row = {"plant": sp, "n_cand": len(cand), "n_part": len(part_c)}
    for name, s in scores.items():
        top10 = cand[np.argsort(-s)[:10]]
        row[name] = len(part_c & set(top10.tolist())) / len(part_c)
    rows.append(row)

df = pd.DataFrame(rows)
print(f"test plants with >=1 co-occurring partner: {len(df)} / {len(partners)}")
print(f"fraction of true partners that co-occur (mean): {np.mean(frac_partners_cooc):.3f}")
print(f"median candidate-set size (N>0): {df['n_cand'].median():.0f}")
print("\nrecall@10 WITHIN co-occurring candidates (mean over plants):")
for name in ["rank_n"] + list(pipes):
    print(f"  {name:<14} {df[name].mean():.4f}")
