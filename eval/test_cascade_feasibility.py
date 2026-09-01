import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from antheia import models
from antheia.config import load_config
from antheia.metrics import bootstrap_mean
from antheia.pairs import load_split
from antheia.store import FeatureStore
from antheia.taxonomy import TaxonomyAffinity

# Exp 41 showed phenology improves broad ORDERING (AUC 0.757 among co-occurring) but not the
# top of the list. If true, Delta should raise recall at LARGE k even where it fails at k=10.
# That is the feasibility condition for a CASCADE: retrieve a candidate set with N+Delta
# (where phenology pays), then re-rank with taxonomy (where affinity pays).
cfg = load_config()
store = FeatureStore(cfg)
edges = pd.read_parquet(cfg["edges"])
split = load_split(cfg["edges"].parent / "split_v1.json")
train_pos, test_pos = edges[edges["plant"].isin(split["train"])], edges[edges["plant"].isin(split["test"])]
aff = TaxonomyAffinity(store, train_pos, cfg["paths"]["globi"])
known = set(zip(edges["plant"], edges["pollinator"]))
seed, nboot = cfg["eval"]["seed"], cfg["eval"]["bootstrap_n"]
KS = [10, 50, 100, 500, 1000]

pi, qp, qn = models.rank_contrast_sets(train_pos, store, 10, seed, known)
def blk_pair(c, p, q):
    return aff.pairs(p, q) if c == "tax" else store.assemble(p, q, (c,))
def blk_plant(c, p):
    return aff.plant(p) if c == "tax" else store.assemble_plant(p, (c,))

SPECS = {"N": ["n"], "N+Δ": ["n", "delta"], "N+localΔ": ["n", "delta_local"],
         "N+tax": ["n", "tax"], "N+tax+localΔ": ["n", "tax", "delta_local"]}
comps = sorted({c for v in SPECS.values() for c in v})
Xp = {c: blk_pair(c, pi, qp) for c in comps}
Xn = {c: blk_pair(c, pi, qn) for c in comps}
pipes = {}
for name, spec in SPECS.items():
    d = np.hstack([Xp[c] for c in spec]) - np.hstack([Xn[c] for c in spec])
    pl = Pipeline([("s", StandardScaler(with_mean=False)),
                   ("lr", LogisticRegression(max_iter=1000, random_state=seed, fit_intercept=False))])
    pl.fit(np.vstack([d, -d]), np.concatenate([np.ones(len(d)), np.zeros(len(d))]))
    pipes[name] = pl

partners = {sp: set(store.idx_polls(g["pollinator"])) for sp, g in test_pos.groupby("plant")}
rec = {n: {k: [] for k in KS} for n in SPECS}
for i, (sp, part) in enumerate(partners.items()):
    p = store.p2i[sp]
    b = {c: blk_plant(c, p) for c in comps}
    for name, spec in SPECS.items():
        s = pipes[name].decision_function(np.hstack([b[c] for c in spec]))
        order = np.argsort(-s)
        for k in KS:
            rec[name][k].append(len(part & set(order[:k].tolist())) / len(part))
    if (i + 1) % 150 == 0:
        print(f"  {i+1}/{len(partners)}", flush=True)

print(f"\n{'model':<16}" + "".join(f"{'R@'+str(k):>12}" for k in KS))
tbl = {}
for name in SPECS:
    tbl[name] = {k: np.mean(rec[name][k]) for k in KS}
    print(f"{name:<16}" + "".join(f"{tbl[name][k]:>12.4f}" for k in KS))

print("\nΔ effect at each k (N+localΔ minus N):")
for k in KS:
    print(f"  R@{k:<5} {tbl['N+localΔ'][k] - tbl['N'][k]:+.4f}")
print("\nCascade is viable iff Δ helps at LARGE k (better candidate set) even while hurting at k=10.")
pd.DataFrame(tbl).T.to_csv(cfg["edges"].parent / "cascade_feasibility.csv")
