import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from antheia import models
from antheia.config import load_config
from antheia.metrics import bootstrap_mean, ranking_metrics
from antheia.pairs import load_split
from antheia.store import FeatureStore
from antheia.taxonomy import TaxonomyAffinity

# Does the CHOICE of ranking metric change which model wins? If recall@10, nDCG@10, MAP and MRR
# all order the models identically, the metric debate is moot for our conclusions.
GEO = ["vf", "vp", "n", "delta", "delta_local", "frs", "prs"]
LINEAR = {"N only": ["n"], "N+tax+localD": ["n", "tax", "delta_local"], "degree null": None}
cfg = load_config()
store = FeatureStore(cfg)
edges = pd.read_parquet(cfg["edges"])
split = load_split(cfg["edges"].parent / "split_v1.json")
train_pos, test_pos = edges[edges["plant"].isin(split["train"])], edges[edges["plant"].isin(split["test"])]
aff = TaxonomyAffinity(store, train_pos, cfg["paths"]["globi"])
known = set(zip(edges["plant"], edges["pollinator"]))
seed, nboot = cfg["eval"]["seed"], cfg["eval"]["bootstrap_n"]
bp = lambda a, b, c: aff.pairs(a, b) if c == "tax" else store.assemble(a, b, (c,))

pi, qp, qn = models.rank_contrast_sets(train_pos, store, 10, seed, known)
comps = sorted({c for v in LINEAR.values() if v for c in v} | set(GEO))
Xp = {c: bp(pi, qp, c) for c in comps}
Xn = {c: bp(pi, qn, c) for c in comps}
scorers = {}
for name, spec in LINEAR.items():
    if spec is None:
        continue
    d = np.hstack([Xp[c] for c in spec]) - np.hstack([Xn[c] for c in spec])
    pipe = Pipeline([("s", StandardScaler(with_mean=False)),
                     ("lr", LogisticRegression(max_iter=1000, random_state=seed, fit_intercept=False))])
    pipe.fit(np.vstack([d, -d]), np.concatenate([np.ones(len(d)), np.zeros(len(d))]))
    scorers[name] = (spec, lambda M, pp=pipe: pp.decision_function(M))
pos_rows = np.arange(0, len(pi), 10)
gbm = HistGradientBoostingClassifier(random_state=seed, max_iter=800, learning_rate=0.05,
                                     min_samples_leaf=100, l2_regularization=5.0)
gbm.fit(np.vstack([np.hstack([Xp[c][pos_rows] for c in GEO]), np.hstack([Xn[c] for c in GEO])]),
        np.concatenate([np.ones(len(pos_rows)), np.zeros(len(pi))]))
scorers["GBM geo"] = (GEO, lambda M: gbm.predict_proba(M)[:, 1])
print("models ready", flush=True)

partners = {sp: set(store.idx_polls(g["pollinator"])) for sp, g in test_pos.groupby("plant")}
acc = {n: [] for n in list(scorers) + ["degree null"]}
degs = []
for i, (sp, part) in enumerate(partners.items()):
    p = store.p2i[sp]
    blk = {c: (aff.plant(p) if c == "tax" else store.assemble_plant(p, (c,))) for c in comps}
    degs.append(len(part))
    acc["degree null"].append(ranking_metrics(store.Prs.astype(float), part))
    for n, (spec, fn) in scorers.items():
        acc[n].append(ranking_metrics(fn(np.hstack([blk[c] for c in spec])), part))
    if (i + 1) % 150 == 0:
        print(f"  {i+1}/{len(partners)}", flush=True)

keys = ["recall@10", "nrecall@10", "ndcg@10", "map", "mrr", "hit@10", "median_rank_first"]
print(f"\ntest plants {len(degs)}, median degree {np.median(degs):.0f}, "
      f"share with >10 partners {np.mean(np.array(degs) > 10):.1%}\n")
print(f"{'model':<16}" + "".join(f"{k:>13}" for k in keys))
rows = []
for n in acc:
    r = {"model": n}
    line = f"{n:<16}"
    for k in keys:
        v = np.array([d[k] for d in acc[n]], float)
        v = v[~np.isnan(v)]
        r[k] = np.median(v) if k == "median_rank_first" else v.mean()
        line += f"{r[k]:>13.4f}" if k != "median_rank_first" else f"{r[k]:>13.0f}"
    rows.append(r); print(line)
df = pd.DataFrame(rows)
df.to_csv(cfg["edges"].parent / "metric_comparison.csv", index=False)
print("\nrank order by each metric (best first):")
for k in keys[:-2]:
    print(f"  {k:<12} " + " > ".join(df.sort_values(k, ascending=False)["model"]))
