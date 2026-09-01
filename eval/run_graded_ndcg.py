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
from antheia.pairs import load_split
from antheia.store import FeatureStore
from antheia.taxonomy import TaxonomyAffinity

# nDCG's main advantage over MAP is GRADED relevance. We have a natural grade (n_records per edge).
# BUT n_records is documentation intensity, which we showed is process-contaminated — so grading by
# it may quietly reintroduce the popularity bias. Test three gain definitions and see whether the
# model ordering (and especially the popularity null's position) moves.
GEO = ["vf", "vp", "n", "delta", "delta_local", "frs", "prs"]
cfg = load_config()
store = FeatureStore(cfg)
edges = pd.read_parquet(cfg["edges"])
prov = pd.read_csv(cfg["edges"].parent / "edge_provenance_v1.csv")
edges = edges.merge(prov, on=["plant", "pollinator"], how="left")
edges["tier1"] = edges["n_inat"].notna() & (edges["n_inat"] < edges["n"])
split = load_split(cfg["edges"].parent / "split_v1.json")
train_pos, test_pos = edges[edges["plant"].isin(split["train"])], edges[edges["plant"].isin(split["test"])]
aff = TaxonomyAffinity(store, train_pos, cfg["paths"]["globi"])
known = set(zip(edges["plant"], edges["pollinator"]))
seed = cfg["eval"]["seed"]
bp = lambda a, b, c: aff.pairs(a, b) if c == "tax" else store.assemble(a, b, (c,))

pi, qp, qn = models.rank_contrast_sets(train_pos, store, 10, seed, known)
comps = sorted(set(GEO) | {"tax", "n", "delta_local"})
Xp = {c: bp(pi, qp, c) for c in comps}
Xn = {c: bp(pi, qn, c) for c in comps}
scorers = {}
for name, spec in {"N only": ["n"], "N+tax+localD": ["n", "tax", "delta_local"]}.items():
    d = np.hstack([Xp[c] for c in spec]) - np.hstack([Xn[c] for c in spec])
    pipe = Pipeline([("s", StandardScaler(with_mean=False)),
                     ("lr", LogisticRegression(max_iter=1000, random_state=seed, fit_intercept=False))])
    pipe.fit(np.vstack([d, -d]), np.concatenate([np.ones(len(d)), np.zeros(len(d))]))
    scorers[name] = (spec, lambda M, pp=pipe: pp.decision_function(M))
pr = np.arange(0, len(pi), 10)
gbm = HistGradientBoostingClassifier(random_state=seed, max_iter=800, learning_rate=0.05,
                                     min_samples_leaf=100, l2_regularization=5.0)
gbm.fit(np.vstack([np.hstack([Xp[c][pr] for c in GEO]), np.hstack([Xn[c] for c in GEO])]),
        np.concatenate([np.ones(len(pr)), np.zeros(len(pi))]))
scorers["GBM geo"] = (GEO, lambda M: gbm.predict_proba(M)[:, 1])
print("models ready", flush=True)

K = 10
disc = 1.0 / np.log2(np.arange(2, K + 2))
GAINS = {"binary": lambda g: np.ones(len(g)),
         "log n_records": lambda g: np.log1p(g["n_records"].to_numpy(float)),
         "curated-weighted": lambda g: 1.0 + 2.0 * g["tier1"].fillna(False).to_numpy(float)}
acc = {gname: {m: [] for m in list(scorers) + ["degree null"]} for gname in GAINS}

for i, (sp, g) in enumerate(test_pos.groupby("plant")):
    p = store.p2i[sp]
    qidx = np.array([store.q2i[q] for q in g["pollinator"]])
    blk = {c: (aff.plant(p) if c == "tax" else store.assemble_plant(p, (c,))) for c in comps}
    svecs = {"degree null": store.Prs.astype(float)}
    for n, (spec, fn) in scorers.items():
        svecs[n] = fn(np.hstack([blk[c] for c in spec]))
    for gname, gfun in GAINS.items():
        gain = dict(zip(qidx, gfun(g)))
        ideal = np.sort(np.fromiter(gain.values(), float))[::-1][:K]
        idcg = (ideal * disc[:len(ideal)]).sum()
        if idcg <= 0:
            continue
        for m, s in svecs.items():
            top = np.argsort(-s)[:K]
            dcg = sum(gain.get(t, 0.0) * disc[j] for j, t in enumerate(top))
            acc[gname][m].append(dcg / idcg)
    if (i + 1) % 150 == 0:
        print(f"  {i+1}/{test_pos['plant'].nunique()}", flush=True)

print(f"\nnDCG@10 under three gain definitions\n")
print(f"{'model':<16}" + "".join(f"{g:>18}" for g in GAINS))
rows = []
for m in list(scorers) + ["degree null"]:
    line = f"{m:<16}"
    r = {"model": m}
    for gname in GAINS:
        v = float(np.mean(acc[gname][m]))
        r[gname] = v
        line += f"{v:>18.4f}"
    rows.append(r); print(line)
df = pd.DataFrame(rows)
df.to_csv(cfg["edges"].parent / "graded_ndcg.csv", index=False)
print("\nordering by gain definition:")
for gname in GAINS:
    print(f"  {gname:<18} " + " > ".join(df.sort_values(gname, ascending=False)["model"]))
print("\nIf 'log n_records' promotes the degree null, grading by documentation intensity")
print("reintroduces the popularity bias and binary/curated gains are safer.")
