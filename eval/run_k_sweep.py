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
from antheia.metrics import bootstrap_mean
from antheia.pairs import load_split
from antheia.store import FeatureStore
from antheia.taxonomy import TaxonomyAffinity

# Which cutoffs to report? If the model ordering is stable across k, the choice is presentational.
# If it flips (e.g. popularity baselines catching up at large k), that must be reported.
KS = [1, 5, 10, 20, 50, 100]
GEO = ["vf", "vp", "n", "delta", "delta_local", "frs", "prs"]
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

KMAX = max(KS)
disc_full = 1.0 / np.log2(np.arange(2, KMAX + 2))
names = list(scorers) + ["degree null"]
acc = {m: {f"{met}@{k}": [] for met in ("recall", "nrecall", "ndcg", "map") for k in KS} for m in names}
degs = []
for i, (sp, g) in enumerate(test_pos.groupby("plant")):
    p = store.p2i[sp]
    rel = {store.q2i[q] for q in g["pollinator"]}
    R = len(rel); degs.append(R)
    blk = {c: (aff.plant(p) if c == "tax" else store.assemble_plant(p, (c,))) for c in comps}
    svec = {"degree null": store.Prs.astype(float)}
    for n, (spec, fn) in scorers.items():
        svec[n] = fn(np.hstack([blk[c] for c in spec]))
    for m, s in svec.items():
        top = np.argpartition(-s, KMAX)[:KMAX]
        top = top[np.argsort(-s[top])]
        hit = np.fromiter((1.0 if t in rel else 0.0 for t in top), float, KMAX)
        csum = np.cumsum(hit)
        prec = csum / np.arange(1, KMAX + 1)
        for k in KS:
            h = csum[k - 1]
            acc[m][f"recall@{k}"].append(h / R)
            acc[m][f"nrecall@{k}"].append(h / min(R, k))
            idcg = disc_full[:min(R, k)].sum()
            acc[m][f"ndcg@{k}"].append(float((hit[:k] * disc_full[:k]).sum() / idcg) if idcg > 0 else 0.0)
            acc[m][f"map@{k}"].append(float((prec[:k] * hit[:k]).sum() / min(R, k)))
    if (i + 1) % 150 == 0:
        print(f"  {i+1}/{test_pos['plant'].nunique()}", flush=True)

print(f"\ntest plants {len(degs)} | median degree {np.median(degs):.0f} | "
      f"share degree>k: " + ", ".join(f"k={k}:{np.mean(np.array(degs)>k):.0%}" for k in KS) + "\n")
rows = []
for met in ("recall", "nrecall", "ndcg", "map"):
    print(f"--- {met} ---")
    print(f"{'model':<16}" + "".join(f"{'@'+str(k):>10}" for k in KS))
    for m in names:
        line = f"{m:<16}"
        for k in KS:
            v = float(np.mean(acc[m][f"{met}@{k}"]))
            rows.append({"metric": met, "k": k, "model": m, "value": v})
            line += f"{v:>10.4f}"
        print(line)
    print()
df = pd.DataFrame(rows)
df.to_csv(cfg["edges"].parent / "k_sweep.csv", index=False)
print("ordering stability (best-first) per metric x k:")
flips = 0
for met in ("recall", "nrecall", "ndcg", "map"):
    base = None
    for k in KS:
        o = tuple(df[(df.metric == met) & (df.k == k)].sort_values("value", ascending=False)["model"])
        if base is None:
            base = o
        elif o != base:
            flips += 1
            print(f"  FLIP {met}@{k}: {' > '.join(o)}")
    print(f"  {met:<8} reference: {' > '.join(base)}")
print(f"\ntotal ordering flips across all metric x k combinations: {flips}")
