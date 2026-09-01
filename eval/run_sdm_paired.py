import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import eval.run_sdm_prauc as base  # reuses its data build (module-level)

# Marginal CIs overlapped ~95%, but the arms are highly correlated so a PAIRED cluster bootstrap
# (resampling test plants) is the powerful test of SDM-vs-GBIF on pooled PR-AUC.
te, Bte, Btr, tr = base.te, base.Bte, base.Btr, base.tr
seed = base.seed
scores = {}
for name, keys in {"gbif": ["base", "gbif"], "sdm": ["base", "sdm"], "base": ["base"]}.items():
    Xtr = np.hstack([Btr[k] for k in keys]); Xte = np.hstack([Bte[k] for k in keys])
    m = Pipeline([("s", StandardScaler()), ("lr", LogisticRegression(max_iter=2000, random_state=seed))])
    m.fit(Xtr, tr["label"])
    scores[name] = m.decision_function(Xte)

y = te["label"].to_numpy()
plants = te["plant"].to_numpy()
groups = {p: np.flatnonzero(plants == p) for p in np.unique(plants)}
names = list(groups)
rng = np.random.default_rng(seed)
diffs = {"sdm-gbif": [], "gbif-base": [], "sdm-base": []}
for _ in range(2000):
    pick = np.concatenate([groups[names[i]] for i in rng.integers(0, len(names), len(names))])
    yy = y[pick]
    if yy.min() == yy.max():
        continue
    ap = {k: average_precision_score(yy, v[pick]) for k, v in scores.items()}
    diffs["sdm-gbif"].append(ap["sdm"] - ap["gbif"])
    diffs["gbif-base"].append(ap["gbif"] - ap["base"])
    diffs["sdm-base"].append(ap["sdm"] - ap["base"])

print(f"{'contrast':<14}{'mean Δ':>10}{'95% CI':>22}{'p (two-sided)':>16}")
for k, v in diffs.items():
    v = np.array(v)
    p = 2 * min((v <= 0).mean(), (v >= 0).mean())
    print(f"{k:<14}{v.mean():>+10.4f}   [{np.percentile(v,2.5):+.4f},{np.percentile(v,97.5):+.4f}]"
          f"{max(p,5e-4):>14.4f}")
pd.DataFrame({k: [np.mean(v), np.percentile(v, 2.5), np.percentile(v, 97.5)] for k, v in diffs.items()},
             index=["mean", "lo", "hi"]).T.to_csv(base.cfg["edges"].parent / "sdm_paired_v1.csv")
