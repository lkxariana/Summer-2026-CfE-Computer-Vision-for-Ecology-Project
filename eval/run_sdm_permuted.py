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
import eval.run_sdm_prauc as B

# Plan step 5, now warranted: exp 39 found SDM > GBIF on pooled PR-AUC (+0.0063, p=.0005).
# Shuffle the species -> SDM-surface assignment. If permuted SDM reproduces the gain over base,
# the "gain" is feature capacity / scale, not the right species' phenology, and exp 39 dies.
cfg, store, sdm, seed = B.cfg, B.store, B.sdm, B.seed
tr, te, Btr, Bte = B.tr, B.te, B.Btr, B.Bte
aff = B.aff

rng = np.random.default_rng(123)
perm = rng.permutation(len(sdm.species))
A_perm = sdm.A_sdm[perm]


def sdm_block_permuted(df):
    p_idx = store.idx_plants(df["plant"])
    q_rows = np.array([sdm.sp2row[s] for s in df["pollinator"]])
    ds = np.minimum(store.FC[p_idx], A_perm[q_rows]).sum(1)[:, None]
    db = np.zeros((len(df), 1))
    for p in np.unique(p_idx):
        m = p_idx == p
        db[m, 0] = sdm.delta_bilateral(p, perm[q_rows[m]])   # wrong species' surface, same geometry
    return np.hstack([ds, db])


print("building permuted SDM features...", flush=True)
Ptr, Pte = sdm_block_permuted(tr), sdm_block_permuted(te)

arms = {
    "base": (Btr["base"], Bte["base"]),
    "base+SDM (real)": (np.hstack([Btr["base"], Btr["sdm"]]), np.hstack([Bte["base"], Bte["sdm"]])),
    "base+SDM (permuted)": (np.hstack([Btr["base"], Ptr]), np.hstack([Bte["base"], Pte])),
}
scores = {}
for name, (Xtr, Xte) in arms.items():
    m = Pipeline([("s", StandardScaler()), ("lr", LogisticRegression(max_iter=2000, random_state=seed))])
    m.fit(Xtr, tr["label"])
    scores[name] = m.decision_function(Xte)
    print(f"  trained {name}", flush=True)

y = te["label"].to_numpy(); plants = te["plant"].to_numpy()
groups = {p: np.flatnonzero(plants == p) for p in np.unique(plants)}
names = list(groups)
rng2 = np.random.default_rng(seed)
diffs = {"real-base": [], "permuted-base": [], "real-permuted": []}
for _ in range(2000):
    pick = np.concatenate([groups[names[i]] for i in rng2.integers(0, len(names), len(names))])
    yy = y[pick]
    if yy.min() == yy.max():
        continue
    ap = {k: average_precision_score(yy, v[pick]) for k, v in scores.items()}
    diffs["real-base"].append(ap["base+SDM (real)"] - ap["base"])
    diffs["permuted-base"].append(ap["base+SDM (permuted)"] - ap["base"])
    diffs["real-permuted"].append(ap["base+SDM (real)"] - ap["base+SDM (permuted)"])

print(f"\n{'contrast':<18}{'mean Δ PR-AUC':>16}{'95% CI':>24}{'p':>10}")
out = []
for k, v in diffs.items():
    v = np.array(v)
    p = 2 * min((v <= 0).mean(), (v >= 0).mean())
    out.append({"contrast": k, "mean": v.mean(), "lo": np.percentile(v, 2.5), "hi": np.percentile(v, 97.5)})
    print(f"{k:<18}{v.mean():>+16.4f}   [{np.percentile(v,2.5):+.4f},{np.percentile(v,97.5):+.4f}]{max(p,5e-4):>10.4f}")
pd.DataFrame(out).to_csv(cfg["edges"].parent / "sdm_permuted_v1.csv", index=False)
print("\nIf permuted-base ~= real-base, exp 39's gain is capacity, not phenology.")
