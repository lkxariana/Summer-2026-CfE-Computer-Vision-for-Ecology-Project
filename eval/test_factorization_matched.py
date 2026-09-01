import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from antheia.config import load_config

# Falsification of exp 26: is the R^2 contrast (iNat .187 vs curated .398) real structure, or an
# artifact of the two strata having different n_records variance? R^2 = 1 - resid/total, so a
# stratum whose counts are more homogeneous gets a low R^2 even under identical generative structure.
# Test A: match the two strata on the n_records distribution, then recompute.
# Test B: report residual SD (scale-free-ish) alongside R^2.
# Test C: permutation control — shuffle provenance labels, recompute the gap.
cfg = load_config()
edges = pd.read_parquet(cfg["edges"])
prov = pd.read_csv(cfg["edges"].parent / "edge_provenance_v1.csv")
e = edges.merge(prov, on=["plant", "pollinator"], how="inner")
e["curated"] = (e["n_inat"] < e["n"]).astype(int)


def fit(d, seed=0):
    y = np.log1p(d["n"].to_numpy(float))
    pl, po = pd.Categorical(d["plant"]), pd.Categorical(d["pollinator"])
    n = len(d)
    X = sparse.hstack([
        sparse.csr_matrix((np.ones(n), (np.arange(n), pl.codes)), shape=(n, len(pl.categories))),
        sparse.csr_matrix((np.ones(n), (np.arange(n), po.codes)), shape=(n, len(po.categories))),
    ]).tocsr()
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n); cut = int(0.8 * n)
    tr, te = idx[:cut], idx[cut:]
    m = Ridge(alpha=1.0, solver="sparse_cg", max_iter=2000).fit(X[tr], y[tr])
    pred = m.predict(X[te])
    ss_res = float(((y[te] - pred) ** 2).sum())
    ss_tot = float(((y[te] - y[tr].mean()) ** 2).sum())
    return 1 - ss_res / ss_tot, np.sqrt(ss_res / len(te)), y.std()


print("=== raw (exp 26) ===")
for lab, d in [("iNat-only", e[e.curated == 0]), ("curated", e[e.curated == 1])]:
    r2, rsd, ysd = fit(d)
    print(f"  {lab:<12} n={len(d):>6,}  R2={r2:.3f}  residSD={rsd:.3f}  y_SD={ysd:.3f}")

print("\n=== Test A: strata matched on n_records distribution ===")
cur = e[e.curated == 1]
inat = e[e.curated == 0]
rng = np.random.default_rng(0)
keep = []
for nval, g in cur.groupby("n"):
    pool = inat[inat["n"] == nval]
    if len(pool) == 0:
        continue
    k = min(len(g), len(pool))
    keep.append(pool.sample(k, random_state=0))
    keep.append(g.sample(k, random_state=0))
m = pd.concat(keep)
mi, mc = m[m.curated == 0], m[m.curated == 1]
print(f"  matched pairs per n-stratum: iNat n={len(mi):,}, curated n={len(mc):,}")
for lab, d in [("iNat-only", mi), ("curated", mc)]:
    r2, rsd, ysd = fit(d)
    print(f"  {lab:<12} R2={r2:.3f}  residSD={rsd:.3f}  y_SD={ysd:.3f}")

print("\n=== Test C: permutation control (provenance labels shuffled) ===")
gaps = []
for s in range(5):
    ep = e.copy()
    ep["curated"] = ep["curated"].sample(frac=1, random_state=s).to_numpy()
    a, _, _ = fit(ep[ep.curated == 0], seed=s)
    b, _, _ = fit(ep[ep.curated == 1], seed=s)
    gaps.append(b - a)
print(f"  shuffled curated-minus-iNat R2 gap: {np.mean(gaps):+.3f} ± {np.std(gaps):.3f}  (observed: +0.211)")
print("\nIf the matched-strata gap collapses toward the permutation baseline, exp 26 was a variance artifact.")
