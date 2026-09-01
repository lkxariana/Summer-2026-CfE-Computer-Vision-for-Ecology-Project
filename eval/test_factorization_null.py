import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import poisson
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from antheia.config import load_config

# Reviewer objection to exp 26/28: low-mean, overdispersed counts yield low R^2 MECHANICALLY even
# when the true model IS multiplicative. Marginal-distribution matching does not fix mean-variance
# scaling. Correct null: simulate each stratum FROM ITS OWN FITTED multiplicative model
# (zero-truncated Poisson, since we only observe documented edges) and recompute R^2.
# If simulated R^2 reproduces the observed gap, the factorization claim dies.
cfg = load_config()
edges = pd.read_parquet(cfg["edges"])
prov = pd.read_csv(cfg["edges"].parent / "edge_provenance_v1.csv")
e = edges.merge(prov, on=["plant", "pollinator"], how="inner")
e["curated"] = (e["n_inat"] < e["n"]).astype(int)


def design(d):
    pl, po = pd.Categorical(d["plant"]), pd.Categorical(d["pollinator"])
    n = len(d)
    return sparse.hstack([
        sparse.csr_matrix((np.ones(n), (np.arange(n), pl.codes)), shape=(n, len(pl.categories))),
        sparse.csr_matrix((np.ones(n), (np.arange(n), po.codes)), shape=(n, len(po.categories))),
    ]).tocsr()


def r2_of(X, y, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y)); cut = int(0.8 * len(y))
    tr, te = idx[:cut], idx[cut:]
    m = Ridge(alpha=1.0, solver="sparse_cg", max_iter=2000).fit(X[tr], y[tr])
    pred = m.predict(X[te])
    return 1 - float(((y[te] - pred) ** 2).sum()) / float(((y[te] - y[tr].mean()) ** 2).sum())


def ztpoisson(mu, rng):
    """Zero-truncated Poisson draws (documented edges necessarily have n>=1)."""
    mu = np.clip(mu, 1e-3, None)
    u = rng.uniform(poisson.cdf(0, mu), 1.0)
    return np.maximum(poisson.ppf(u, mu), 1)


print(f"{'stratum':<12}{'observed R2':>13}{'null R2 (fitted multiplicative)':>34}{'verdict':>26}")
res = {}
for lab, d in [("iNat-only", e[e.curated == 0]), ("curated", e[e.curated == 1])]:
    X = design(d)
    y = np.log1p(d["n"].to_numpy(float))
    obs = r2_of(X, y)
    # fit on ALL rows to get the generative multiplicative mean, then simulate
    m = Ridge(alpha=1.0, solver="sparse_cg", max_iter=2000).fit(X, y)
    mu = np.expm1(np.clip(m.predict(X), 0, None))
    sims = []
    for s in range(5):
        rng = np.random.default_rng(100 + s)
        ysim = np.log1p(ztpoisson(mu, rng))
        sims.append(r2_of(X, ysim, seed=s))
    res[lab] = (obs, np.mean(sims), np.std(sims))
    print(f"{lab:<12}{obs:>13.3f}{np.mean(sims):>26.3f} ± {np.std(sims):.3f}", end="")
    print(f"{'obs << null' if obs < np.mean(sims) - 0.05 else 'obs ~ null':>22}")

obs_gap = res["curated"][0] - res["iNat-only"][0]
null_gap = res["curated"][1] - res["iNat-only"][1]
print(f"\nobserved gap (curated - iNat): {obs_gap:+.3f}")
print(f"null gap under true multiplicative model: {null_gap:+.3f} ± {max(res['curated'][2], res['iNat-only'][2]):.3f}")
print(f"gap NOT explained by count distribution: {obs_gap - null_gap:+.3f}")
if abs(obs_gap - null_gap) < 0.05:
    print("\nVERDICT: the gap is reproduced by the count distribution alone -> exp 26/28 DIES.")
else:
    print("\nVERDICT: the gap survives the parametric null -> factorization failure is real,")
    print("and the shortfall of observed vs null R^2 measures genuine pair-level structure.")
