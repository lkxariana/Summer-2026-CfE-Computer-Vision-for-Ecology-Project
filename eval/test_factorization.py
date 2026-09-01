import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from antheia.config import load_config

# Integrated-model reviewers' key warning: for interaction data the detection term is assumed to
# factorize as p_i * q_j. But an iNaturalist flower-visit photo documents BOTH partners in a single
# act, so p_i and q_j should be coupled and the product form should fail. Test: fit species main
# effects to log documentation intensity and measure how much PAIR-specific structure remains.
cfg = load_config()
edges = pd.read_parquet(cfg["edges"])
prov = pd.read_csv(cfg["edges"].parent / "edge_provenance_v1.csv")
e = edges.merge(prov, on=["plant", "pollinator"], how="inner")
e["curated"] = (e["n_inat"] < e["n"]).astype(int)


def factorization_fit(d, label):
    """R^2 of an additive species-main-effects model for log n_records (product form in count space)."""
    y = np.log1p(d["n"].to_numpy(float))
    pl = pd.Categorical(d["plant"])
    po = pd.Categorical(d["pollinator"])
    n = len(d)
    X = sparse.hstack([
        sparse.csr_matrix((np.ones(n), (np.arange(n), pl.codes)), shape=(n, len(pl.categories))),
        sparse.csr_matrix((np.ones(n), (np.arange(n), po.codes)), shape=(n, len(po.categories))),
    ]).tocsr()
    rng = np.random.default_rng(0)
    idx = rng.permutation(n)
    cut = int(0.8 * n)
    tr, te = idx[:cut], idx[cut:]
    m = Ridge(alpha=1.0, fit_intercept=True, solver="sparse_cg", max_iter=2000)
    m.fit(X[tr], y[tr])
    pred = m.predict(X[te])
    ss_res = float(((y[te] - pred) ** 2).sum())
    ss_tot = float(((y[te] - y[tr].mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot
    print(f"{label:<34}n={n:>7,}  R^2(species main effects)={r2:>6.3f}  "
          f"residual SD={np.sqrt(ss_res / len(te)):.3f}")
    return r2


print("Does documentation intensity factorize into species main effects?")
print("(high R^2 => product form p_i*q_j plausible; low R^2 => pair-specific coupling dominates)\n")
r_all = factorization_fit(e, "all edges")
r_inat = factorization_fit(e[e["curated"] == 0], "iNaturalist-only edges")
r_cur = factorization_fit(e[e["curated"] == 1], "curated-supported edges")

print(f"\nmulti-record edges only (n>=2), where intensity is better estimated:")
e2 = e[e["n"] >= 2]
factorization_fit(e2[e2["curated"] == 0], "  iNat-only, n>=2")
factorization_fit(e2[e2["curated"] == 1], "  curated, n>=2")

print("\nInterpretation:")
if r_inat < 0.35:
    print(f"  iNat R^2={r_inat:.3f} — species main effects explain little; documentation intensity is")
    print("  strongly PAIR-specific, consistent with single-act coupling (one photo documents both).")
    print("  The p_i*q_j detection factorization that integrated interaction models assume is NOT")
    print("  supported for the dominant data source.")
else:
    print(f"  iNat R^2={r_inat:.3f} — product form is defensible for the dominant source.")
print(f"  curated vs iNat R^2: {r_cur:.3f} vs {r_inat:.3f}")
