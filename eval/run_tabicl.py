"""TabICL on the pair-scoring problem: is "trees beat neural nets on tabular" still true here?

The conclusion that a boosted ranker beats every neural model on these features rests on Grinsztajn,
Oyallon & Varoquaux (2022). Tabular foundation models have moved since: TabPFN v2 (Nature, 2025) and
TabICL, which scales in-context learning to hundreds of thousands of rows, now frequently beat GBDTs
on small-to-medium tabular problems. Our pair problem is ~132k training rows over 122 features, in
range.

Same features as the boosted ranker, same split, same metrics, so the comparison isolates the model
class. In-context learning has no training loop: the context is the labelled sample, and inference
runs at roughly 800 rows/s, so scoring all 663 x 13,124 pairs takes about three hours.
"""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from antheia.baselines import REGISTRY
from antheia.metrics import bootstrap_mean, retrieval_metrics
from antheia.store import UniverseStore


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--context", type=int, default=50000, help="labelled rows given as context")
    ap.add_argument("--n-neg", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    from tabicl import TabICLClassifier

    store = UniverseStore(curves="modelled")
    split = json.load(open(ROOT / "data/splits/plants_75_10_15.json"))
    e = pd.read_parquet(ROOT / "data/network/edges.parquet")
    e = e[e.plant.isin(store.p2i) & e.pollinator.isin(store.q2i)]
    train = e[e.plant.isin(set(split["train"]))]
    test = e[e.plant.isin(set(split["val"]))]; test = test[test.tier == "A"]
    tp = sorted(set(test.plant))
    part = {sp: set(store.idx_polls(g.pollinator)) for sp, g in test.groupby("plant")}

    # identical feature construction to the boosted ranker, so only the model class differs
    gbm = REGISTRY["ours_gbm"]()
    gbm.fit(train, store)
    rng = np.random.default_rng(args.seed)
    pi = store.idx_plants(train["plant"]); qi = store.idx_polls(train["pollinator"])
    known = set(zip(pi.tolist(), qi.tolist()))
    np_ = np.repeat(pi, args.n_neg)
    nq_ = rng.integers(0, len(store.polls), len(np_))
    keep = [i for i, (a, b) in enumerate(zip(np_, nq_)) if (a, b) not in known]
    np_, nq_ = np_[keep], nq_[keep]
    Xp, Xn = gbm._features(pi, qi), gbm._features(np_, nq_)
    X = np.vstack([Xp, Xn]); y = np.concatenate([np.ones(len(Xp)), np.zeros(len(Xn))])
    idx = rng.permutation(len(X))[:args.context]
    X, y = X[idx].astype(np.float32), y[idx].astype(int)
    print(f"[context] {X.shape} rows, positive rate {y.mean():.4f}", flush=True)

    clf = TabICLClassifier(device="cuda")
    t0 = time.time()
    clf.fit(X, y)
    print(f"[fit] {time.time() - t0:.1f}s", flush=True)

    nq = len(store.polls)
    S = np.empty((len(tp), nq), np.float32)
    t0 = time.time()
    for r, sp in enumerate(tp):
        p = store.p2i[sp]
        F = gbm._features(np.full(nq, p), np.arange(nq)).astype(np.float32)
        S[r] = clf.predict_proba(F)[:, 1]
        if r % 25 == 0:
            el = time.time() - t0
            print(f"  plant {r + 1}/{len(tp)}  {el / max(r + 1, 1) * (len(tp) - r - 1) / 60:.0f} min left",
                  flush=True)

    pp = pd.DataFrame([retrieval_metrics(S[i], part[sp], ks=(10, 20)) for i, sp in enumerate(tp)])
    Y = np.zeros(S.shape, np.int8)
    for i, sp in enumerate(tp):
        Y[i, list(part[sp])] = 1
    pr = average_precision_score(Y.ravel(), S.ravel().astype(np.float64))
    m, lo, hi, _ = bootstrap_mean(pp["nrecall@10"].to_numpy(), 1000, args.seed)
    print(f"\n  TabICL   nR@10 {m:.4f} [{lo:.4f},{hi:.4f}]  MAP {pp['ap'].mean():.4f}  PR {pr:.4f}")
    print(f"  (boosted ranker on identical features: nR@10 0.3259  MAP 0.1822  PR 0.1032)")
    pd.DataFrame([dict(config="TabICL", nrecall10=m, lo=lo, hi=hi,
                       map=pp["ap"].mean(), pr_auc=pr)]).to_csv(
        ROOT / "results/tabicl_val_tierA.csv", index=False)
    np.save(ROOT / "results/scores_tabicl_val.npy", S)
    print("[wrote] results/tabicl_val_tierA.csv")


if __name__ == "__main__":
    main()
