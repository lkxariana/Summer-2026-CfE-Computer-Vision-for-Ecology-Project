import argparse
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from antheia import models
from antheia.config import load_config
from antheia.metrics import bootstrap_mean
from antheia.pairs import load_split
from antheia.store import FeatureStore
from antheia.taxonomy import TaxonomyAffinity

GEO = ["vf", "vp", "n", "delta", "delta_local", "frs", "prs"]
SPECS = {"geo": GEO, "geo_taxshare": GEO + ["taxshare"]}   # shares only — smoother than raw counts
GRID = [
    {"max_iter": 400, "learning_rate": 0.08, "min_samples_leaf": 40, "l2_regularization": 1.0},
    {"max_iter": 800, "learning_rate": 0.05, "min_samples_leaf": 40, "l2_regularization": 1.0},
    {"max_iter": 400, "learning_rate": 0.08, "min_samples_leaf": 100, "l2_regularization": 5.0},
    {"max_iter": 800, "learning_rate": 0.05, "min_samples_leaf": 100, "l2_regularization": 5.0},
    {"max_iter": 400, "learning_rate": 0.08, "min_samples_leaf": 40, "l2_regularization": 1.0, "max_leaf_nodes": 15},
    {"max_iter": 800, "learning_rate": 0.05, "min_samples_leaf": 100, "l2_regularization": 5.0, "max_leaf_nodes": 15},
]


def main():
    ap = argparse.ArgumentParser(description="GBM tuning on the val plants; single final test readout")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    store = FeatureStore(cfg)
    edges = pd.read_parquet(cfg["edges"])
    known_pos = set(zip(edges["plant"], edges["pollinator"]))
    seed, nboot = cfg["eval"]["seed"], cfg["eval"]["bootstrap_n"]
    split = load_split(cfg["edges"].parent / "split_v1.json")
    train_pos = edges[edges["plant"].isin(split["train"])]
    val_pos = edges[edges["plant"].isin(split["val"])]
    test_pos = edges[edges["plant"].isin(split["test"])]
    aff = TaxonomyAffinity(store, train_pos, cfg["paths"]["globi"])

    def bp(p_idx, q_idx, c):
        if c == "taxshare":
            t = aff.pairs(p_idx, q_idx)
            return t[:, [1, 3]]
        return store.assemble(p_idx, q_idx, (c,))

    def bplant(p, c):
        if c == "taxshare":
            return aff.plant(p)[:, [1, 3]]
        return store.assemble_plant(p, (c,))

    pi, qp, qn = models.rank_contrast_sets(train_pos, store, 10, seed, known_pos)
    comps = sorted({c for s in SPECS.values() for c in s})
    Xp = {c: bp(pi, qp, c) for c in comps}
    Xn = {c: bp(pi, qn, c) for c in comps}
    pos_rows = np.arange(0, len(pi), 10)
    y = np.concatenate([np.ones(len(pos_rows)), np.zeros(len(pi))])

    def rank_eval(clf, spec, pos_df):
        partners = {sp: set(store.idx_polls(g["pollinator"])) for sp, g in pos_df.groupby("plant")}
        r10s = []
        for sp, part in partners.items():
            p = store.p2i[sp]
            Xq = np.hstack([bplant(p, c) for c in spec])
            s = clf.predict_proba(Xq)[:, 1]
            t10 = set(np.argpartition(-s, 10)[:10].tolist())
            r10s.append(len(part & t10) / len(part))
        return np.array(r10s)

    results, fitted = [], {}
    for spec_name, spec in SPECS.items():
        X = np.vstack([np.hstack([Xp[c][pos_rows] for c in spec]), np.hstack([Xn[c] for c in spec])])
        for gi, params in enumerate(GRID):
            t0 = time.time()
            clf = HistGradientBoostingClassifier(random_state=seed, **params)
            clf.fit(X, y)
            val_r10 = rank_eval(clf, spec, val_pos).mean()
            results.append({"spec": spec_name, "grid": gi, **params, "val_recall@10": val_r10})
            fitted[(spec_name, gi)] = (clf, spec)
            print(f"  {spec_name} grid{gi} val R@10 {val_r10:.4f} ({time.time() - t0:.0f}s)", flush=True)

    res = pd.DataFrame(results).sort_values("val_recall@10", ascending=False)
    best = res.iloc[0]
    clf, spec = fitted[(best["spec"], best["grid"])]
    test_r = rank_eval(clf, spec, test_pos)
    r10, lo, hi, _ = bootstrap_mean(test_r, nboot, seed)
    print(f"\nBEST on val: {best['spec']} grid{int(best['grid'])} (val {best['val_recall@10']:.4f})")
    print(f"FINAL TEST recall@10: {r10:.4f} [{lo:.4f},{hi:.4f}]")
    res.to_csv(cfg["edges"].parent / "gbm_tune_v1.csv", index=False)
    with open(cfg["edges"].parent / "gbm_tune_v1_final.txt", "w") as f:
        f.write(f"best={best['spec']} grid{int(best['grid'])} val={best['val_recall@10']:.4f} "
                f"test_recall@10={r10:.4f} [{lo:.4f},{hi:.4f}]\n")


if __name__ == "__main__":
    main()
