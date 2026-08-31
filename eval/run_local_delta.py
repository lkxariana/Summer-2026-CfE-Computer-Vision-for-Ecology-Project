import argparse
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from antheia import models
from antheia.config import load_config
from antheia.metrics import bootstrap_mean, pair_metrics, rank_metrics
from antheia.pairs import load_split, sample_negatives
from antheia.store import FeatureStore

RANKERS = ["rank_n", "rank_ldelta", "rank_n_ldelta", "rank_scalar_local", "rank_full_local"]
COOC_MODELS = ["rank_n", "rank_n_ldelta", "rank_scalar_local"]


def main():
    ap = argparse.ArgumentParser(description="Local per-bin delta: rankers + classifier on the frozen split")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    store = FeatureStore(cfg)
    edges = pd.read_parquet(cfg["edges"])
    known_pos = set(zip(edges["plant"], edges["pollinator"]))
    seed, nboot, ks = cfg["eval"]["seed"], cfg["eval"]["bootstrap_n"], cfg["eval"]["ks"]
    split = load_split(cfg["edges"].parent / "split_v1.json")
    train_pos = edges[edges["plant"].isin(split["train"])]
    test_pos = edges[edges["plant"].isin(split["test"])]

    # Sanity: how does local delta relate to the range-averaged delta?
    pi, qp, qn = models.rank_contrast_sets(train_pos, store, 10, seed, known_pos)
    sample = np.random.default_rng(0).choice(len(pi), 20_000, replace=False)
    d_g = store.assemble(pi[sample], qp[sample], ("delta",))[:, 0]
    d_l = store.assemble(pi[sample], qp[sample], ("delta_local",))[:, 0]
    print(f"delta vs delta_local (20k positive pairs): corr {np.corrcoef(d_g, d_l)[0, 1]:.3f} | "
          f"means {d_g.mean():.3f}/{d_l.mean():.3f} | local zeros {np.mean(d_l == 0):.2%}", flush=True)

    # Assemble contrast components once, reuse across specs.
    comps = sorted({c for r in RANKERS for c in models.RANK_SPECS[r]})
    t0 = time.time()
    Xp = {c: store.assemble(pi, qp, (c,)) for c in comps}
    Xn = {c: store.assemble(pi, qn, (c,)) for c in comps}
    print(f"contrast components assembled ({time.time() - t0:.0f}s)", flush=True)

    neg_te = sample_negatives(test_pos, split["test"], store,
                              cfg["negatives"]["ratio"], cfg["negatives"]["mode"], seed + 1, known_pos)
    te_base = pd.concat([test_pos[["plant", "pollinator"]].assign(label=1), neg_te.assign(label=0)], ignore_index=True)
    te_pi, te_qi = store.idx_plants(te_base["plant"]), store.idx_polls(te_base["pollinator"])
    te_X = {c: store.assemble(te_pi, te_qi, (c,)) for c in comps}
    partners = {sp: set(store.idx_polls(g["pollinator"])) for sp, g in test_pos.groupby("plant")}

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    rows, pipes = [], {}
    for name in RANKERS:
        spec = models.RANK_SPECS[name]
        t0 = time.time()
        d = np.hstack([Xp[c] for c in spec]) - np.hstack([Xn[c] for c in spec])
        X = np.vstack([d, -d])
        y = np.concatenate([np.ones(len(d)), np.zeros(len(d))])
        pipe = Pipeline([("scale", StandardScaler(with_mean=False)),
                         ("lr", LogisticRegression(max_iter=1000, random_state=seed, fit_intercept=False))])
        pipe.fit(X, y)
        pipes[name] = pipe
        pm = pair_metrics(te_base["label"].values, pipe.decision_function(np.hstack([te_X[c] for c in spec])))
        rk = rank_metrics(models.spec_scorer(store, spec, pipe), partners, store, ks)
        r10, r10_lo, r10_hi, _ = bootstrap_mean(rk["recall@10"].values, nboot, seed)
        rows.append({"model": name, "recall@10": r10, "r10_lo": r10_lo, "r10_hi": r10_hi,
                     "recall@50": rk["recall@50"].mean(), "hit@10": rk["hit@10"].mean(),
                     "pr_auc_pooled": pm["pr_auc"]})
        print(f"  {name:<18} R@10 {r10:.4f} [{r10_lo:.4f},{r10_hi:.4f}]  R@50 {rk['recall@50'].mean():.4f}"
              f"  hit@10 {rk['hit@10'].mean():.4f}  pooledPR {pm['pr_auc']:.4f}  ({time.time() - t0:.0f}s)", flush=True)

    # Classifier with local delta, for pooled-PR comparison against scalar (0.6517).
    neg_tr = sample_negatives(train_pos, sorted(set(store.plants) - set(split["test"]) - set(split["val"])),
                              store, cfg["negatives"]["ratio"], cfg["negatives"]["mode"], seed, known_pos)
    tr = pd.concat([train_pos[["plant", "pollinator"]].assign(label=1), neg_tr.assign(label=0)], ignore_index=True)
    clf = models.train(store, "scalar_local", tr, tr["label"].values, seed)
    pm = pair_metrics(te_base["label"].values, models.pair_scores(store, "scalar_local", clf, te_base))
    rows.append({"model": "clf_scalar_local", "pr_auc_pooled": pm["pr_auc"]})
    print(f"  clf_scalar_local pooledPR {pm['pr_auc']:.4f}  (classifier scalar was 0.6517)", flush=True)

    # Co-occurring-candidates slice.
    rng = np.random.default_rng(0)
    cooc_rows = []
    for sp, part in partners.items():
        p = store.p2i[sp]
        cand = np.flatnonzero(store.N_full[p] > 0)
        part_c = part & set(cand.tolist())
        if not part_c:
            continue
        jitter = rng.uniform(0, 1e-6, size=len(cand))
        row = {}
        for name in COOC_MODELS:
            spec = models.RANK_SPECS[name]
            s = pipes[name].decision_function(store.assemble_plant(p, spec)[cand]) + jitter
            top10 = cand[np.argsort(-s)[:10]]
            row[name] = len(part_c & set(top10.tolist())) / len(part_c)
        cooc_rows.append(row)
    cooc = pd.DataFrame(cooc_rows)
    print("\nrecall@10 within co-occurring candidates:")
    for name in COOC_MODELS:
        print(f"  {name:<18} {cooc[name].mean():.4f}")

    df = pd.DataFrame(rows)
    out = cfg["edges"].parent / "local_delta_v1.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
