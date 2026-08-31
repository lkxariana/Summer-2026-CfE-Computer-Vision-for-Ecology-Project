import argparse
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

GEO = ["vf", "vp", "n", "delta", "delta_local", "frs", "prs"]
LINEAR = {"rank_n": ["n"], "rank_n_tax": ["n", "tax"], "rank_n_tax_ldelta": ["n", "tax", "delta_local"]}


def main():
    ap = argparse.ArgumentParser(description="Do the new winners transfer to curated (Tier-1) labels? All models trained on train_all.")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    store = FeatureStore(cfg)
    edges = pd.read_parquet(cfg["edges"])
    prov = pd.read_csv(cfg["edges"].parent / "edge_provenance_v1.csv")
    edges = edges.merge(prov, on=["plant", "pollinator"], how="left")
    edges["tier1"] = edges["n_inat"].notna() & (edges["n_inat"] < edges["n"])
    known_pos = set(zip(edges["plant"], edges["pollinator"]))
    seed, nboot = cfg["eval"]["seed"], cfg["eval"]["bootstrap_n"]
    split = load_split(cfg["edges"].parent / "split_v1.json")
    train_pos = edges[edges["plant"].isin(split["train"])]
    test_pos = edges[edges["plant"].isin(split["test"])]
    test_t1 = test_pos[test_pos["tier1"]]
    aff = TaxonomyAffinity(store, train_pos, cfg["paths"]["globi"])

    def bp(p_idx, q_idx, c):
        return aff.pairs(p_idx, q_idx) if c == "tax" else store.assemble(p_idx, q_idx, (c,))

    pi, qp, qn = models.rank_contrast_sets(train_pos, store, 10, seed, known_pos)
    comps = sorted({c for s in LINEAR.values() for c in s} | set(GEO))
    Xp = {c: bp(pi, qp, c) for c in comps}
    Xn = {c: bp(pi, qn, c) for c in comps}

    scorers = {}
    for name, spec in LINEAR.items():
        d = np.hstack([Xp[c] for c in spec]) - np.hstack([Xn[c] for c in spec])
        X, y = np.vstack([d, -d]), np.concatenate([np.ones(len(d)), np.zeros(len(d))])
        pipe = Pipeline([("scale", StandardScaler(with_mean=False)),
                         ("lr", LogisticRegression(max_iter=1000, random_state=seed, fit_intercept=False))])
        pipe.fit(X, y)
        scorers[name] = (spec, lambda M, pp=pipe: pp.decision_function(M))

    pos_rows = np.arange(0, len(pi), 10)
    Xg = np.vstack([np.hstack([Xp[c][pos_rows] for c in GEO]), np.hstack([Xn[c] for c in GEO])])
    yg = np.concatenate([np.ones(len(pos_rows)), np.zeros(len(pi))])
    gbm = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08, min_samples_leaf=40,
                                         l2_regularization=1.0, random_state=seed)
    gbm.fit(Xg, yg)
    scorers["gbm_geo"] = (GEO, lambda M: gbm.predict_proba(M)[:, 1])
    print("models trained", flush=True)

    partners_all = {sp: set(store.idx_polls(g["pollinator"])) for sp, g in test_pos.groupby("plant")}
    partners_t1 = {sp: set(store.idx_polls(g["pollinator"])) for sp, g in test_t1.groupby("plant")}
    res = {(m, ev): [] for m in scorers for ev in ("all", "t1")}
    for i, (sp, part_all) in enumerate(partners_all.items()):
        p = store.p2i[sp]
        blk = {c: (aff.plant(p) if c == "tax" else store.assemble_plant(p, (c,))) for c in comps}
        part_t1 = partners_t1.get(sp, set())
        for name, (spec, fn) in scorers.items():
            s = fn(np.hstack([blk[c] for c in spec]))
            t10 = set(np.argpartition(-s, 10)[:10].tolist())
            res[(name, "all")].append(len(part_all & t10) / len(part_all))
            if part_t1:
                sm = s.copy()
                sm[list(part_all - part_t1)] = -np.inf
                t10m = set(np.argpartition(-sm, 10)[:10].tolist())
                res[(name, "t1")].append(len(part_t1 & t10m) / len(part_t1))
        if (i + 1) % 150 == 0:
            print(f"  {i + 1}/{len(partners_all)}", flush=True)

    rows = []
    for (m, ev), vals in res.items():
        r10, lo, hi, _ = bootstrap_mean(np.array(vals), nboot, seed)
        rows.append({"model": m, "eval": ev, "n_plants": len(vals), "recall@10": r10, "lo": lo, "hi": hi})
    df = pd.DataFrame(rows).sort_values(["eval", "model"])
    df.to_csv(cfg["edges"].parent / "transfer_v1.csv", index=False)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
