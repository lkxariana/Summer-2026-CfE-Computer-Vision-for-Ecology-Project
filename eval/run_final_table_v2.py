import argparse
import copy
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from antheia import models
from antheia.config import load_config
from antheia.metrics import bootstrap_mean, pooled_metrics, retrieval_metrics
from antheia.pairs import load_split
from antheia.store import FeatureStore
from antheia.taxonomy import TaxonomyAffinity
from antheia.twotower import Data, make_ranker, train_model

GEO = ["vf", "vp", "n", "delta", "delta_local", "frs", "prs"]
LINEAR = {"rank_n": ["n"], "rank_n_tax_ldelta": ["n", "tax", "delta_local"]}
NN_EMBS = ["none", "bioclip", "bioclip2"]


def main():
    ap = argparse.ArgumentParser(description="Final roster: refit all models, per-plant scores, paired significance")
    ap.add_argument("--config", default=None)
    ap.add_argument("--device", default="cuda:1")
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
    aff = TaxonomyAffinity(store, train_pos, cfg["paths"]["globi"])
    out_dir = cfg["edges"].parent / "final_scores"
    out_dir.mkdir(exist_ok=True)

    def bp(p_idx, q_idx, c):
        return aff.pairs(p_idx, q_idx) if c == "tax" else store.assemble(p_idx, q_idx, (c,))

    # Shared training contrasts + components.
    pi, qp, qn = models.rank_contrast_sets(train_pos, store, 10, seed, known_pos)
    comps = sorted({c for s in LINEAR.values() for c in s} | set(GEO))
    Xp = {c: bp(pi, qp, c) for c in comps}
    Xn = {c: bp(pi, qn, c) for c in comps}
    scorers = {"degree_rank": (None, None)}

    for name, spec in LINEAR.items():
        d = np.hstack([Xp[c] for c in spec]) - np.hstack([Xn[c] for c in spec])
        X, y = np.vstack([d, -d]), np.concatenate([np.ones(len(d)), np.zeros(len(d))])
        pipe = Pipeline([("scale", StandardScaler(with_mean=False)),
                         ("lr", LogisticRegression(max_iter=1000, random_state=seed, fit_intercept=False))])
        pipe.fit(X, y)
        scorers[name] = (spec, lambda M, pp=pipe: pp.decision_function(M))

    tune = pd.read_csv(cfg["edges"].parent / "gbm_tune_v1.csv").sort_values("val_recall@10", ascending=False)
    best = tune[tune["spec"] == "geo"].iloc[0]
    params = {k: best[k] for k in ["max_iter", "learning_rate", "min_samples_leaf", "l2_regularization"]}
    params = {k: (int(v) if k in ("max_iter", "min_samples_leaf") else float(v)) for k, v in params.items()}
    if not np.isnan(best.get("max_leaf_nodes", np.nan)):
        params["max_leaf_nodes"] = int(best["max_leaf_nodes"])
    pos_rows = np.arange(0, len(pi), 10)
    Xg = np.vstack([np.hstack([Xp[c][pos_rows] for c in GEO]), np.hstack([Xn[c] for c in GEO])])
    yg = np.concatenate([np.ones(len(pos_rows)), np.zeros(len(pi))])
    gbm = HistGradientBoostingClassifier(random_state=seed, **params)
    gbm.fit(Xg, yg)
    scorers["gbm_geo_tuned"] = (GEO, lambda M: gbm.predict_proba(M)[:, 1])
    print(f"linear + gbm ready (gbm params {params})", flush=True)

    # Neural rosters: shared pools/wide via a base Data, dense towers swapped per variant.
    t0 = time.time()
    base = Data(store, edges, split, cfg, emb=None, seed=seed)
    print(f"base NN data ({time.time() - t0:.0f}s)", flush=True)
    val_partners = {sp: set(store.idx_polls(g["pollinator"]))
                    for sp, g in edges[edges["plant"].isin(split["val"])].groupby("plant")}
    val_wide = {sp: base.wide_plant(store.p2i[sp]) for sp in val_partners}
    test_wide = {sp: base.wide_plant(store.p2i[sp]) for sp in test_pos["plant"].unique()}
    print(f"wide caches built ({time.time() - t0:.0f}s)", flush=True)

    nn_rankers = {}
    for embname in NN_EMBS:
        data = copy.copy(base)
        if embname != "none":
            prefix = {"bioclip": "bioclip_text", "bioclip2": "bioclip2_text"}[embname]
            ep = np.load(cfg["cache_dir"] / f"{prefix}_plants.npy").astype(np.float32)
            eq = np.load(cfg["cache_dir"] / f"{prefix}_polls.npy").astype(np.float32)
            data.p_dense = np.hstack([base.p_dense, ep])
            data.q_dense = np.hstack([base.q_dense, eq])

        def val_eval(m, d=data):
            scorer = make_ranker(m, d, args.device)
            return float(np.mean([len(part & set(np.argpartition(-scorer(store.p2i[sp], val_wide[sp]), 10)[:10].tolist())) / len(part)
                                  for sp, part in val_partners.items()]))

        m, _ = train_model(data, args.device, seed=seed, val_eval=val_eval, log=lambda s: print(f"  [{embname}] {s}", flush=True))
        torch.save(m.state_dict(), cfg["cache_dir"] / f"twotower_{embname}.pt")
        nn_rankers[f"twotower_{embname}"] = (make_ranker(m, data, args.device), data)

    # Per-plant evaluation for every model.
    partners_all = {sp: set(store.idx_polls(g["pollinator"])) for sp, g in test_pos.groupby("plant")}
    partners_t1 = {sp: set(store.idx_polls(g["pollinator"]))
                   for sp, g in test_pos[test_pos["tier1"]].groupby("plant")}
    names = list(scorers) + list(nn_rankers)
    keys = ("recall@10", "recall@20", "ndcg@10", "ndcg@20", "nrecall@10", "rank_first", "t1_recall@10")
    rec = {n: {"plant": [], **{k: [] for k in keys}} for n in names}
    for i, (sp, part) in enumerate(partners_all.items()):
        p = store.p2i[sp]
        blk = {c: (aff.plant(p) if c == "tax" else store.assemble_plant(p, (c,))) for c in comps}
        pt1 = partners_t1.get(sp, set())
        svec = {"degree_rank": store.Prs.astype(np.float64)}
        for n, (spec, fn) in scorers.items():
            if spec is not None:
                svec[n] = fn(np.hstack([blk[c] for c in spec]))
        for n, (rk, _) in nn_rankers.items():
            svec[n] = rk(p, test_wide[sp])
        for n, s in svec.items():
            m = retrieval_metrics(s, part)
            rec[n]["plant"].append(sp)
            for key in ("recall@10", "recall@20", "ndcg@10", "ndcg@20", "nrecall@10", "rank_first"):
                rec[n][key].append(m[key])
            if pt1:
                sm = s.copy()
                sm[list(part - pt1)] = -np.inf
                rec[n]["t1_recall@10"].append(retrieval_metrics(sm, pt1)["recall@10"])
        if (i + 1) % 150 == 0:
            print(f"  eval {i + 1}/{len(partners_all)}", flush=True)

    rows = []
    for n in names:
        pd.DataFrame({"plant": rec[n]["plant"], **{k: rec[n][k] for k in
                      ("recall@10", "recall@20", "ndcg@10", "ndcg@20", "rank_first")}}
                     ).to_csv(out_dir / f"{n}.csv", index=False)
        r10, lo, hi, _ = bootstrap_mean(np.array(rec[n]["recall@10"]), nboot, seed)
        nd10, ndlo, ndhi, _ = bootstrap_mean(np.array(rec[n]["ndcg@10"]), nboot, seed)
        t10, tlo, thi, _ = bootstrap_mean(np.array(rec[n]["t1_recall@10"]), nboot, seed)
        med = float(np.nanmedian(rec[n]["rank_first"]))
        rows.append({"model": n, "recall@10": r10, "r10_lo": lo, "r10_hi": hi,
                     "recall@20": np.mean(rec[n]["recall@20"]),
                     "ndcg@10": nd10, "nd10_lo": ndlo, "nd10_hi": ndhi,
                     "ndcg@20": np.mean(rec[n]["ndcg@20"]),
                     "nrecall@10": np.mean(rec[n]["nrecall@10"]),
                     "median_rank_first": med, "t1_recall@10": t10})
        print(f"  {n:<20} R@10 {r10:.4f} R@20 {np.mean(rec[n]['recall@20']):.4f} | "
              f"nDCG@10 {nd10:.4f} nDCG@20 {np.mean(rec[n]['ndcg@20']):.4f} | "
              f"medRank {med:.0f} | T1 R@10 {t10:.4f}", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(cfg["edges"].parent / "final_table_v2.csv", index=False)

    # Paired plant-level bootstrap on r10_all.
    rng = np.random.default_rng(seed)
    R = {n: np.array(rec[n]["recall@10"]) for n in names}
    npl = len(next(iter(R.values())))
    idx = rng.integers(0, npl, size=(10_000, npl))
    sig = []
    for a in names:
        for b in names:
            if a >= b:
                continue
            diffs = R[a][idx].mean(1) - R[b][idx].mean(1)
            d = R[a].mean() - R[b].mean()
            p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
            sig.append({"a": a, "b": b, "delta_r10": d,
                        "lo": np.percentile(diffs, 2.5), "hi": np.percentile(diffs, 97.5),
                        "p_two_sided": max(p, 1e-4)})
    pd.DataFrame(sig).to_csv(cfg["edges"].parent / "final_significance_v2.csv", index=False)
    print("\n" + table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
