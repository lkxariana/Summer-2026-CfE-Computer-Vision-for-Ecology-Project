import argparse
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from antheia import models
from antheia.config import load_config
from antheia.metrics import bootstrap_mean
from antheia.pairs import load_split
from antheia.store import FeatureStore
from antheia.taxonomy import TaxonomyAffinity
from antheia.twotower import Data, WideDeep, make_ranker

GEO = ["vf", "vp", "n", "delta", "delta_local", "frs", "prs"]


def norm_recall(hits, k, degree):
    """Recall@k divided by its ceiling min(1, k/degree) — comparable across plant degrees."""
    return hits / min(1.0, k / degree)


def main():
    ap = argparse.ArgumentParser(description="Rank-average ensemble of GBM + two-tower, with degree-normalized metrics")
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

    pi, qp, qn = models.rank_contrast_sets(train_pos, store, 10, seed, known_pos)
    Xp = {c: store.assemble(pi, qp, (c,)) for c in GEO}
    Xn = {c: store.assemble(pi, qn, (c,)) for c in GEO}
    tune = pd.read_csv(cfg["edges"].parent / "gbm_tune_v1.csv").sort_values("val_recall@10", ascending=False)
    b = tune[tune["spec"] == "geo"].iloc[0]
    gbm = HistGradientBoostingClassifier(random_state=seed, max_iter=int(b["max_iter"]),
                                         learning_rate=float(b["learning_rate"]),
                                         min_samples_leaf=int(b["min_samples_leaf"]),
                                         l2_regularization=float(b["l2_regularization"]))
    pos_rows = np.arange(0, len(pi), 10)
    gbm.fit(np.vstack([np.hstack([Xp[c][pos_rows] for c in GEO]), np.hstack([Xn[c] for c in GEO])]),
            np.concatenate([np.ones(len(pos_rows)), np.zeros(len(pi))]))
    print("gbm refit", flush=True)

    t0 = time.time()
    data = Data(store, edges, split, cfg, emb=None, seed=seed)
    ep = np.load(cfg["cache_dir"] / "bioclip_text_plants.npy").astype(np.float32)
    eq = np.load(cfg["cache_dir"] / "bioclip_text_polls.npy").astype(np.float32)
    data.p_dense = np.hstack([data.p_dense, ep])
    data.q_dense = np.hstack([data.q_dense, eq])
    m = WideDeep(data.p_dense.shape[1], data.q_dense.shape[1],
                 data.n_pgen, data.n_pfam, data.n_qgen, data.n_qfam).to(args.device)
    m.load_state_dict(torch.load(cfg["cache_dir"] / "twotower_bioclip.pt"))
    nn_score = make_ranker(m, data, args.device)
    print(f"nn checkpoint loaded ({time.time() - t0:.0f}s)", flush=True)

    partners = {sp: set(store.idx_polls(g["pollinator"])) for sp, g in test_pos.groupby("plant")}
    partners_t1 = {sp: set(store.idx_polls(g["pollinator"]))
                   for sp, g in test_pos[test_pos["tier1"]].groupby("plant")}
    names = ["gbm_geo_tuned", "twotower_bioclip", "ens_rank_avg", "ens_score_z"]
    rec = {n: {"r10": [], "nr10": [], "r50": [], "hit": [], "t1": []} for n in names}

    for i, (sp, part) in enumerate(partners.items()):
        p = store.p2i[sp]
        s_gbm = gbm.predict_proba(np.hstack([store.assemble_plant(p, (c,)) for c in GEO]))[:, 1]
        s_nn = nn_score(p, data.wide_plant(p))
        r_gbm = np.argsort(np.argsort(-s_gbm))
        r_nn = np.argsort(np.argsort(-s_nn))
        svec = {"gbm_geo_tuned": s_gbm, "twotower_bioclip": s_nn,
                "ens_rank_avg": -(r_gbm + r_nn) / 2.0,
                "ens_score_z": ((s_gbm - s_gbm.mean()) / (s_gbm.std() + 1e-9)
                                + (s_nn - s_nn.mean()) / (s_nn.std() + 1e-9))}
        pt1 = partners_t1.get(sp, set())
        for n, s in svec.items():
            top50 = np.argpartition(-s, 50)[:50]
            top50 = top50[np.argsort(-s[top50])]
            t10 = set(top50[:10].tolist())
            r10 = len(part & t10) / len(part)
            rec[n]["r10"].append(r10)
            rec[n]["nr10"].append(norm_recall(r10, 10, len(part)))
            rec[n]["r50"].append(len(part & set(top50.tolist())) / len(part))
            rec[n]["hit"].append(float(len(part & t10) > 0))
            if pt1:
                sm = s.copy()
                sm[list(part - pt1)] = -np.inf
                rec[n]["t1"].append(len(pt1 & set(np.argpartition(-sm, 10)[:10].tolist())) / len(pt1))
        if (i + 1) % 150 == 0:
            print(f"  {i + 1}/{len(partners)}", flush=True)

    rows = []
    for n in names:
        r10, lo, hi, _ = bootstrap_mean(np.array(rec[n]["r10"]), nboot, seed)
        t1, tlo, thi, _ = bootstrap_mean(np.array(rec[n]["t1"]), nboot, seed)
        rows.append({"model": n, "recall@10": r10, "lo": lo, "hi": hi,
                     "norm_recall@10": np.mean(rec[n]["nr10"]), "recall@50": np.mean(rec[n]["r50"]),
                     "hit@10": np.mean(rec[n]["hit"]), "t1_recall@10": t1})
        print(f"  {n:<18} R@10 {r10:.4f} [{lo:.4f},{hi:.4f}] normR@10 {np.mean(rec[n]['nr10']):.4f} "
              f"R@50 {np.mean(rec[n]['r50']):.4f} hit {np.mean(rec[n]['hit']):.4f} T1 {t1:.4f}", flush=True)

    rng = np.random.default_rng(seed)
    A, B = np.array(rec["ens_rank_avg"]["r10"]), np.array(rec["twotower_bioclip"]["r10"])
    idx = rng.integers(0, len(A), size=(10_000, len(A)))
    d = (A[idx] - B[idx]).mean(1)
    print(f"\nensemble vs best single: Δ={A.mean() - B.mean():+.4f} "
          f"[{np.percentile(d, 2.5):+.4f},{np.percentile(d, 97.5):+.4f}] "
          f"p={2 * min((d <= 0).mean(), (d >= 0).mean()):.4f}")
    pd.DataFrame(rows).to_csv(cfg["edges"].parent / "ensemble_v1.csv", index=False)


if __name__ == "__main__":
    main()
