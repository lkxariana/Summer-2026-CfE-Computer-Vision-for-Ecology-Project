import argparse
import sys
import time
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
from antheia.store import FeatureStore
from antheia.taxonomy import TaxonomyAffinity
from antheia.twotower import Data, make_ranker, train_model

GEO = ["vf", "vp", "n", "delta", "delta_local", "frs", "prs"]
LINEAR = {"rank_n": ["n"], "rank_n_tax_ldelta": ["n", "tax", "delta_local"]}


def main():
    ap = argparse.ArgumentParser(description="Prospective validation: train on pre-cutoff edges, rank newly documented pairs")
    ap.add_argument("--config", default=None)
    ap.add_argument("--cutoff", type=int, default=2020)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--with-nn", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    store = FeatureStore(cfg)
    edges = pd.read_parquet(cfg["edges"])
    years = pd.read_csv(cfg["edges"].parent / "edge_years_v1.csv")
    edges = edges.merge(years, on=["plant", "pollinator"], how="left")
    known_pos = set(zip(edges["plant"], edges["pollinator"]))
    seed, nboot = cfg["eval"]["seed"], cfg["eval"]["bootstrap_n"]

    old = edges[edges["first_year"] <= args.cutoff].reset_index(drop=True)
    new = edges[edges["first_year"] > args.cutoff].reset_index(drop=True)
    # Discovery framing: rank only among plants already known before the cutoff.
    new = new[new["plant"].isin(set(old["plant"]))].reset_index(drop=True)
    print(f"cutoff {args.cutoff}: train edges {len(old):,} | newly documented test edges {len(new):,} "
          f"on {new['plant'].nunique():,} known plants", flush=True)

    aff = TaxonomyAffinity(store, old, cfg["paths"]["globi"])

    def bp(p_idx, q_idx, c):
        return aff.pairs(p_idx, q_idx) if c == "tax" else store.assemble(p_idx, q_idx, (c,))

    t0 = time.time()
    pi, qp, qn = models.rank_contrast_sets(old, store, 10, seed, known_pos)
    comps = sorted({c for s in LINEAR.values() for c in s} | set(GEO))
    Xp = {c: bp(pi, qp, c) for c in comps}
    Xn = {c: bp(pi, qn, c) for c in comps}
    print(f"contrasts ready ({time.time() - t0:.0f}s)", flush=True)

    scorers = {}
    for name, spec in LINEAR.items():
        d = np.hstack([Xp[c] for c in spec]) - np.hstack([Xn[c] for c in spec])
        X, y = np.vstack([d, -d]), np.concatenate([np.ones(len(d)), np.zeros(len(d))])
        pipe = Pipeline([("scale", StandardScaler(with_mean=False)),
                         ("lr", LogisticRegression(max_iter=1000, random_state=seed, fit_intercept=False))])
        pipe.fit(X, y)
        scorers[name] = (spec, lambda M, pp=pipe: pp.decision_function(M))

    tune = pd.read_csv(cfg["edges"].parent / "gbm_tune_v1.csv").sort_values("val_recall@10", ascending=False)
    best = tune[tune["spec"] == "geo"].iloc[0]
    params = {"max_iter": int(best["max_iter"]), "learning_rate": float(best["learning_rate"]),
              "min_samples_leaf": int(best["min_samples_leaf"]), "l2_regularization": float(best["l2_regularization"])}
    pos_rows = np.arange(0, len(pi), 10)
    gbm = HistGradientBoostingClassifier(random_state=seed, **params)
    gbm.fit(np.vstack([np.hstack([Xp[c][pos_rows] for c in GEO]), np.hstack([Xn[c] for c in GEO])]),
            np.concatenate([np.ones(len(pos_rows)), np.zeros(len(pi))]))
    scorers["gbm_geo_tuned"] = (GEO, lambda M: gbm.predict_proba(M)[:, 1])
    print("linear + gbm trained", flush=True)

    nn = None
    if args.with_nn:
        # Early-stop on held-out pre-cutoff plants so the neural model gets the same
        # selection treatment as the sklearn models (which were tuned on the main val split).
        t0 = time.time()
        old_plants = sorted(set(old["plant"]))
        rng_v = np.random.default_rng(seed)
        val_plants = set(rng_v.choice(old_plants, size=max(50, len(old_plants) // 10), replace=False))
        fit_edges = old[~old["plant"].isin(val_plants)].reset_index(drop=True)
        val_edges = old[old["plant"].isin(val_plants)]
        data = Data(store, fit_edges, {"train": sorted(set(fit_edges["plant"]))}, cfg, emb=None, seed=seed)
        ep = np.load(cfg["cache_dir"] / "bioclip_text_plants.npy").astype(np.float32)
        eq = np.load(cfg["cache_dir"] / "bioclip_text_polls.npy").astype(np.float32)
        data.p_dense = np.hstack([data.p_dense, ep])
        data.q_dense = np.hstack([data.q_dense, eq])
        vp_partners = {sp: set(store.idx_polls(g["pollinator"])) for sp, g in val_edges.groupby("plant")}
        vp_wide = {sp: data.wide_plant(store.p2i[sp]) for sp in vp_partners}

        def nn_val(m):
            sc = make_ranker(m, data, args.device)
            return float(np.mean([len(part & set(np.argpartition(-sc(store.p2i[sp], vp_wide[sp]), 10)[:10].tolist())) / len(part)
                                  for sp, part in vp_partners.items()]))

        m, hist = train_model(data, args.device, seed=seed, val_eval=nn_val,
                              log=lambda s: print(f"  [nn] {s}", flush=True))
        nn = (make_ranker(m, data, args.device), data)
        print(f"nn trained, best val {max(hist):.4f} ({time.time() - t0:.0f}s)", flush=True)

    old_partners = {sp: set(store.idx_polls(g["pollinator"])) for sp, g in old.groupby("plant")}
    new_partners = {sp: set(store.idx_polls(g["pollinator"])) for sp, g in new.groupby("plant")}
    names = list(scorers) + (["twotower_bioclip"] if nn else [])
    rec = {n: {"r10": [], "r50": [], "hit": []} for n in names}
    t0 = time.time()
    for i, (sp, part_new) in enumerate(new_partners.items()):
        p = store.p2i[sp]
        blk = {c: (aff.plant(p) if c == "tax" else store.assemble_plant(p, (c,))) for c in comps}
        svec = {n: fn(np.hstack([blk[c] for c in spec])) for n, (spec, fn) in scorers.items()}
        if nn:
            svec["twotower_bioclip"] = nn[0](p, nn[1].wide_plant(p))
        mask = list(old_partners.get(sp, set()))     # already-documented partners are not discoveries
        for n, s in svec.items():
            s = s.copy()
            s[mask] = -np.inf
            top50 = np.argpartition(-s, 50)[:50]
            top50 = top50[np.argsort(-s[top50])]
            t10 = set(top50[:10].tolist())
            rec[n]["r10"].append(len(part_new & t10) / len(part_new))
            rec[n]["r50"].append(len(part_new & set(top50.tolist())) / len(part_new))
            rec[n]["hit"].append(float(len(part_new & t10) > 0))
        if (i + 1) % 300 == 0:
            print(f"  {i + 1}/{len(new_partners)} plants ({time.time() - t0:.0f}s)", flush=True)

    rows = []
    for n in names:
        r10, lo, hi, _ = bootstrap_mean(np.array(rec[n]["r10"]), nboot, seed)
        rows.append({"model": n, "recall@10": r10, "lo": lo, "hi": hi,
                     "recall@50": np.mean(rec[n]["r50"]), "hit@10": np.mean(rec[n]["hit"]),
                     "n_plants": len(rec[n]["r10"])})
        print(f"  {n:<20} R@10 {r10:.4f} [{lo:.4f},{hi:.4f}]  R@50 {np.mean(rec[n]['r50']):.4f}"
              f"  hit@10 {np.mean(rec[n]['hit']):.4f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(cfg["edges"].parent / f"temporal_{args.cutoff}_v1.csv", index=False)
    print("\n" + df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
