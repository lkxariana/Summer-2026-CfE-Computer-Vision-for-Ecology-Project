import argparse
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from antheia.config import load_config
from antheia.metrics import bootstrap_mean
from antheia.pairs import load_split
from antheia.store import FeatureStore
from antheia.twotower import Data, make_ranker, train_model
from eval.run_biashead import process_features  # noqa: E402

# Second, structurally DIFFERENT remedy (exp 32 tested only an additive bias head):
# domain-adaptation importance weighting. Estimate P(curated | species-level process features)
# and reweight training positives by P(curated)/P(iNat) so the effective training distribution
# matches the curated documentation process. If the transfer gap closes, S1 is wrong.


def main():
    ap = argparse.ArgumentParser(description="Importance-reweighting remedy toward the curated process")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    cfg = load_config()
    store = FeatureStore(cfg)
    edges = pd.read_parquet(cfg["edges"])
    prov = pd.read_csv(cfg["edges"].parent / "edge_provenance_v1.csv")
    edges = edges.merge(prov, on=["plant", "pollinator"], how="left")
    edges["tier1"] = edges["n_inat"].notna() & (edges["n_inat"] < edges["n"])
    split = load_split(cfg["edges"].parent / "split_v1.json")
    seed, nboot = args.seed, cfg["eval"]["bootstrap_n"]
    P, Q = process_features(cfg, store)

    emb = (np.load(cfg["cache_dir"] / "bioclip_text_plants.npy"),
           np.load(cfg["cache_dir"] / "bioclip_text_polls.npy"))
    g = pd.read_csv(cfg["paths"]["globi"], usecols=["sourceTaxonName", "sourceTaxonOrderName"]).dropna()
    order = g.groupby("sourceTaxonName")["sourceTaxonOrderName"].agg(lambda s: s.mode().iat[0])
    hym = {store.q2i[s] for s in store.polls if order.get(s, "UNK") == "Hymenoptera"}

    data = Data(store, edges, split, cfg, emb=emb, seed=seed)
    train_pos = edges[edges["plant"].isin(split["train"])].reset_index(drop=True)
    X = np.hstack([P[data.pos_pi], Q[data.pos_qi]])
    y = train_pos["tier1"].fillna(False).astype(int).to_numpy()
    clf = Pipeline([("s", StandardScaler()), ("lr", LogisticRegression(max_iter=2000))]).fit(X, y)
    p_cur = np.clip(clf.predict_proba(X)[:, 1], 0.02, 0.98)
    w = p_cur / (1 - p_cur)
    w = np.clip(w / w.mean(), 0.1, 10.0)
    print(f"importance weights: mean {w.mean():.2f} median {np.median(w):.2f} "
          f"p90 {np.percentile(w,90):.2f} max {w.max():.2f}", flush=True)

    val_p = {sp: set(store.idx_polls(gg["pollinator"]))
             for sp, gg in edges[edges["plant"].isin(split["val"])].groupby("plant")}
    val_w = {sp: data.wide_plant(store.p2i[sp]) for sp in val_p}

    def val_eval(m):
        sc = make_ranker(m, data, args.device)
        return float(np.mean([len(pp & set(np.argpartition(-sc(store.p2i[sp], val_w[sp]), 10)[:10].tolist())) / len(pp)
                              for sp, pp in val_p.items()]))

    res = {}
    for tag, ww in [("unweighted", None), ("reweighted_to_curated", w)]:
        t0 = time.time()
        m, _ = train_model(data, args.device, seed=seed, val_eval=val_eval, weights=ww,
                           log=lambda s: print(f"  [{tag}] {s}", flush=True))
        sc = make_ranker(m, data, args.device)
        test = edges[edges["plant"].isin(split["test"])]
        r_all, r_cur, r_inat = [], [], []
        for sp, gg in test.groupby("plant"):
            s_ = sc(store.p2i[sp], data.wide_plant(store.p2i[sp]))
            recs = gg.to_dict("records")
            allp = {store.q2i[r["pollinator"]] for r in recs}
            r_all.append(len(allp & set(np.argpartition(-s_, 10)[:10].tolist())) / len(allp))
            for bucket, want in ((r_cur, True), (r_inat, False)):
                tg = {store.q2i[r["pollinator"]] for r in recs
                      if bool(r["tier1"]) == want and store.q2i[r["pollinator"]] in hym}
                if not tg:
                    continue
                sm = s_.copy()
                blk = np.ones(len(sm), bool); blk[list(hym)] = False
                sm[blk] = -np.inf
                sm[list(allp - tg)] = -np.inf
                bucket.append(len(tg & set(np.argpartition(-sm, 10)[:10].tolist())) / len(tg))
        a, lo, hi, _ = bootstrap_mean(np.array(r_all), nboot, seed)
        res[tag] = (a, lo, hi, np.mean(r_cur), np.mean(r_inat), np.mean(r_cur) / np.mean(r_inat))
        print(f"  [{tag}] R@10 {a:.4f} [{lo:.4f},{hi:.4f}] | Hym cur {np.mean(r_cur):.4f} "
              f"vs iNat {np.mean(r_inat):.4f} ratio {res[tag][5]:.2f} ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n{'model':<24}{'R@10':>10}{'ratio':>8}")
    for k, v in res.items():
        print(f"{k:<24}{v[0]:>10.4f}{v[5]:>8.2f}")
    d = abs(1 - res["reweighted_to_curated"][5]) - abs(1 - res["unweighted"][5])
    print(f"\ngap-to-parity change: {d:+.3f}  (negative = remedy WORKED -> S1 wrong)")
    pd.DataFrame([{"model": k, "r10": v[0], "lo": v[1], "hi": v[2], "hym_cur": v[3],
                   "hym_inat": v[4], "ratio": v[5]} for k, v in res.items()]).to_csv(
        cfg["edges"].parent / "reweight_v1.csv", index=False)


if __name__ == "__main__":
    main()
