import argparse
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from antheia.config import load_config
from antheia.metrics import bootstrap_mean
from antheia.pairs import load_split
from antheia.store import FeatureStore
from antheia.twotower import Data, make_ranker, train_model


def main():
    ap = argparse.ArgumentParser(description="Stage B: wide&deep two-tower neural ranker")
    ap.add_argument("--config", default=None)
    ap.add_argument("--emb", default="none", choices=["none", "bioclip", "bioclip2"])
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tag = args.tag or f"twotower_{args.emb}"

    cfg = load_config(args.config)
    store = FeatureStore(cfg)
    edges = pd.read_parquet(cfg["edges"])
    prov = pd.read_csv(cfg["edges"].parent / "edge_provenance_v1.csv")
    edges = edges.merge(prov, on=["plant", "pollinator"], how="left")
    edges["tier1"] = edges["n_inat"].notna() & (edges["n_inat"] < edges["n"])
    split = load_split(cfg["edges"].parent / "split_v1.json")
    nboot, seed = cfg["eval"]["bootstrap_n"], args.seed

    emb = None
    if args.emb != "none":
        prefix = {"bioclip": "bioclip_text", "bioclip2": "bioclip2_text"}[args.emb]
        emb = (np.load(cfg["cache_dir"] / f"{prefix}_plants.npy"),
               np.load(cfg["cache_dir"] / f"{prefix}_polls.npy"))
        print(f"tower emb: {prefix} {emb[0].shape[1]}D", flush=True)

    t0 = time.time()
    data = Data(store, edges, split, cfg, emb=emb, seed=seed)
    print(f"data prepared ({time.time() - t0:.0f}s)", flush=True)

    def partners_of(plants_subset, df):
        sub = df[df["plant"].isin(plants_subset)]
        return {sp: set(store.idx_polls(g["pollinator"])) for sp, g in sub.groupby("plant")}

    val_partners = partners_of(split["val"], edges)
    val_wide = {sp: data.wide_plant(store.p2i[sp]) for sp in val_partners}
    print(f"val wide cache built ({time.time() - t0:.0f}s)", flush=True)

    def val_eval(m):
        scorer = make_ranker(m, data, args.device)
        r10s = []
        for sp, part in val_partners.items():
            s = scorer(store.p2i[sp], val_wide[sp])
            t10 = set(np.argpartition(-s, 10)[:10].tolist())
            r10s.append(len(part & t10) / len(part))
        return float(np.mean(r10s))

    m, hist = train_model(data, args.device, seed=seed, val_eval=val_eval,
                          log=lambda s: print(s, flush=True))

    test_partners = partners_of(split["test"], edges)
    t1 = edges[edges["tier1"]]
    test_t1_partners = partners_of(split["test"], t1)
    scorer = make_ranker(m, data, args.device)
    r_all, hit_all, r_t1 = [], [], []
    for sp, part in test_partners.items():
        w = data.wide_plant(store.p2i[sp])
        s = scorer(store.p2i[sp], w)
        t10 = set(np.argpartition(-s, 10)[:10].tolist())
        r_all.append(len(part & t10) / len(part))
        hit_all.append(float(len(part & t10) > 0))
        pt1 = test_t1_partners.get(sp, set())
        if pt1:
            sm = s.copy()
            sm[list(part - pt1)] = -np.inf
            t10m = set(np.argpartition(-sm, 10)[:10].tolist())
            r_t1.append(len(pt1 & t10m) / len(pt1))
    r10, lo, hi, _ = bootstrap_mean(np.array(r_all), nboot, seed)
    r10t, lot, hit_, _ = bootstrap_mean(np.array(r_t1), nboot, seed)
    print(f"\nFINAL {tag}: test R@10 {r10:.4f} [{lo:.4f},{hi:.4f}] hit@10 {np.mean(hit_all):.4f} | "
          f"Tier-1 R@10 {r10t:.4f} [{lot:.4f},{hit_:.4f}] (n={len(r_t1)})")
    pd.DataFrame([{"model": tag, "recall@10": r10, "lo": lo, "hi": hi, "hit@10": np.mean(hit_all),
                   "t1_recall@10": r10t, "t1_lo": lot, "t1_hi": hit_, "n_t1_plants": len(r_t1),
                   "best_val": max(hist) if hist else np.nan}]).to_csv(
        cfg["edges"].parent / f"{tag}.csv", index=False)


if __name__ == "__main__":
    main()
