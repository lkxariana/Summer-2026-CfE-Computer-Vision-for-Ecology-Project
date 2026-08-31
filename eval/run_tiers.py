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
from antheia import models
from antheia.config import load_config
from antheia.metrics import bootstrap_mean
from antheia.pairs import load_split
from antheia.store import FeatureStore

RANKERS = ["rank_n", "rank_full", "rank_scalar_local", "rank_full_local"]


def fit_rankers(store, train_pos, known_pos, seed):
    pi, qp, qn = models.rank_contrast_sets(train_pos, store, 10, seed, known_pos)
    comps = sorted({c for r in RANKERS for c in models.RANK_SPECS[r]})
    Xp = {c: store.assemble(pi, qp, (c,)) for c in comps}
    Xn = {c: store.assemble(pi, qn, (c,)) for c in comps}
    pipes = {}
    for name in RANKERS:
        spec = models.RANK_SPECS[name]
        d = np.hstack([Xp[c] for c in spec]) - np.hstack([Xn[c] for c in spec])
        X, y = np.vstack([d, -d]), np.concatenate([np.ones(len(d)), np.zeros(len(d))])
        pipe = Pipeline([("scale", StandardScaler(with_mean=False)),
                         ("lr", LogisticRegression(max_iter=1000, random_state=seed, fit_intercept=False))])
        pipe.fit(X, y)
        pipes[name] = pipe
    return pipes


def main():
    ap = argparse.ArgumentParser(description="Evidence-tier 2x2: train {all, tier1} x eval {all, tier1} on the frozen split")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    store = FeatureStore(cfg)
    edges = pd.read_parquet(cfg["edges"])
    prov = pd.read_csv(cfg["edges"].parent / "edge_provenance_v1.csv").rename(
        columns={"plant": "plant", "poll": "pollinator"})
    edges = edges.merge(prov, on=["plant", "pollinator"], how="left")
    edges["tier1"] = edges["n_inat"].notna() & (edges["n_inat"] < edges["n"])
    known_pos = set(zip(edges["plant"], edges["pollinator"]))
    seed, nboot = cfg["eval"]["seed"], cfg["eval"]["bootstrap_n"]
    split = load_split(cfg["edges"].parent / "split_v1.json")

    train_all = edges[edges["plant"].isin(split["train"])]
    train_t1 = train_all[train_all["tier1"]]
    test_pos = edges[edges["plant"].isin(split["test"])]
    test_t1 = test_pos[test_pos["tier1"]]
    print(f"provenance join: {edges['n_inat'].notna().mean():.1%} of edges matched | tier1 edges: {edges['tier1'].sum():,}")
    print(f"train edges all/t1: {len(train_all):,}/{len(train_t1):,} | test edges all/t1: {len(test_pos):,}/{len(test_t1):,}")
    print(f"test plants all/with-t1-partner: {test_pos['plant'].nunique()}/{test_t1['plant'].nunique()}", flush=True)

    partners_all = {sp: set(store.idx_polls(g["pollinator"])) for sp, g in test_pos.groupby("plant")}
    partners_t1 = {sp: set(store.idx_polls(g["pollinator"])) for sp, g in test_t1.groupby("plant")}

    print("fitting rankers on both training sets...", flush=True)
    pipes = {"train_all": fit_rankers(store, train_all, known_pos, seed),
             "train_t1": fit_rankers(store, train_t1, known_pos, seed)}

    comps = sorted({c for r in RANKERS for c in models.RANK_SPECS[r]})
    per_plant = {}
    recalls = {(tr, name, ev): [] for tr in pipes for name in RANKERS for ev in ("all", "t1")}
    t0 = time.time()
    for i, (sp, part_all) in enumerate(partners_all.items()):
        p = store.p2i[sp]
        blocks = {c: store.assemble_plant(p, (c,)) for c in comps}
        part_t1 = partners_t1.get(sp, set())
        for tr, ps in pipes.items():
            for name in RANKERS:
                spec = models.RANK_SPECS[name]
                s = ps[name].decision_function(np.hstack([blocks[c] for c in spec]))
                top10_all = set(np.argpartition(-s, 10)[:10].tolist())
                recalls[(tr, name, "all")].append(len(part_all & top10_all) / len(part_all))
                if part_t1:
                    sm = s.copy()
                    sm[list(part_all - part_t1)] = -np.inf
                    top10 = set(np.argpartition(-sm, 10)[:10].tolist())
                    recalls[(tr, name, "t1")].append(len(part_t1 & top10) / len(part_t1))
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(partners_all)} plants ({time.time() - t0:.0f}s)", flush=True)

    rows = []
    for (tr, name, ev), vals in recalls.items():
        r10, lo, hi, _ = bootstrap_mean(np.array(vals), nboot, seed)
        rows.append({"training": tr, "model": name, "eval": ev, "n_plants": len(vals),
                     "recall@10": r10, "lo": lo, "hi": hi})
    df = pd.DataFrame(rows).sort_values(["eval", "training", "model"])
    out = cfg["edges"].parent / "tiers_v1.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved {out}")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
