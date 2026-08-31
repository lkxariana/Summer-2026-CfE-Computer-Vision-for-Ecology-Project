import argparse
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from antheia import models
from antheia.config import load_config
from antheia.metrics import bootstrap_mean, bootstrap_pr_by_plant, pair_metrics, rank_metrics
from antheia.pairs import kfold_plants, plant_split, sample_negatives
from antheia.store import FeatureStore


def run_once(store, edges, known_pos, train_plants, test_plants, model, ratio, mode, seed, ks):
    """Trains on train-plant pairs, returns pair metrics + ranking metrics on test plants."""
    t0 = time.time()
    train_pos = edges[edges["plant"].isin(train_plants)]
    test_pos = edges[edges["plant"].isin(test_plants)]
    pool = sorted(set(store.plants) - set(test_plants))

    neg_tr = sample_negatives(train_pos, pool, store, ratio, mode, seed, known_pos)
    tr = pd.concat([train_pos[["plant", "pollinator"]].assign(label=1), neg_tr.assign(label=0)], ignore_index=True)
    pipe = models.train(store, model, tr, tr["label"].values, seed)

    neg_te = sample_negatives(test_pos, sorted(test_plants), store, ratio, mode, 10_000 + seed, known_pos)
    te = pd.concat([test_pos[["plant", "pollinator"]].assign(label=1), neg_te.assign(label=0)], ignore_index=True)
    te["score"] = models.pair_scores(store, model, pipe, te)
    pm = pair_metrics(te["label"].values, te["score"].values)

    partners = {sp: set(store.idx_polls(g["pollinator"])) for sp, g in test_pos.groupby("plant")}
    rk = rank_metrics(models.plant_scorer(store, model, pipe), partners, store, ks)
    row = {"n_test_plants": len(partners), "n_test_pos": len(test_pos), **pm,
           "mean_recall@10": rk["recall@10"].mean(), "mean_hit@10": rk["hit@10"].mean(),
           "mean_recall@50": rk["recall@50"].mean(), "runtime_s": round(time.time() - t0, 1)}
    return row, te, rk


def main():
    ap = argparse.ArgumentParser(description="Split-stability analysis: repeated holdouts vs k-fold")
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", default="scalar")
    ap.add_argument("--reps", type=int, default=6)
    args = ap.parse_args()

    cfg = load_config(args.config)
    store = FeatureStore(cfg)
    edges = pd.read_parquet(cfg["edges"])
    known_pos = set(zip(edges["plant"], edges["pollinator"]))
    ratio, mode = cfg["negatives"]["ratio"], cfg["negatives"]["mode"]
    ks = cfg["eval"]["ks"]
    rows = []

    for rep in range(args.reps):
        split = plant_split(edges, seed=rep, test_frac=cfg["split"]["test_frac"])
        row, te, rk = run_once(store, edges, known_pos, split["train"] + split["val"], split["test"],
                               args.model, ratio, mode, rep, ks)
        rows.append({"scheme": "holdout15", "rep": rep, **row})
        print(f"holdout rep {rep}: {row}", flush=True)
        if rep == 0:
            _, lo_r, hi_r, sd_r = bootstrap_mean(rk["recall@10"].values, cfg["eval"]["bootstrap_n"], 0)
            _, lo_p, hi_p, sd_p = bootstrap_pr_by_plant(te, 300, 0)
            boot = {"recall@10_ci": (round(lo_r, 4), round(hi_r, 4)), "recall@10_boot_std": round(sd_r, 4),
                    "pr_ci": (round(lo_p, 4), round(hi_p, 4)), "pr_boot_std": round(sd_p, 4)}
            print(f"rep-0 within-split bootstrap: {boot}", flush=True)

    for k in (3, 5):
        folds = kfold_plants(edges, k, seed=0)
        for i, test_plants in enumerate(folds):
            train_plants = sorted(set().union(*[f for j, f in enumerate(folds) if j != i]))
            row, _, _ = run_once(store, edges, known_pos, train_plants, test_plants,
                                 args.model, ratio, mode, 100 * k + i, ks)
            rows.append({"scheme": f"{k}fold", "rep": i, **row})
            print(f"{k}-fold fold {i}: {row}", flush=True)

    df = pd.DataFrame(rows)
    out = cfg["edges"].parent / "verify_splits.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved {out}\n")
    for scheme, g in df.groupby("scheme"):
        print(f"{scheme:>10}: PR-AUC {g['pr_auc'].mean():.4f} ± {g['pr_auc'].std(ddof=0):.4f} | "
              f"recall@10 {g['mean_recall@10'].mean():.4f} ± {g['mean_recall@10'].std(ddof=0):.4f} | "
              f"hit@10 {g['mean_hit@10'].mean():.4f}")


if __name__ == "__main__":
    main()
