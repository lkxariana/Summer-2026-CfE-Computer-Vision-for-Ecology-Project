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
from antheia.pairs import plant_split, sample_negatives, save_split
from antheia.store import FeatureStore


def main():
    ap = argparse.ArgumentParser(description="Corrected baselines on the frozen plant split")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    store = FeatureStore(cfg)
    edges = pd.read_parquet(cfg["edges"])
    known_pos = set(zip(edges["plant"], edges["pollinator"]))
    ratio, mode, ks = cfg["negatives"]["ratio"], cfg["negatives"]["mode"], cfg["eval"]["ks"]
    seed, nboot = cfg["eval"]["seed"], cfg["eval"]["bootstrap_n"]

    split = plant_split(edges, seed=cfg["split"]["seed"],
                        test_frac=cfg["split"]["test_frac"], val_frac=cfg["split"]["val_frac"])
    save_split(split, cfg["edges"].parent / "split_v1.json")
    train_pos = edges[edges["plant"].isin(split["train"])]
    test_pos = edges[edges["plant"].isin(split["test"])]
    pool = sorted(set(store.plants) - set(split["test"]) - set(split["val"]))
    print(f"plants train/val/test: {len(split['train'])}/{len(split['val'])}/{len(split['test'])} | "
          f"positives train/test: {len(train_pos):,}/{len(test_pos):,}", flush=True)

    neg_tr = sample_negatives(train_pos, pool, store, ratio, mode, seed, known_pos)
    tr = pd.concat([train_pos[["plant", "pollinator"]].assign(label=1), neg_tr.assign(label=0)], ignore_index=True)
    y_tr = tr["label"].values
    neg_te = sample_negatives(test_pos, split["test"], store, ratio, mode, seed + 1, known_pos)
    te_base = pd.concat([test_pos[["plant", "pollinator"]].assign(label=1), neg_te.assign(label=0)], ignore_index=True)
    partners = {sp: set(store.idx_polls(g["pollinator"])) for sp, g in test_pos.groupby("plant")}

    rows = []

    def evaluate(name, scores_te, scorer):
        te = te_base.assign(score=scores_te)
        pm = pair_metrics(te["label"].values, te["score"].values)
        pr, pr_lo, pr_hi, _ = bootstrap_pr_by_plant(te, 300, seed)
        rk = rank_metrics(scorer, partners, store, ks)
        r10, r10_lo, r10_hi, _ = bootstrap_mean(rk["recall@10"].values, nboot, seed)
        rows.append({"model": name, "pr_auc": pr, "pr_lo": pr_lo, "pr_hi": pr_hi,
                     "roc_auc": pm["roc_auc"], "recall@10": r10, "r10_lo": r10_lo, "r10_hi": r10_hi,
                     "recall@50": rk["recall@50"].mean(), "hit@10": rk["hit@10"].mean()})
        print(f"  {name:<12} PR {pr:.4f} [{pr_lo:.4f},{pr_hi:.4f}]  R@10 {r10:.4f} [{r10_lo:.4f},{r10_hi:.4f}]"
              f"  hit@10 {rk['hit@10'].mean():.4f}", flush=True)

    qi_te = store.idx_polls(te_base["pollinator"])
    evaluate("degree_rank", store.Prs[qi_te], lambda pi: store.Prs)

    for name in models.MODELS:
        t0 = time.time()
        pipe = models.train(store, name, tr, y_tr, seed)
        evaluate(name, models.pair_scores(store, name, pipe, te_base), models.plant_scorer(store, name, pipe))
        print(f"  ({name} trained+evaluated in {time.time() - t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    out = cfg["edges"].parent / "baselines_v1.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved {out}")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
