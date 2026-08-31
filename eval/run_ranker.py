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


def main():
    ap = argparse.ArgumentParser(description="Stage A: linear pairwise rankers on the frozen plant split")
    ap.add_argument("--config", default=None)
    ap.add_argument("--contrasts", type=int, default=10, help="negative candidates per positive")
    args = ap.parse_args()

    cfg = load_config(args.config)
    store = FeatureStore(cfg)
    edges = pd.read_parquet(cfg["edges"])
    known_pos = set(zip(edges["plant"], edges["pollinator"]))
    seed, nboot, ks = cfg["eval"]["seed"], cfg["eval"]["bootstrap_n"], cfg["eval"]["ks"]

    split = load_split(cfg["edges"].parent / "split_v1.json")
    train_pos = edges[edges["plant"].isin(split["train"])]
    test_pos = edges[edges["plant"].isin(split["test"])]
    print(f"train/test positives: {len(train_pos):,}/{len(test_pos):,}", flush=True)

    # Same pooled test pair set as run_baselines (identical seed) for comparable PR-AUC.
    neg_te = sample_negatives(test_pos, split["test"], store,
                              cfg["negatives"]["ratio"], cfg["negatives"]["mode"], seed + 1, known_pos)
    te_base = pd.concat([test_pos[["plant", "pollinator"]].assign(label=1), neg_te.assign(label=0)], ignore_index=True)
    partners = {sp: set(store.idx_polls(g["pollinator"])) for sp, g in test_pos.groupby("plant")}

    t0 = time.time()
    pi, qi_pos, qi_neg = models.rank_contrast_sets(train_pos, store, args.contrasts, seed, known_pos)
    print(f"contrast sets: {len(pi):,} comparisons ({time.time() - t0:.0f}s)", flush=True)

    rows = []
    for name, spec in models.RANK_SPECS.items():
        t0 = time.time()
        pipe = models.train_pairwise(store, spec, pi, qi_pos, qi_neg, seed)
        scores = pipe.decision_function(store.assemble(store.idx_plants(te_base["plant"]),
                                                       store.idx_polls(te_base["pollinator"]), spec))
        pm = pair_metrics(te_base["label"].values, scores)
        rk = rank_metrics(models.spec_scorer(store, spec, pipe), partners, store, ks)
        r10, r10_lo, r10_hi, _ = bootstrap_mean(rk["recall@10"].values, nboot, seed)
        rows.append({"model": name, "recall@10": r10, "r10_lo": r10_lo, "r10_hi": r10_hi,
                     "recall@50": rk["recall@50"].mean(), "hit@10": rk["hit@10"].mean(),
                     "pr_auc_pooled": pm["pr_auc"]})
        print(f"  {name:<14} R@10 {r10:.4f} [{r10_lo:.4f},{r10_hi:.4f}]  R@50 {rk['recall@50'].mean():.4f}"
              f"  hit@10 {rk['hit@10'].mean():.4f}  pooledPR {pm['pr_auc']:.4f}  ({time.time() - t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    out = cfg["edges"].parent / "ranker_v1.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved {out}")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
