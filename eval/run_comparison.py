"""Run every comparison method on the frozen universe and splits, and emit the results table.

Two objectives, reported side by side because they answer different questions:

  retrieval   for each held-out plant, rank all 13,124 candidate pollinators. This is the task a
              field ecologist has -- given a plant, which visitors should I look for -- and the one
              cold-start features have to carry, since the plant was never seen in training.
  pooled      PR-AUC over every (test plant, candidate) pair at the network's true connectance.
              No negative sampling: all candidates are scored anyway for retrieval, so the pooled
              set is the complete block and the positive rate is the real one.

Per-plant scores are persisted so significance tests and error analysis need no refit.
"""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from antheia.baselines import REGISTRY
from antheia.metrics import bootstrap_mean, retrieval_metrics
from antheia.store import UniverseStore

KS = (10, 20)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--methods", nargs="*", default=list(REGISTRY))
    ap.add_argument("--curves", choices=["observed", "modelled"], default="modelled")
    ap.add_argument("--split", default=ROOT / "data/splits/plants_75_10_15.json")
    ap.add_argument("--part", default="test", choices=["val", "test"])
    ap.add_argument("--out", default=ROOT / "results")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-scores", action="store_true")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    store = UniverseStore(curves=args.curves)
    split = json.load(open(args.split))
    edges = pd.read_parquet(ROOT / "data/network/edges.parquet")
    edges = edges[edges["plant"].isin(store.p2i) & edges["pollinator"].isin(store.q2i)]

    held = set(split[args.part])
    train = edges[edges["plant"].isin(set(split["train"]))]
    test = edges[edges["plant"].isin(held)]
    test_plants = sorted({p for p in test["plant"]})
    partners = {sp: set(store.idx_polls(g["pollinator"])) for sp, g in test.groupby("plant")}
    print(f"[data] train {len(train):,} interactions over {train['plant'].nunique():,} plants | "
          f"{args.part} {len(test):,} over {len(test_plants):,} plants | "
          f"curves={args.curves}", flush=True)

    pi_test = store.idx_plants(test_plants)
    rows, per_plant = [], []
    for name in args.methods:
        t0 = time.time()
        model = REGISTRY[name]()
        model.fit(train, store)
        S = np.empty((len(test_plants), len(store.polls)), dtype=np.float32)
        for r, p in enumerate(pi_test):
            S[r] = model.score_plant(int(p))

        recs = []
        for r, sp in enumerate(test_plants):
            m = retrieval_metrics(S[r], partners[sp], ks=KS)
            m.update(plant=sp, method=name, degree=len(partners[sp]))
            recs.append(m)
        pp = pd.DataFrame(recs)
        per_plant.append(pp)

        y = np.zeros(S.shape, dtype=np.int8)
        for r, sp in enumerate(test_plants):
            y[r, list(partners[sp])] = 1
        yf, sf = y.ravel(), S.ravel().astype(np.float64)
        row = {"method": name, "reference": model.reference, "cold_start": model.cold_start,
               "pr_auc": average_precision_score(yf, sf), "roc_auc": roc_auc_score(yf, sf),
               "connectance": yf.mean(), "seconds": time.time() - t0}
        for k in KS:
            mean, lo, hi, _ = bootstrap_mean(pp[f"recall@{k}"].to_numpy(), args.bootstrap, args.seed)
            row[f"recall@{k}"], row[f"recall@{k}_lo"], row[f"recall@{k}_hi"] = mean, lo, hi
            row[f"ndcg@{k}"] = pp[f"ndcg@{k}"].mean()
        row["median_rank_first"] = pp["rank_first"].median()
        rows.append(row)
        print(f"  {name:<20} R@10 {row['recall@10']:.4f} [{row['recall@10_lo']:.4f},"
              f"{row['recall@10_hi']:.4f}]  nDCG@10 {row['ndcg@10']:.4f}  "
              f"PR {row['pr_auc']:.4f}  ({row['seconds']:.0f}s)", flush=True)
        if args.save_scores:
            np.save(out / f"scores_{name}_{args.part}.npy", S)

    tag = f"{args.part}_{args.curves}"
    pd.concat(per_plant).to_parquet(out / f"per_plant_{tag}.parquet", index=False)
    df = pd.DataFrame(rows).sort_values("recall@10", ascending=False)
    df.to_csv(out / f"comparison_{tag}.csv", index=False)
    print("\n" + df[["method", "recall@10", "recall@20", "ndcg@10", "pr_auc",
                     "median_rank_first"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n[wrote] {out}/comparison_{tag}.csv")


if __name__ == "__main__":
    main()
