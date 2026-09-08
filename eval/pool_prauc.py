"""Pool PR-AUC over seeds for a set of saved score matrices, with a paired bootstrap over plants.

PR-AUC is pooled over all (plant, candidate) pairs, so there is no per-plant value to average, and
averaging score matrices across seeds would be an ensemble (it inflates AP). So every quantity is
computed per seed and averaged over seeds; the paired bootstrap resamples *plants* with the same
draws for every arm and seed, and the arm difference is averaged over seeds before its interval.
Also reports PR-AUC re-expressed at other prevalences: precision at every threshold is recomputed
with negatives down-weighted so that the positive share equals the target prevalence -- the
population version of "PR-AUC under balanced / 3:1 negative sampling", using every negative rather
than a sample. ROC-AUC is prevalence-free and is reported alongside.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from antheia.store import UniverseStore


def ap_at_prevalence(y, s, prev):
    """Average precision if negatives were re-weighted so positives make up `prev` of the data."""
    order = np.argsort(-s, kind="stable"); y = y[order]
    n_pos, n_neg = y.sum(), (1 - y).sum()
    w_neg = (n_pos * (1 - prev) / prev) / n_neg           # total negative weight under the target prevalence
    tp = np.cumsum(y); fp = np.cumsum(1 - y) * w_neg
    prec = tp / (tp + fp)
    return float((prec * y).sum() / n_pos)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stem", default="block_ablation")
    ap.add_argument("--ref", default="full")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--prevalences", nargs="*", type=float, default=[0.25, 0.5])
    args = ap.parse_args()

    files = sorted((ROOT / "results").glob(f"{args.stem}_scores_s*.npz"))
    runs = [np.load(f, allow_pickle=True) for f in files]
    arms = [k for k in runs[0].files if k != "plants"]
    tp = list(runs[0]["plants"])
    store = UniverseStore(curves="modelled")
    split = json.load(open(ROOT / "data/splits/plants_75_10_15.json"))
    e = pd.read_parquet(ROOT / "data/network/edges.parquet")
    e = e[e.plant.isin(store.p2i) & e.pollinator.isin(store.q2i)]
    test = e[e.plant.isin(set(split["val"])) & (e.tier == "A")]
    Y = np.zeros((len(tp), len(store.polls)), np.int8)
    for sp, g in test.groupby("plant"):
        if sp in tp:
            Y[tp.index(sp), store.idx_polls(g.pollinator)] = 1
    prev = Y.mean()
    print(f"{len(runs)} seeds, {len(tp)} plants x {Y.shape[1]} candidates, {int(Y.sum()):,} positives, "
          f"prevalence {prev:.5f} (chance PR-AUC)\n")

    rng = np.random.default_rng(42)
    idx = [rng.integers(0, len(tp), len(tp)) for _ in range(args.boot)]
    plant_of = np.repeat(np.arange(len(tp)), Y.shape[1])

    def boot_ap(y, s):
        """AP under plant-resampling as a weighted AP along one global sort: O(N) per resample."""
        order = np.argsort(-s, kind="stable"); ys = y[order].astype(np.float64); ps = plant_of[order]
        out = np.empty(len(idx))
        for k, ii in enumerate(idx):
            wv = np.bincount(ii, minlength=len(tp)).astype(np.float64)[ps]
            tp_ = np.cumsum(wv * ys); tot = np.cumsum(wv)
            prec = np.divide(tp_, tot, out=np.zeros_like(tp_), where=tot > 0)
            out[k] = (wv * ys * prec).sum() / tp_[-1]
        return out
    boots = {}
    rows = []
    hdr = f"{'arm':<18} {'PR-AUC':>7} {'per-seed':<24} {'ROC':>6} " + " ".join(f"PR@{p:g}" for p in args.prevalences)
    print(hdr)
    y = Y.ravel()
    for a in arms:
        per, rocs, adjs, bs = [], [], [], []
        for r in runs:
            if a not in r.files:
                continue
            s = r[a].astype(np.float32).ravel()
            per.append(average_precision_score(y, s)); rocs.append(roc_auc_score(y, s))
            adjs.append([ap_at_prevalence(y, s, p) for p in args.prevalences]); bs.append(boot_ap(y, s))
        pr, roc, adj = np.mean(per), np.mean(rocs), np.mean(adjs, 0)
        boots[a] = np.mean(bs, 0)
        print(f"{a:<18} {pr:7.4f} {' '.join(f'{v:.4f}' for v in per):<24} {roc:6.3f} " + " ".join(f"{v:6.3f}" for v in adj))
        rows.append(dict(arm=a, pr_auc=pr, pr_auc_seeds=";".join(f"{v:.4f}" for v in per), roc_auc=roc,
                         **{f"pr_at_{p:g}": v for p, v in zip(args.prevalences, adj)}))
    print(f"\npaired bootstrap over plants ({args.boot} resamples), PR-AUC vs {args.ref}:")
    for a in arms:
        if a == args.ref:
            continue
        d = boots[a] - boots[args.ref]
        lo, hi = np.percentile(d, [2.5, 97.5]); p = 2 * min((d > 0).mean(), (d < 0).mean())
        print(f"  {a:<18} {d.mean():+.4f} [{lo:+.4f},{hi:+.4f}] p={min(p, 1):.4f}")
        rows.append(dict(arm=f"{a} - {args.ref}", pr_auc=d.mean(), lo=lo, hi=hi, p=min(p, 1)))
    pd.DataFrame(rows).to_csv(ROOT / f"results/{args.stem}_prauc_pooled.csv", index=False)
    print(f"\n[wrote] results/{args.stem}_prauc_pooled.csv")


if __name__ == "__main__":
    main()
