"""Small capacity/sweep for the neural pair ranker, on the validation split."""
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from antheia.metrics import retrieval_metrics
from antheia.pairnet import PairRanker
from antheia.store import UniverseStore
from sklearn.metrics import average_precision_score

CONFIGS = [
    dict(epochs=20, n_cand=512),
    dict(epochs=40, n_cand=512),
    dict(epochs=40, n_cand=1024, in_batch=128),
    dict(epochs=40, n_cand=512, bce_weight=0.2),
    dict(epochs=40, n_cand=512, bce_weight=1.0),
    dict(epochs=60, n_cand=1024, in_batch=128, lr=1e-3),
]


def main():
    store = UniverseStore(curves="modelled")
    split = json.load(open(ROOT / "data/splits/plants_75_10_15.json"))
    e = pd.read_parquet(ROOT / "data/network/edges.parquet")
    e = e[e.plant.isin(store.p2i) & e.pollinator.isin(store.q2i)]
    train = e[e.plant.isin(set(split["train"]))]
    test = e[e.plant.isin(set(split["val"]))]; test = test[test.tier == "A"]
    tp = sorted(set(test.plant))
    part = {sp: set(store.idx_polls(g.pollinator)) for sp, g in test.groupby("plant")}
    rows = []
    for cfg in CONFIGS:
        t0 = time.time()
        m = PairRanker(**cfg).fit(train, store)
        S = np.vstack([m.score_plant(store.p2i[sp]) for sp in tp])
        pp = pd.DataFrame([retrieval_metrics(S[i], part[sp], ks=(10, 20)) for i, sp in enumerate(tp)])
        Y = np.zeros(S.shape, np.int8)
        for i, sp in enumerate(tp):
            Y[i, list(part[sp])] = 1
        r = dict(cfg, nrecall10=pp["nrecall@10"].mean(), map=pp["ap"].mean(),
                 pr_auc=average_precision_score(Y.ravel(), S.ravel().astype(np.float64)),
                 ndcg10=pp["ndcg@10"].mean(), secs=time.time() - t0)
        rows.append(r)
        print("  " + "  ".join(f"{k}={v}" for k, v in cfg.items()) +
              f"  -> nR@10 {r['nrecall10']:.4f}  MAP {r['map']:.4f}  PR {r['pr_auc']:.4f}"
              f"  ({r['secs']:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv(ROOT / "results/pairnet_sweep_val.csv", index=False)
    print("[wrote] results/pairnet_sweep_val.csv")


if __name__ == "__main__":
    main()
