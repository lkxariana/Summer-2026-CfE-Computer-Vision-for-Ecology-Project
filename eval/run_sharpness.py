"""Recover top-of-list contrast in the embedding model.

The first version ordered candidates correctly and separated them barely: softmax entropy 9.04
against a 9.48 maximum over 13,124 candidates, top-1 to top-50 gap 0.97 standard deviations where
the boosted ranker reaches 14.35. Recall at ten depends entirely on that separation, pooled PR-AUC
does not, which is exactly the split the two models showed.

Each run reports the contrast statistics alongside the metrics, so a gain in recall can be attributed
to sharper scores rather than assumed.
"""
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from antheia.embednet import EmbedRanker
from antheia.metrics import retrieval_metrics

RUNS = [
    ("baseline (flat)",              dict(learn_temperature=False, bce_weight=0.5)),
    ("+ learned temperature",        dict(bce_weight=0.5)),
    ("+ temp, bce=0.1",              dict(bce_weight=0.1)),
    ("+ temp, bce=0.0",              dict(bce_weight=0.0)),
    ("+ temp, bce=0.1, cand=1024",   dict(bce_weight=0.1, n_cand=1024)),
    ("+ temp, bce=0.1, hard=128",    dict(bce_weight=0.1, hard_negatives=128)),
    ("+ temp, bce=0.1, cand=1024, hard=128", dict(bce_weight=0.1, n_cand=1024, hard_negatives=128)),
]


def contrast(S):
    Z = (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-9)
    p = np.exp(Z - Z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    ent = float(-(p * np.log(p + 1e-12)).sum(1).mean())
    srt = np.sort(Z, 1)[:, ::-1]
    return ent, float((srt[:, 0] - srt[:, 49]).mean())


def main():
    from antheia.store import UniverseStore
    store = UniverseStore(curves="modelled")
    split = json.load(open(ROOT / "data/splits/plants_75_10_15.json"))
    e = pd.read_parquet(ROOT / "data/network/edges.parquet")
    e = e[e.plant.isin(store.p2i) & e.pollinator.isin(store.q2i)]
    train = e[e.plant.isin(set(split["train"]))]
    test = e[e.plant.isin(set(split["val"]))]; test = test[test.tier == "A"]
    tp = sorted(set(test.plant))
    part = {sp: set(store.idx_polls(g.pollinator)) for sp, g in test.groupby("plant")}
    print(f"[data] val {len(test):,} interactions over {len(tp)} plants "
          f"| max entropy ln(13124) = {np.log(len(store.polls)):.3f}", flush=True)

    rows = []
    for label, kw in RUNS:
        t0 = time.time()
        m = EmbedRanker(epochs=25, use_genus_context=False, use_tier_head=False, **kw).fit(train, store)
        S = np.vstack([m.score_plant(store.p2i[sp]) for sp in tp])
        pp = pd.DataFrame([retrieval_metrics(S[i], part[sp], ks=(10, 20)) for i, sp in enumerate(tp)])
        Y = np.zeros(S.shape, np.int8)
        for i, sp in enumerate(tp):
            Y[i, list(part[sp])] = 1
        pr = average_precision_score(Y.ravel(), S.ravel().astype(np.float64))
        ent, gap = contrast(S)
        print(f"  {label:<40} nR@10 {pp['nrecall@10'].mean():.4f}  MAP {pp['ap'].mean():.4f}  "
              f"PR {pr:.4f}  entropy {ent:.3f}  gap {gap:5.2f}  ({time.time()-t0:.0f}s)", flush=True)
        rows.append(dict(config=label, nrecall10=pp["nrecall@10"].mean(), map=pp["ap"].mean(),
                         pr_auc=pr, entropy=ent, top_gap=gap))
        del m; torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(ROOT / "results/embednet_sharpness_val.csv", index=False)
    print("[wrote] results/embednet_sharpness_val.csv")


if __name__ == "__main__":
    main()
