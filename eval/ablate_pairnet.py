"""Why does the neural pair ranker trail the gradient-boosted one?

Each row is a complete training run differing from the reference in exactly one design decision, so
the difference in score is attributable to that decision. All runs share seed, split, features and
evaluation; differences are tested by paired bootstrap over the same held-out plants.

Hypotheses under test, in the order they were formed from the data:

  norm=layer   LayerNorm normalises each pair across its own heterogeneous features, so the meaning
               of an affinity value depends on that pair's unrelated magnitudes. Feature skews range
               from 1.0 to 20.8 and scales from 0.03 to 16,481.
  norm=none    Establishes how much of any normalisation effect is scaling per se.
  logq=False   The logQ correction removes popularity from the softmax, but popularity alone reaches
               0.244 nrecall@10 on this data, so the correction may be discarding real signal.
  loss=softmax Isolates the ranking term; predicts good retrieval, poor calibration.
  loss=bce     Isolates the calibration term; predicts the reverse.
  no q-emb     13,124 pollinator embeddings on 132k training edges is 420k parameters against a
               thin supervision signal; tests whether they overfit.
"""
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from antheia.metrics import bootstrap_mean, paired_bootstrap, retrieval_metrics
from antheia.pairnet import PairRanker
from antheia.store import UniverseStore

RUNS = [
    ("reference (feature-standardised)", {}),
    ("norm = LayerNorm",                 dict(norm="layer")),
    ("norm = none",                      dict(norm="none")),
    ("no logQ correction",               dict(logq=False)),
    ("loss = softmax only",              dict(loss="softmax")),
    ("loss = binary only",               dict(loss="bce")),
    ("no pollinator embedding",          dict(use_q_emb=False)),
]


def evaluate(model, store, test_plants, partners):
    S = np.vstack([model.score_plant(store.p2i[sp]) for sp in test_plants])     # [P, Q]
    pp = pd.DataFrame([retrieval_metrics(S[i], partners[sp], ks=(10, 20))
                       for i, sp in enumerate(test_plants)])
    Y = np.zeros(S.shape, np.int8)
    for i, sp in enumerate(test_plants):
        Y[i, list(partners[sp])] = 1
    mean, lo, hi, _ = bootstrap_mean(pp["nrecall@10"].to_numpy(), 1000, 42)
    return pp, {"nrecall@10": mean, "lo": lo, "hi": hi, "recall@10": pp["recall@10"].mean(),
                "map": pp["ap"].mean(), "ndcg@10": pp["ndcg@10"].mean(),
                "pr_auc": average_precision_score(Y.ravel(), S.ravel().astype(np.float64))}


def main():
    store = UniverseStore(curves="modelled")
    split = json.load(open(ROOT / "data/splits/plants_75_10_15.json"))
    e = pd.read_parquet(ROOT / "data/network/edges.parquet")
    e = e[e.plant.isin(store.p2i) & e.pollinator.isin(store.q2i)]
    train = e[e.plant.isin(set(split["train"]))]
    test = e[e.plant.isin(set(split["val"]))]
    test = test[test.tier == "A"]
    tp = sorted(set(test.plant))
    partners = {sp: set(store.idx_polls(g.pollinator)) for sp, g in test.groupby("plant")}
    print(f"[data] train {len(train):,} | val {len(test):,} over {len(tp)} plants", flush=True)

    rows, ref_pp = [], None
    for label, kw in RUNS:
        t0 = time.time()
        model = PairRanker(**kw).fit(train, store)
        pp, r = evaluate(model, store, tp, partners)
        line = (f"  {label:<34} nR@10 {r['nrecall@10']:.4f} [{r['lo']:.4f},{r['hi']:.4f}]  "
                f"MAP {r['map']:.4f}  PR {r['pr_auc']:.4f}")
        if ref_pp is not None:
            d, lo, hi, pv = paired_bootstrap(pp["nrecall@10"].to_numpy(),
                                             ref_pp["nrecall@10"].to_numpy(), 2000, 42)
            r.update(delta=d, delta_lo=lo, delta_hi=hi, delta_p=pv)
            line += f"   vs ref {d:+.4f} [{lo:+.4f},{hi:+.4f}] p={pv:.3f}"
        print(line + f"  ({time.time() - t0:.0f}s)", flush=True)
        rows.append(dict(config=label, **r))
        if ref_pp is None:
            ref_pp = pp
        del model
        import torch; torch.cuda.empty_cache()

    pd.DataFrame(rows).to_csv(ROOT / "results/pairnet_ablation_val_tierA.csv", index=False)
    print("[wrote] results/pairnet_ablation_val_tierA.csv")


if __name__ == "__main__":
    main()
