"""Does the sharp taxonomic lookup close the embedding model's retrieval gap?

Ranking metrics depend only on the order of candidates within a plant, so the flatness of the score
distribution -- entropy 9.04 against a 9.48 ceiling -- cannot itself cost recall at ten; a learned
temperature confirmed it, leaving every metric bit-identical. The gap is in the ordering at the top
of the list, and the most likely reason is the one signal the embedding model has to reconstruct
rather than read: the affinity table records that this exact pollinator was seen on this exact plant
genus, which BioCLIP similarity can only approximate smoothly.

Each run is a complete training of the embedding model, differing in what it is given.
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
from antheia.metrics import paired_bootstrap, retrieval_metrics
from antheia.store import UniverseStore

RUNS = [
    ("embeddings only (reference)",        dict()),
    ("+ explicit affinity",                dict(use_affinity=True)),
    ("+ affinity + genus context",         dict(use_affinity=True, use_genus_context=True)),
    ("+ affinity, hard negatives 128",     dict(use_affinity=True, hard_negatives=128)),
    ("+ affinity + genus ctx + hard 128",  dict(use_affinity=True, use_genus_context=True,
                                                hard_negatives=128)),
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
    print(f"[data] val {len(test):,} over {len(tp)} plants  (booster reference: nR@10 0.3259, "
          f"PR 0.1032)", flush=True)

    ref, rows = None, []
    for label, kw in RUNS:
        t0 = time.time()
        base = dict(epochs=25, use_genus_context=False, use_tier_head=False, bce_weight=0.5)
        base.update(kw)
        m = EmbedRanker(**base).fit(train, store)
        S = np.vstack([m.score_plant(store.p2i[sp]) for sp in tp])
        pp = pd.DataFrame([retrieval_metrics(S[i], part[sp], ks=(10, 20)) for i, sp in enumerate(tp)])
        Y = np.zeros(S.shape, np.int8)
        for i, sp in enumerate(tp):
            Y[i, list(part[sp])] = 1
        pr = average_precision_score(Y.ravel(), S.ravel().astype(np.float64))
        line = (f"  {label:<36} nR@10 {pp['nrecall@10'].mean():.4f}  MAP {pp['ap'].mean():.4f}  "
                f"PR {pr:.4f}")
        if ref is not None:
            d, lo, hi, pv = paired_bootstrap(pp["nrecall@10"].to_numpy(), ref["nrecall@10"].to_numpy(),
                                             2000, 42)
            line += f"   vs ref {d:+.4f} [{lo:+.4f},{hi:+.4f}] p={pv:.3f}"
        print(line + f"  ({time.time()-t0:.0f}s)", flush=True)
        rows.append(dict(config=label, nrecall10=pp["nrecall@10"].mean(),
                         map=pp["ap"].mean(), pr_auc=pr))
        if ref is None:
            ref = pp
        del m; torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(ROOT / "results/embednet_hybrid_val.csv", index=False)
    print("[wrote] results/embednet_hybrid_val.csv")


if __name__ == "__main__":
    main()
