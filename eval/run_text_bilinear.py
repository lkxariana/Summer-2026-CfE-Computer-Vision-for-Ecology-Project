"""Learned plant-to-pollinator compatibility in BioCLIP-2 species space.

Both hand-built reductions of these embeddings failed -- a prototype cosine, and kernel smoothing of
the affinity table -- and both failed in a way that bounded the reduction rather than the
representation: the smoothing degraded the model as much with a permuted neighbour graph as with the
real one. Here the projections are learned, so the model chooses what in species space predicts
interaction.

Control: species-permuted embeddings, keeping dimensionality and the parameter count and destroying
the correspondence between a taxon and its vector.
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
from antheia.metrics import paired_bootstrap, retrieval_metrics
from antheia.pairnet import PairRanker
from antheia.store import UniverseStore

TEXT = Path("/scratch/cher/antheia-data/text_embeddings")


def main():
    store = UniverseStore(curves="modelled")
    split = json.load(open(ROOT / "data/splits/plants_75_10_15.json"))
    e = pd.read_parquet(ROOT / "data/network/edges.parquet")
    e = e[e.plant.isin(store.p2i) & e.pollinator.isin(store.q2i)]
    train = e[e.plant.isin(set(split["train"]))]
    test = e[e.plant.isin(set(split["val"]))]; test = test[test.tier == "A"]
    tp = sorted(set(test.plant))
    part = {sp: set(store.idx_polls(g.pollinator)) for sp, g in test.groupby("plant")}

    runs = [("text_rank=0 (reference)", dict(text_rank=0)),
            ("text_rank=32", dict(text_rank=32)),
            ("text_rank=64", dict(text_rank=64)),
            ("text_rank=128", dict(text_rank=128)),
            ("text_rank=64, wd=1e-2", dict(text_rank=64, weight_decay=1e-2))]
    ref, rows = None, []
    for label, kw in runs:
        t0 = time.time()
        m = PairRanker(epochs=20, **kw).fit(train, store)
        S = np.vstack([m.score_plant(store.p2i[sp]) for sp in tp])
        pp = pd.DataFrame([retrieval_metrics(S[i], part[sp], ks=(10, 20)) for i, sp in enumerate(tp)])
        Y = np.zeros(S.shape, np.int8)
        for i, sp in enumerate(tp):
            Y[i, list(part[sp])] = 1
        pr = average_precision_score(Y.ravel(), S.ravel().astype(np.float64))
        line = (f"  {label:<26} nR@10 {pp['nrecall@10'].mean():.4f}  "
                f"MAP {pp['ap'].mean():.4f}  PR {pr:.4f}")
        if ref is not None:
            d, lo, hi, pv = paired_bootstrap(pp["nrecall@10"].to_numpy(), ref["nrecall@10"].to_numpy(), 2000, 42)
            line += f"   vs ref {d:+.4f} [{lo:+.4f},{hi:+.4f}] p={pv:.3f}"
        print(line + f"  ({time.time() - t0:.0f}s)", flush=True)
        rows.append(dict(config=label, nrecall10=pp["nrecall@10"].mean(), map=pp["ap"].mean(), pr_auc=pr))
        if ref is None:
            ref = pp
        del m; torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(ROOT / "results/text_bilinear_val.csv", index=False)
    print("[wrote] results/text_bilinear_val.csv")


if __name__ == "__main__":
    main()
