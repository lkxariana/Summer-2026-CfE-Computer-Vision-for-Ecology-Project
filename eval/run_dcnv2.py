"""DCN-V2 cross layers against hand-chosen crosses.

DCN-V2's critique of Wide & Deep is that choosing the cross features "falls back to the feature
engineering problem". These runs put learned bounded-degree crossing beside the hand-built back-off
crosses and together, to separate whether the gain is from crossing at all or from choosing well.
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
    ("deep only (reference)",        dict()),
    ("DCN-V2, 2 cross layers",       dict(cross_layers=2)),
    ("DCN-V2, 4 cross layers",       dict(cross_layers=4)),
    ("back-off crosses only",        dict(backoff_crosses=True, use_degree_offset=True)),
    ("DCN-V2 (3) + back-off",        dict(cross_layers=3, backoff_crosses=True,
                                          use_degree_offset=True)),
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
    print(f"[data] val {len(test):,} over {len(tp)} plants | booster 0.3259 / PR 0.1032", flush=True)

    ref, rows = None, []
    for label, kw in RUNS:
        base = dict(epochs=25, use_genus_context=False, use_tier_head=False)
        base.update(kw)
        t0 = time.time()
        m = EmbedRanker(**base).fit(train, store)
        S = np.vstack([m.score_plant(store.p2i[sp]) for sp in tp])
        pp = pd.DataFrame([retrieval_metrics(S[i], part[sp], ks=(10, 20)) for i, sp in enumerate(tp)])
        Y = np.zeros(S.shape, np.int8)
        for i, sp in enumerate(tp):
            Y[i, list(part[sp])] = 1
        pr = average_precision_score(Y.ravel(), S.ravel().astype(np.float64))
        line = (f"  {label:<28} nR@10 {pp['nrecall@10'].mean():.4f}  MAP {pp['ap'].mean():.4f}  "
                f"PR {pr:.4f}")
        if ref is not None:
            d, lo, hi, pv = paired_bootstrap(pp["nrecall@10"].to_numpy(), ref["nrecall@10"].to_numpy(),
                                             2000, 42)
            line += f"   vs ref {d:+.4f} [{lo:+.4f},{hi:+.4f}] p={pv:.3f}"
        print(line + f"  ({time.time()-t0:.0f}s)", flush=True)
        rows.append(dict(config=label, nrecall10=pp["nrecall@10"].mean(), map=pp["ap"].mean(), pr_auc=pr))
        if ref is None:
            ref = pp
        del m; torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(ROOT / "results/dcnv2_val_tierA.csv", index=False)
    print("[wrote] results/dcnv2_val_tierA.csv")


if __name__ == "__main__":
    main()
