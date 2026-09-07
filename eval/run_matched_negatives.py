"""Prevalence-matched negatives: force the conditional spatio-temporal signal.

Ranked alone, per-cell co-activity puts widespread pollinators at the top rather than partners --
AUC 0.665 inside its own top 200. Conditioned on prevalence it is strongly discriminative, adding
0.125 held-out AUC at the top 200 over popularity and range overlap together. Uniform negatives let
the loss be satisfied by predicting prevalence, so that conditional signal is never required.
Matching each negative to its positive's recorded degree closes that route.
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
    ("uniform negatives (reference)", dict()),
    ("50% degree-matched",            dict(matched_negatives=0.5)),
    ("100% degree-matched",           dict(matched_negatives=1.0)),
    ("100% matched + wide affinity",  dict(matched_negatives=1.0, use_wide_affinity=True,
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
    print(f"[data] val {len(test):,} over {len(tp)} plants | booster 0.3199 / PR 0.0998", flush=True)

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
        line = (f"  {label:<32} nR@10 {pp['nrecall@10'].mean():.4f}  MAP {pp['ap'].mean():.4f}  "
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
    pd.DataFrame(rows).to_csv(ROOT / "results/matched_negatives_val.csv", index=False)
    print("[wrote] results/matched_negatives_val.csv")


if __name__ == "__main__":
    main()
