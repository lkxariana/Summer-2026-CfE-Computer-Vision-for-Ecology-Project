"""The wide path: sparse cross-products on a linear term, bypassing the encoder.

Handing the taxonomic affinity count to the pair MLP left retrieval unchanged (p=0.99) and collapsed
pooled PR-AUC from 0.165 to 0.073. Wide & Deep's claim is that such a feature belongs on a linear
path added to the logit, because a deep path over-generalises exactly where the interaction matrix is
sparse and high-rank -- 0.12% connectance here. Each row isolates which part of that path matters.

The plant-side degree term is constant (zero) for every held-out plant, so it can shift a plant's
scores but cannot reorder its candidates; it is included to check it does no harm rather than because
it can help at test time.
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
    ("deep only (reference)",            dict()),
    ("+ wide affinity",                  dict(use_wide_affinity=True)),
    ("+ wide affinity + degree offset",  dict(use_wide_affinity=True, use_degree_offset=True)),
    ("+ wide, hierarchy prompts",        dict(use_wide_affinity=True, text_variant="bioclip2_hier")),
    ("+ wide + genus context",           dict(use_wide_affinity=True, use_genus_context=True)),
    ("+ wide + tier head",               dict(use_wide_affinity=True, use_tier_head=True)),
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
    print(f"[data] val {len(test):,} over {len(tp)} plants "
          f"| booster reference nR@10 0.3259 / PR 0.1032", flush=True)

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
        line = (f"  {label:<34} nR@10 {pp['nrecall@10'].mean():.4f}  MAP {pp['ap'].mean():.4f}  "
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
    pd.DataFrame(rows).to_csv(ROOT / "results/widedeep_val_tierA.csv", index=False)
    print("[wrote] results/widedeep_val_tierA.csv")


if __name__ == "__main__":
    main()
