import argparse
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from antheia import models
from antheia.config import load_config
from antheia.metrics import bootstrap_mean
from antheia.pairs import load_split
from antheia.store import FeatureStore
from antheia.taxonomy import TaxonomyAffinity

GEO = ["vf", "vp", "n", "delta", "delta_local", "frs", "prs"]
VARIANTS = {"gbm_geo": GEO, "gbm_full": GEO + ["tax"]}


def main():
    ap = argparse.ArgumentParser(description="Gradient-boosted contextual ranker (tests the linear ceiling)")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    store = FeatureStore(cfg)
    edges = pd.read_parquet(cfg["edges"])
    known_pos = set(zip(edges["plant"], edges["pollinator"]))
    seed, nboot = cfg["eval"]["seed"], cfg["eval"]["bootstrap_n"]
    split = load_split(cfg["edges"].parent / "split_v1.json")
    train_pos = edges[edges["plant"].isin(split["train"])]
    test_pos = edges[edges["plant"].isin(split["test"])]
    aff = TaxonomyAffinity(store, train_pos, cfg["paths"]["globi"])

    def blocks_pairs(p_idx, q_idx, comp):
        return aff.pairs(p_idx, q_idx) if comp == "tax" else store.assemble(p_idx, q_idx, (comp,))

    t0 = time.time()
    pi, qp, qn = models.rank_contrast_sets(train_pos, store, 10, seed, known_pos)
    pos_rows = np.arange(0, len(pi), 10)  # each positive appears 10x in the contrast arrays
    comps = sorted({c for spec in VARIANTS.values() for c in spec})
    Xpos = {c: blocks_pairs(pi[pos_rows], qp[pos_rows], c) for c in comps}
    Xneg = {c: blocks_pairs(pi, qn, c) for c in comps}
    y = np.concatenate([np.ones(len(pos_rows)), np.zeros(len(pi))])
    print(f"train rows: {len(y):,} ({time.time() - t0:.0f}s to assemble)", flush=True)

    partners = {sp: set(store.idx_polls(g["pollinator"])) for sp, g in test_pos.groupby("plant")}
    rows = []
    for name, spec in VARIANTS.items():
        t0 = time.time()
        X = np.vstack([np.hstack([Xpos[c] for c in spec]), np.hstack([Xneg[c] for c in spec])])
        clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08, min_samples_leaf=40,
                                             l2_regularization=1.0, random_state=seed)
        clf.fit(X, y)
        r10s, r50s, hits = [], [], []
        for sp, part in partners.items():
            p = store.p2i[sp]
            Xq = np.hstack([aff.plant(p) if c == "tax" else store.assemble_plant(p, (c,)) for c in spec])
            s = clf.predict_proba(Xq)[:, 1]
            top50 = np.argpartition(-s, 50)[:50]
            top50 = top50[np.argsort(-s[top50])]
            t10 = set(top50[:10].tolist())
            r10s.append(len(part & t10) / len(part))
            r50s.append(len(part & set(top50.tolist())) / len(part))
            hits.append(float(len(part & t10) > 0))
        r10, lo, hi, _ = bootstrap_mean(np.array(r10s), nboot, seed)
        rows.append({"model": name, "recall@10": r10, "lo": lo, "hi": hi,
                     "recall@50": np.mean(r50s), "hit@10": np.mean(hits)})
        print(f"  {name:<10} R@10 {r10:.4f} [{lo:.4f},{hi:.4f}]  R@50 {np.mean(r50s):.4f}"
              f"  hit@10 {np.mean(hits):.4f}  ({time.time() - t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(cfg["edges"].parent / "gbm_v1.csv", index=False)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
