import argparse
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from antheia import models
from antheia.config import load_config
from antheia.metrics import bootstrap_mean
from antheia.pairs import load_split
from antheia.store import FeatureStore
from antheia.taxonomy import TaxonomyAffinity

SPECS = {
    "rank_n": ["n"],
    "rank_tax": ["tax"],
    "rank_n_tax": ["n", "tax"],
    "rank_n_tax_ldelta": ["n", "tax", "delta_local"],
    "rank_full_tax": ["vp", "n", "delta", "delta_local", "prs", "tax"],
}
HYBRIDS = ["hyb_n_tax", "hyb_n_ldelta", "hyb_n_tax_ldelta"]
CI_MODELS = ["rank_n", "rank_n_tax", "hyb_n_tax", "hyb_n_tax_ldelta"]


def main():
    ap = argparse.ArgumentParser(description="Taxonomy affinity features, N-first hybrids, segment diagnostics")
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
    seen = {sp: aff.genus_seen_in_train(store.p2i[sp]) for sp in test_pos["plant"].unique()}
    print(f"test plants with genus seen in training: {np.mean(list(seen.values())):.1%}", flush=True)

    def blocks_pairs(p_idx, q_idx, comp):
        return aff.pairs(p_idx, q_idx) if comp == "tax" else store.assemble(p_idx, q_idx, (comp,))

    t0 = time.time()
    pi, qp, qn = models.rank_contrast_sets(train_pos, store, 10, seed, known_pos)
    comps = sorted({c for spec in SPECS.values() for c in spec})
    Xp = {c: blocks_pairs(pi, qp, c) for c in comps}
    Xn = {c: blocks_pairs(pi, qn, c) for c in comps}
    print(f"contrasts + components: {len(pi):,} rows ({time.time() - t0:.0f}s)", flush=True)

    pipes = {}
    for name, spec in SPECS.items():
        d = np.hstack([Xp[c] for c in spec]) - np.hstack([Xn[c] for c in spec])
        X, y = np.vstack([d, -d]), np.concatenate([np.ones(len(d)), np.zeros(len(d))])
        pipe = Pipeline([("scale", StandardScaler(with_mean=False)),
                         ("lr", LogisticRegression(max_iter=1000, random_state=seed, fit_intercept=False))])
        pipe.fit(X, y)
        pipes[name] = pipe

    partners = {sp: set(store.idx_polls(g["pollinator"])) for sp, g in test_pos.groupby("plant")}
    per_plant = []
    t0 = time.time()
    for i, (sp, part) in enumerate(partners.items()):
        p = store.p2i[sp]
        blk = {c: (aff.plant(p) if c == "tax" else store.assemble_plant(p, (c,))) for c in comps}
        tax_tie = (blk["tax"][:, 1] + blk["tax"][:, 3]) / 2.001          # genus + family share, < 1
        ld_tie = np.clip(blk["delta_local"][:, 0], 0, 1) / 1.001
        n_vec = blk["n"][:, 0]
        scores = {name: pipes[name].decision_function(np.hstack([blk[c] for c in spec]))
                  for name, spec in SPECS.items()}
        scores["hyb_n_tax"] = n_vec + tax_tie
        scores["hyb_n_ldelta"] = n_vec + ld_tie
        scores["hyb_n_tax_ldelta"] = n_vec + (tax_tie + ld_tie) / 2.001
        row = {"plant": sp, "degree": len(part), "genus_seen": seen[sp]}
        for name, s in scores.items():
            top10 = set(np.argpartition(-s, 10)[:10].tolist())
            row[f"r10_{name}"] = len(part & top10) / len(part)
            row[f"hit_{name}"] = float(len(part & top10) > 0)
        per_plant.append(row)
        if (i + 1) % 150 == 0:
            print(f"  {i + 1}/{len(partners)} plants ({time.time() - t0:.0f}s)", flush=True)

    df = pd.DataFrame(per_plant)
    rows = []
    for name in list(SPECS) + HYBRIDS:
        r10, lo, hi, _ = bootstrap_mean(df[f"r10_{name}"].values, nboot, seed)
        rows.append({"model": name, "recall@10": r10,
                     "lo": lo if name in CI_MODELS else np.nan,
                     "hi": hi if name in CI_MODELS else np.nan,
                     "hit@10": df[f"hit_{name}"].mean(),
                     "r10_genus_seen": df.loc[df["genus_seen"], f"r10_{name}"].mean(),
                     "r10_genus_unseen": df.loc[~df["genus_seen"], f"r10_{name}"].mean()})
        print(f"  {name:<18} R@10 {r10:.4f}  hit@10 {df[f'hit_{name}'].mean():.4f}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(cfg["edges"].parent / "taxonomy_v1.csv", index=False)
    df.to_csv(cfg["edges"].parent / "taxonomy_v1_per_plant.csv", index=False)

    for m in ["rank_n_tax", "hyb_n_tax"]:
        a, b = df[f"hit_{m}"], df["hit_rank_n"]
        print(f"\nwin/loss {m} vs rank_n (hit@10): both {int((a * b).sum())}, "
              f"only_{m} {int((a * (1 - b)).sum())}, only_n {int(((1 - a) * b).sum())}, "
              f"neither {int(((1 - a) * (1 - b)).sum())}")
    for q in range(4):
        sub = df[pd.qcut(df["degree"], 4, labels=False, duplicates="drop") == q]
        print(f"degree Q{q + 1} (n={len(sub)}): rank_n {sub['r10_rank_n'].mean():.3f} | "
              f"n_tax {sub['r10_rank_n_tax'].mean():.3f} | hyb_n_tax {sub['r10_hyb_n_tax'].mean():.3f}")
    print("\n" + out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
