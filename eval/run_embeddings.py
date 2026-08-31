import argparse
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
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

LINEAR_SPECS = {
    "rank_n": ["n"],
    "rank_n_embsim": ["n", "embsim"],
    "rank_n_tax_embsim": ["n", "tax", "embsim"],
    "rank_vq_n_tax_embsim_ld": ["vq32", "n", "tax", "embsim", "delta_local"],
}
GBM_SPEC = ["vf", "vp", "n", "delta", "delta_local", "frs", "prs", "tax", "embsim", "vq32"]


def main():
    ap = argparse.ArgumentParser(description="BioCLIP text embeddings as ranking features")
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

    Ep = np.load(cfg["cache_dir"] / "bioclip_text_plants.npy").astype(np.float32)   # unit-norm
    Eq = np.load(cfg["cache_dir"] / "bioclip_text_polls.npy").astype(np.float32)
    vq32 = PCA(n_components=32, random_state=seed).fit_transform(Eq).astype(np.float64)
    print(f"embeddings: plants {Ep.shape}, polls {Eq.shape}", flush=True)

    def blocks_pairs(p_idx, q_idx, comp):
        if comp == "tax":
            return aff.pairs(p_idx, q_idx)
        if comp == "embsim":
            return np.einsum("ij,ij->i", Ep[p_idx], Eq[q_idx]).astype(np.float64)[:, None]
        if comp == "vq32":
            return vq32[q_idx]
        return store.assemble(p_idx, q_idx, (comp,))

    def blocks_plant(p, comp):
        if comp == "tax":
            return aff.plant(p)
        if comp == "embsim":
            return (Eq @ Ep[p]).astype(np.float64)[:, None]
        if comp == "vq32":
            return vq32
        return store.assemble_plant(p, (comp,))

    t0 = time.time()
    pi, qp, qn = models.rank_contrast_sets(train_pos, store, 10, seed, known_pos)
    comps = sorted({c for s in LINEAR_SPECS.values() for c in s} | set(GBM_SPEC))
    Xp = {c: blocks_pairs(pi, qp, c) for c in comps}
    Xn = {c: blocks_pairs(pi, qn, c) for c in comps}
    print(f"components assembled ({time.time() - t0:.0f}s)", flush=True)

    scorers = {}
    for name, spec in LINEAR_SPECS.items():
        d = np.hstack([Xp[c] for c in spec]) - np.hstack([Xn[c] for c in spec])
        X, y = np.vstack([d, -d]), np.concatenate([np.ones(len(d)), np.zeros(len(d))])
        pipe = Pipeline([("scale", StandardScaler(with_mean=False)),
                         ("lr", LogisticRegression(max_iter=1000, random_state=seed, fit_intercept=False))])
        pipe.fit(X, y)
        scorers[name] = (spec, lambda M, pp=pipe: pp.decision_function(M))

    pos_rows = np.arange(0, len(pi), 10)
    Xg = np.vstack([np.hstack([Xp[c][pos_rows] for c in GBM_SPEC]), np.hstack([Xn[c] for c in GBM_SPEC])])
    yg = np.concatenate([np.ones(len(pos_rows)), np.zeros(len(pi))])
    gbm = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08, min_samples_leaf=40,
                                         l2_regularization=1.0, random_state=seed)
    gbm.fit(Xg, yg)
    scorers["gbm_full_emb"] = (GBM_SPEC, lambda M: gbm.predict_proba(M)[:, 1])
    print("models trained", flush=True)

    partners = {sp: set(store.idx_polls(g["pollinator"])) for sp, g in test_pos.groupby("plant")}
    per = {name: {"r10": [], "hit": []} for name in scorers}
    hyb = {"hyb_n_embsim": {"r10": [], "hit": []}}
    t0 = time.time()
    for i, (sp, part) in enumerate(partners.items()):
        p = store.p2i[sp]
        blk = {c: blocks_plant(p, c) for c in comps}
        for name, (spec, fn) in scorers.items():
            s = fn(np.hstack([blk[c] for c in spec]))
            t10 = set(np.argpartition(-s, 10)[:10].tolist())
            per[name]["r10"].append(len(part & t10) / len(part))
            per[name]["hit"].append(float(len(part & t10) > 0))
        s = blk["n"][:, 0] + (np.clip(blk["embsim"][:, 0], -1, 1) + 1) / 2.001
        t10 = set(np.argpartition(-s, 10)[:10].tolist())
        hyb["hyb_n_embsim"]["r10"].append(len(part & t10) / len(part))
        hyb["hyb_n_embsim"]["hit"].append(float(len(part & t10) > 0))
        if (i + 1) % 150 == 0:
            print(f"  {i + 1}/{len(partners)} plants ({time.time() - t0:.0f}s)", flush=True)

    rows = []
    for name, m in {**per, **hyb}.items():
        r10, lo, hi, _ = bootstrap_mean(np.array(m["r10"]), nboot, seed)
        rows.append({"model": name, "recall@10": r10, "lo": lo, "hi": hi, "hit@10": np.mean(m["hit"])})
        print(f"  {name:<26} R@10 {r10:.4f} [{lo:.4f},{hi:.4f}]  hit@10 {np.mean(m['hit']):.4f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(cfg["edges"].parent / "embeddings_v1.csv", index=False)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
