import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from antheia.config import load_config
from antheia.store import FeatureStore
from antheia.taxonomy import TaxonomyAffinity

# Can each feature family predict WHO documented an interaction (iNaturalist vs curated),
# among edges that are all equally real? High AUC = the feature encodes observation process.
FAMILIES = {
    "vp (pollinator PCA-15)": ["vp"],
    "vf (plant PCA-15)": ["vf"],
    "bioclip_poll (512D)": ["embq"],
    "n_only": ["n"],
    "delta+local": ["delta", "delta_local"],
    "range_sizes": ["frs", "prs"],
    "taxonomy_affinity": ["tax"],
    "all_dense": ["vf", "vp", "embq"],
}


def main():
    ap = argparse.ArgumentParser(description="Provenance probe: do features predict the documentation process?")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    store = FeatureStore(cfg)
    edges = pd.read_parquet(cfg["edges"])
    prov = pd.read_csv(cfg["edges"].parent / "edge_provenance_v1.csv")
    e = edges.merge(prov, on=["plant", "pollinator"], how="inner")
    e = e[e["n"] > 0].copy()
    e["curated"] = (e["n_inat"] < e["n"]).astype(int)
    print(f"edges: {len(e):,} | curated-supported {e['curated'].mean():.1%}")

    aff = TaxonomyAffinity(store, e, cfg["paths"]["globi"])
    Eq = np.load(cfg["cache_dir"] / "bioclip_text_polls.npy").astype(np.float32)
    pi = store.idx_plants(e["plant"])
    qi = store.idx_polls(e["pollinator"])

    def block(c):
        if c == "tax":
            return aff.pairs(pi, qi)
        if c == "embq":
            return Eq[qi].astype(np.float64)
        return store.assemble(pi, qi, (c,))

    cache = {c: block(c) for c in sorted({c for f in FAMILIES.values() for c in f})}
    y = e["curated"].values
    groups = e["plant"].values          # group by plant so the probe must generalize across plants
    gkf = GroupKFold(n_splits=5)

    print(f"\n{'feature family':<26}{'AUC predicting curated-vs-iNat':>32}")
    rows = []
    for name, spec in FAMILIES.items():
        X = np.hstack([cache[c] for c in spec])
        aucs = []
        for tr, te in gkf.split(X, y, groups):
            pipe = Pipeline([("s", StandardScaler()), ("lr", LogisticRegression(max_iter=2000))])
            pipe.fit(X[tr], y[tr])
            aucs.append(roc_auc_score(y[te], pipe.predict_proba(X[te])[:, 1]))
        m, s = np.mean(aucs), np.std(aucs)
        rows.append({"family": name, "auc_mean": m, "auc_std": s, "dims": X.shape[1]})
        print(f"{name:<26}{m:>22.4f} ± {s:.4f}   ({X.shape[1]}D)")

    pd.DataFrame(rows).to_csv(cfg["edges"].parent / "provenance_probe_v1.csv", index=False)
    print("\n0.50 = feature carries no information about documentation process.")
    print("Higher = the feature partly encodes WHO recorded the interaction, not whether it occurs.")


if __name__ == "__main__":
    main()


def order_control():
    """Control: how much provenance signal is just coarse taxon (order/family) composition?"""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    cfg = load_config()
    store = FeatureStore(cfg)
    edges = pd.read_parquet(cfg["edges"])
    prov = pd.read_csv(cfg["edges"].parent / "edge_provenance_v1.csv")
    e = edges.merge(prov, on=["plant", "pollinator"], how="inner")
    e["curated"] = (e["n_inat"] < e["n"]).astype(int)

    g = pd.read_csv(cfg["paths"]["globi"], usecols=["sourceTaxonName", "sourceTaxonOrderName",
                                                    "sourceTaxonFamilyName"]).dropna()
    order = g.groupby("sourceTaxonName")["sourceTaxonOrderName"].agg(lambda s: s.mode().iat[0])
    fam = g.groupby("sourceTaxonName")["sourceTaxonFamilyName"].agg(lambda s: s.mode().iat[0])
    e["order"] = e["pollinator"].map(order).fillna("UNK")
    e["family"] = e["pollinator"].map(fam).fillna("UNK")
    print(f"\norder coverage {(e['order'] != 'UNK').mean():.1%} | family coverage {(e['family'] != 'UNK').mean():.1%}")
    print("\ncurated share by pollinator order (top 6 by volume):")
    t = e.groupby("order").agg(n=("curated", "size"), curated_share=("curated", "mean"))
    print(t.nlargest(6, "n").to_string(float_format=lambda x: f"{x:.3f}"))

    y, groups = e["curated"].values, e["plant"].values
    gkf = GroupKFold(n_splits=5)
    for name, col in [("order one-hot", "order"), ("family one-hot", "family")]:
        X = pd.get_dummies(e[col]).values.astype(float)
        aucs = []
        for tr, te in gkf.split(X, y, groups):
            pipe = Pipeline([("s", StandardScaler()), ("lr", LogisticRegression(max_iter=2000))])
            pipe.fit(X[tr], y[tr])
            aucs.append(roc_auc_score(y[te], pipe.predict_proba(X[te])[:, 1]))
        print(f"{name:<26}{np.mean(aucs):>22.4f} ± {np.std(aucs):.4f}   ({X.shape[1]}D)")
