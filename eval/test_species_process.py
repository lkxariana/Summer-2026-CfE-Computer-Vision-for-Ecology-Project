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

# S2 test: do per-species observation-process signatures, estimated from an INDEPENDENT
# 200M-image corpus, predict the provenance of an INTERACTION edge? If yes, the bias model
# can live at the species level (dense) instead of the edge level (vacuous: 4-13 edge bridges).
cfg = load_config()
prof = pd.read_parquet(cfg["cache_dir"] / "obs_profiles.parquet")
edges = pd.read_parquet(cfg["edges"])
prov = pd.read_csv(cfg["edges"].parent / "edge_provenance_v1.csv")
e = edges.merge(prov, on=["plant", "pollinator"], how="inner")
e["curated"] = (e["n_inat"] < e["n"]).astype(int)

bor = [c for c in prof.columns if c.startswith("bor::")]
spec = prof["bor::preserved_specimen"] if "bor::preserved_specimen" in prof else 0
obs = prof["bor::human_observation"] if "bor::human_observation" in prof else 0
prof["specimen_share"] = spec / prof[bor].sum(1).clip(lower=1)
prof["obs_share"] = obs / prof[bor].sum(1).clip(lower=1)
prof["log_images"] = np.log1p(prof["n_images"])
prof["src_diversity"] = (prof[[c for c in prof.columns if c.startswith("src::")]] > 0).sum(1)

cov = {}
for side in ("plant", "pollinator"):
    j = e[side].map(prof["specimen_share"])
    cov[side] = j.notna().mean()
    for f in ("specimen_share", "log_images", "src_diversity", "n_publishers"):
        e[f"{side}_{f}"] = e[side].map(prof[f]) if f in prof else np.nan
print(f"profile coverage: plants {cov['plant']:.1%} of edges, pollinators {cov['pollinator']:.1%}")
d = e.dropna(subset=["plant_specimen_share", "pollinator_specimen_share"]).copy()
print(f"edges usable: {len(d):,} of {len(e):,} | curated rate {d['curated'].mean():.1%}")

FEATS = {
    "composition only (specimen shares)": ["plant_specimen_share", "pollinator_specimen_share"],
    "magnitude only (log image counts)": ["plant_log_images", "pollinator_log_images"],
    "composition + diversity": ["plant_specimen_share", "pollinator_specimen_share",
                                "plant_src_diversity", "pollinator_src_diversity",
                                "plant_n_publishers", "pollinator_n_publishers"],
    "all species-level process features": ["plant_specimen_share", "pollinator_specimen_share",
                                           "plant_log_images", "pollinator_log_images",
                                           "plant_src_diversity", "pollinator_src_diversity",
                                           "plant_n_publishers", "pollinator_n_publishers"],
}
y, groups = d["curated"].values, d["plant"].values
gkf = GroupKFold(n_splits=5)
print(f"\n{'species-level feature set':<40}{'AUC predicting edge provenance':>32}")
for name, cols in FEATS.items():
    X = d[cols].fillna(0).values
    aucs = []
    for tr, te in gkf.split(X, y, groups):
        pipe = Pipeline([("s", StandardScaler()), ("lr", LogisticRegression(max_iter=2000))])
        pipe.fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], pipe.predict_proba(X[te])[:, 1]))
    print(f"{name:<40}{np.mean(aucs):>22.4f} ± {np.std(aucs):.4f}")

print("\nspecies-level identifiability: how many species are seen under BOTH record types?")
both = ((prof["bor::preserved_specimen"] > 0) & (prof["bor::human_observation"] > 0)).sum()
print(f"  species with >=1 specimen AND >=1 human observation: {both:,} / {len(prof):,} ({both / len(prof):.1%})")
print(f"  (compare: only 4.8% of EDGES are documented by >1 source, bridges of 4-13 edges)")
q = prof["specimen_share"]
print(f"\n  specimen_share distribution: {q.quantile([.1, .25, .5, .75, .9]).round(3).to_dict()}")
r = prof["specimen_share"].corr(prof["log_images"])
print(f"  correlation(specimen_share, log_images) = {r:.3f}")
print("  (near 0 would mean composition is independent of abundance; |r|>0.3 means it is NOT a clean"
      " process signal and partly tracks how much a species is recorded at all)")
print("\nNote: TOL-200M contains 10.2M preserved-specimen records alongside 48.6M human observations,"
      " so both record types are well represented; GBIF occurrences remain a possible larger source.")
