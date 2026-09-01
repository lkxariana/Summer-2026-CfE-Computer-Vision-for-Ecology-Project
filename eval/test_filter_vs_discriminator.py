import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from antheia.config import load_config
from antheia.pairs import load_split
from antheia.store import FeatureStore

# Hypothesis explaining why phenology helps pooled PR-AUC but not per-plant recall:
# phenological overlap is a FEASIBILITY FILTER (they must be active together) rather than a
# DISCRIMINATOR among co-occurring candidates. Test: how well does Delta separate
#   (a) true partners vs random species            <- the "filter" contrast (pooled-like)
#   (b) true partners vs CO-OCCURRING non-partners <- the "discriminator" contrast (ranking-like)
cfg = load_config()
store = FeatureStore(cfg)
edges = pd.read_parquet(cfg["edges"])
split = load_split(cfg["edges"].parent / "split_v1.json")
test_pos = edges[edges["plant"].isin(split["test"])]
known = set(zip(edges["plant"], edges["pollinator"]))
rng = np.random.default_rng(42)

def delta(p, qs):
    return np.minimum(store.FC[p], store.AC[qs]).sum(1)

def delta_local(p, qs):
    return store.delta_local_pairs(np.full(len(qs), p), qs)

rows = []
plants = list(test_pos.groupby("plant"))
rng.shuffle(plants)
for sp, g in plants[:300]:
    p = store.p2i[sp]
    part = np.array([store.q2i[q] for q in g["pollinator"]])
    cooc = np.flatnonzero(store.N_full[p] > 0)
    cooc_non = np.setdiff1d(cooc, part)
    rand_non = rng.integers(0, len(store.polls), min(400, len(cooc_non) or 400))
    rand_non = np.array([q for q in rand_non if (sp, store.polls[q]) not in known])
    if len(part) < 2 or len(cooc_non) < 20 or len(rand_non) < 20:
        continue
    cooc_non = rng.choice(cooc_non, min(400, len(cooc_non)), replace=False)
    for fname, fn in (("delta", delta), ("delta_local", delta_local), ("N", lambda p, q: store.N_full[p, q].astype(float))):
        dp = fn(p, part)
        for cname, neg in (("vs random", rand_non), ("vs co-occurring", cooc_non)):
            dn = fn(p, neg)
            y = np.r_[np.ones(len(dp)), np.zeros(len(dn))]
            s = np.r_[dp, dn]
            if len(np.unique(s)) > 1:
                rows.append({"plant": sp, "feature": fname, "contrast": cname,
                             "auc": roc_auc_score(y, s)})

df = pd.DataFrame(rows)
piv = df.groupby(["feature", "contrast"])["auc"].agg(["mean", "median", "count"]).round(3)
print(f"per-plant AUC of each feature, {df['plant'].nunique()} test plants\n")
print(piv.to_string())
print("\nfilter minus discriminator (mean AUC):")
for f in df["feature"].unique():
    a = df[(df.feature == f) & (df.contrast == "vs random")]["auc"].mean()
    b = df[(df.feature == f) & (df.contrast == "vs co-occurring")]["auc"].mean()
    print(f"  {f:<12} random {a:.3f}  co-occurring {b:.3f}   drop {a-b:+.3f}")
df.to_csv(cfg["edges"].parent / "filter_vs_discriminator.csv", index=False)
print("\nIf Delta drops to ~0.5 against co-occurring candidates but is high against random ones,")
print("phenology is a feasibility filter, not a partner discriminator — which explains the")
print("pooled-PR-AUC vs recall@k divergence exactly.")
