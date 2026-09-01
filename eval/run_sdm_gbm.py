import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from antheia.config import load_config
from antheia.metrics import bootstrap_mean
from antheia.pairs import load_split
from antheia.store import FeatureStore
from antheia.taxonomy import TaxonomyAffinity
from eval.run_sdm import SDMFeatures

# Fairness check for exp 36: the linear ranker may simply be unable to use SDM phenology.
# Trees can express interactions (e.g. "Delta matters only at moderate N"), so if SDM has any
# usable signal a GBM should find it. Also gives the pooled-PR view, where Delta HAS helped before.
cfg = load_config()
store = FeatureStore(cfg)
sdm = SDMFeatures(cfg, store)
seed, nboot = cfg["eval"]["seed"], cfg["eval"]["bootstrap_n"]
edges = pd.read_parquet(cfg["edges"])
prov = pd.read_csv(cfg["edges"].parent / "edge_provenance_v1.csv")
edges = edges.merge(prov, on=["plant", "pollinator"], how="left")
edges["tier1"] = edges["n_inat"].notna() & (edges["n_inat"] < edges["n"])
e = edges[edges["pollinator"].isin(set(sdm.species))].reset_index(drop=True)
split = load_split(cfg["edges"].parent / "split_v1.json")
train_pos, test_pos = e[e["plant"].isin(split["train"])], e[e["plant"].isin(split["test"])]
aff = TaxonomyAffinity(store, train_pos, cfg["paths"]["globi"])
known = set(zip(e["plant"], e["pollinator"]))
rng = np.random.default_rng(seed)

pi, qpos, qneg = [], [], []
for plant, q in zip(train_pos["plant"], train_pos["pollinator"]):
    p, qr = store.p2i[plant], sdm.sp2row[q]
    for _ in range(10):
        r = int(rng.integers(len(sdm.species)))
        if r != qr and (plant, sdm.species[r]) not in known:
            pi.append(p); qpos.append(qr); qneg.append(r)
pi, qpos, qneg = np.array(pi), np.array(qpos), np.array(qneg)


def block(p_idx, q_rows):
    qi = sdm.cand[q_rows]
    n = store.N_full[p_idx, qi].astype(np.float64)[:, None]
    dg = np.minimum(store.FC[p_idx], store.AC[qi]).sum(1)[:, None]
    dl = store.delta_local_pairs(p_idx, qi)[:, None]
    ds = np.minimum(store.FC[p_idx], sdm.A_sdm[q_rows]).sum(1)[:, None]
    db = np.zeros((len(p_idx), 1))
    for p in np.unique(p_idx):
        m = p_idx == p
        db[m, 0] = sdm.delta_bilateral(p, q_rows[m])
    tax = aff.pairs(p_idx, qi)
    frs, prs = store.Frs[p_idx][:, None], store.Prs[qi][:, None]
    return {"base": np.hstack([n, tax, frs, prs]), "gbif": np.hstack([dg, dl]), "sdm": np.hstack([ds, db])}


Bp, Bn = block(pi, qpos), block(pi, qneg)
ARMS = {"base (N+tax+ranges)": ["base"], "base + GBIF temporal": ["base", "gbif"],
        "base + SDM temporal": ["base", "sdm"], "base + both": ["base", "gbif", "sdm"]}
models_ = {}
for name, keys in ARMS.items():
    X = np.vstack([np.hstack([Bp[k] for k in keys]), np.hstack([Bn[k] for k in keys])])
    y = np.concatenate([np.ones(len(pi)), np.zeros(len(pi))])
    m = HistGradientBoostingClassifier(random_state=seed, max_iter=400, learning_rate=0.08,
                                       min_samples_leaf=40, l2_regularization=1.0).fit(X, y)
    models_[name] = (keys, m)
    print(f"trained {name}", flush=True)

allr = np.arange(len(sdm.species))
partners = {sp: {sdm.sp2row[q] for q in g["pollinator"]} for sp, g in test_pos.groupby("plant")}
rec = {n: [] for n in ARMS}
for i, (sp, part) in enumerate(partners.items()):
    p = store.p2i[sp]
    b = block(np.full(len(allr), p), allr)
    for name, (keys, m) in models_.items():
        s = m.predict_proba(np.hstack([b[k] for k in keys]))[:, 1]
        t10 = set(np.argpartition(-s, 10)[:10].tolist())
        rec[name].append(len(part & t10) / len(part))
    if (i + 1) % 150 == 0:
        print(f"  {i+1}/{len(partners)}", flush=True)

print(f"\n{'arm':<28}{'R@10 [95% CI]':>26}")
rows = []
for name in ARMS:
    m_, lo, hi, _ = bootstrap_mean(np.array(rec[name]), nboot, seed)
    rows.append({"arm": name, "r10": m_, "lo": lo, "hi": hi})
    print(f"{name:<28}{m_:>10.4f} [{lo:.3f},{hi:.3f}]")
pd.DataFrame(rows).to_csv(cfg["edges"].parent / "sdm_gbm_v1.csv", index=False)
print("\nIf trees also find no SDM gain, the negative is not a linear-model limitation.")
