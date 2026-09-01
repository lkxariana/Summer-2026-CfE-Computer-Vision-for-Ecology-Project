import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from antheia.config import load_config
from antheia.metrics import bootstrap_pr_by_plant
from antheia.pairs import load_split, sample_negatives
from antheia.store import FeatureStore
from antheia.taxonomy import TaxonomyAffinity
from eval.run_sdm import SDMFeatures

# Every SDM test so far used RANKING (recall@10). The project's original claim was about
# PR-AUC/ROC-AUC on a pooled pair set, and Delta has previously helped there (0.652 -> 0.657)
# while failing at top-k. So test the original claim on its own metric, corrected benchmark.
cfg = load_config()
store = FeatureStore(cfg)
sdm = SDMFeatures(cfg, store)
seed = cfg["eval"]["seed"]
edges = pd.read_parquet(cfg["edges"])
e = edges[edges["pollinator"].isin(set(sdm.species))].reset_index(drop=True)
split = load_split(cfg["edges"].parent / "split_v1.json")
train_pos, test_pos = e[e["plant"].isin(split["train"])], e[e["plant"].isin(split["test"])]
aff = TaxonomyAffinity(store, train_pos, cfg["paths"]["globi"])
known = set(zip(edges["plant"], edges["pollinator"]))
print(f"pairs: train_pos {len(train_pos):,} test_pos {len(test_pos):,}", flush=True)

sdm_pool = sorted(set(sdm.species))


def negs(pos_df, plant_pool, ratio, s):
    rng = np.random.default_rng(s)
    out, seen = [], set()
    plants = np.asarray(plant_pool)
    while len(out) < ratio * len(pos_df):
        pl = rng.choice(plants, 4096)
        po = rng.choice(sdm_pool, 4096)
        for a, b in zip(pl, po):
            if (a, b) not in known and (a, b) not in seen:
                seen.add((a, b)); out.append((a, b))
                if len(out) >= ratio * len(pos_df):
                    break
    return pd.DataFrame(out, columns=["plant", "pollinator"])


tr = pd.concat([train_pos[["plant", "pollinator"]].assign(label=1),
                negs(train_pos, split["train"], 10, seed).assign(label=0)], ignore_index=True)
te = pd.concat([test_pos[["plant", "pollinator"]].assign(label=1),
                negs(test_pos, split["test"], 10, seed + 1).assign(label=0)], ignore_index=True)
print(f"pooled sets: train {len(tr):,} test {len(te):,}", flush=True)


def block(df):
    p_idx = store.idx_plants(df["plant"]); qi = store.idx_polls(df["pollinator"])
    q_rows = np.array([sdm.sp2row[s] for s in df["pollinator"]])
    n = store.N_full[p_idx, qi].astype(np.float64)[:, None]
    dg = np.minimum(store.FC[p_idx], store.AC[qi]).sum(1)[:, None]
    dl = store.delta_local_pairs(p_idx, qi)[:, None]
    ds = np.minimum(store.FC[p_idx], sdm.A_sdm[q_rows]).sum(1)[:, None]
    db = np.zeros((len(df), 1))
    for p in np.unique(p_idx):
        m = p_idx == p
        db[m, 0] = sdm.delta_bilateral(p, q_rows[m])
    return {"base": np.hstack([store.VF[p_idx], store.VP[qi], n, aff.pairs(p_idx, qi)]),
            "gbif": np.hstack([dg, dl]), "sdm": np.hstack([ds, db])}


Btr, Bte = block(tr), block(te)
ARMS = {"base (Vf,Vp,N,tax)": ["base"], "base + GBIF temporal": ["base", "gbif"],
        "base + SDM temporal": ["base", "sdm"], "base + both": ["base", "gbif", "sdm"]}
print(f"\n{'arm':<24}{'LR PR-AUC [95% CI]':>28}{'GBM PR-AUC':>14}")
rows = []
for name, keys in ARMS.items():
    Xtr = np.hstack([Btr[k] for k in keys]); Xte = np.hstack([Bte[k] for k in keys])
    lr = Pipeline([("s", StandardScaler()), ("lr", LogisticRegression(max_iter=2000, random_state=seed))])
    lr.fit(Xtr, tr["label"]); s_lr = lr.decision_function(Xte)
    gb = HistGradientBoostingClassifier(random_state=seed, max_iter=400, learning_rate=0.08,
                                        min_samples_leaf=40, l2_regularization=1.0).fit(Xtr, tr["label"])
    s_gb = gb.predict_proba(Xte)[:, 1]
    pr, lo, hi, _ = bootstrap_pr_by_plant(te.assign(score=s_lr), 300, seed)
    prg, _, _, _ = bootstrap_pr_by_plant(te.assign(score=s_gb), 300, seed)
    rows.append({"arm": name, "lr_pr": pr, "lo": lo, "hi": hi, "gbm_pr": prg})
    print(f"{name:<24}{pr:>10.4f} [{lo:.3f},{hi:.3f}]{prg:>14.4f}")
pd.DataFrame(rows).to_csv(cfg["edges"].parent / "sdm_prauc_v1.csv", index=False)
print("\nThe project's original claim lives on this metric: does SDM temporal beat GBIF temporal here?")
