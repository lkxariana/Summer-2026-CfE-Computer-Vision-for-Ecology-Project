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

# Dan's critique (09-01): SDM's value is generalisation to places/species where GBIF observation is
# THIN. Evaluating on documented interactions tests SDM precisely where GBIF is strongest, which is
# rigged against it. Fix: stratify the SAME test set by GBIF observation density and ask whether the
# GBIF-vs-SDM ordering FLIPS in the sparse stratum.
cfg = load_config()
store = FeatureStore(cfg)
sdm = SDMFeatures(cfg, store)
seed, nboot = cfg["eval"]["seed"], cfg["eval"]["bootstrap_n"]
counts = pd.read_parquet(cfg["cache_dir"] / "gbif_record_counts.parquet")["n_gbif"]

edges = pd.read_parquet(cfg["edges"])
e = edges[edges["pollinator"].isin(set(sdm.species))].reset_index(drop=True)
split = load_split(cfg["edges"].parent / "split_v1.json")
train_pos, test_pos = e[e["plant"].isin(split["train"])], e[e["plant"].isin(split["test"])]
aff = TaxonomyAffinity(store, train_pos, cfg["paths"]["globi"])
known = set(zip(e["plant"], e["pollinator"]))
rng = np.random.default_rng(seed)

n_gbif = np.array([counts.get(s, 0) for s in sdm.species], float)
q_terc = np.digitize(n_gbif, np.quantile(n_gbif[n_gbif > 0], [1/3, 2/3]))
print(f"GBIF records per SDM species: median {np.median(n_gbif):.0f}, "
      f"tercile cuts {np.quantile(n_gbif[n_gbif>0],[1/3,2/3]).round(0)}")
for t in (0, 1, 2):
    print(f"  tercile {t}: {int((q_terc==t).sum())} species, median records {np.median(n_gbif[q_terc==t]):.0f}")

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
    return {"base": np.hstack([n, aff.pairs(p_idx, qi), store.Frs[p_idx][:, None], store.Prs[qi][:, None]]),
            "gbif": np.hstack([dg, dl]), "sdm": np.hstack([ds, db])}


Bp, Bn = block(pi, qpos), block(pi, qneg)
ARMS = {"base": ["base"], "base+GBIF": ["base", "gbif"], "base+SDM": ["base", "sdm"]}
fitted = {}
for name, keys in ARMS.items():
    X = np.vstack([np.hstack([Bp[k] for k in keys]), np.hstack([Bn[k] for k in keys])])
    y = np.concatenate([np.ones(len(pi)), np.zeros(len(pi))])
    fitted[name] = (keys, HistGradientBoostingClassifier(random_state=seed, max_iter=400,
                    learning_rate=0.08, min_samples_leaf=40, l2_regularization=1.0).fit(X, y))
    print(f"trained {name}", flush=True)

allr = np.arange(len(sdm.species))
partners = {sp: {sdm.sp2row[q] for q in g["pollinator"]} for sp, g in test_pos.groupby("plant")}
rec = {n: {t: [] for t in (0, 1, 2)} for n in ARMS}
for i, (sp, part) in enumerate(partners.items()):
    p = store.p2i[sp]
    b = block(np.full(len(allr), p), allr)
    for name, (keys, m) in fitted.items():
        s = m.predict_proba(np.hstack([b[k] for k in keys]))[:, 1]
        for t in (0, 1, 2):
            tg = {r for r in part if q_terc[r] == t}
            if not tg:
                continue
            sm = s.copy()
            sm[list(part - tg)] = -np.inf                 # other true partners are not distractors
            t10 = set(np.argpartition(-sm, 10)[:10].tolist())
            rec[name][t].append(len(tg & t10) / len(tg))
    if (i + 1) % 150 == 0:
        print(f"  {i+1}/{len(partners)}", flush=True)

print(f"\n{'arm':<12}" + "".join(f"{'T'+str(t)+' (sparse→dense)':>22}" for t in (0, 1, 2)))
rows = []
for name in ARMS:
    line = f"{name:<12}"
    for t in (0, 1, 2):
        m_, lo, hi, _ = bootstrap_mean(np.array(rec[name][t]), nboot, seed)
        rows.append({"arm": name, "tercile": t, "r10": m_, "lo": lo, "hi": hi, "n": len(rec[name][t])})
        line += f"{m_:>10.4f} [{lo:.2f},{hi:.2f}]"
    print(line)
pd.DataFrame(rows).to_csv(cfg["edges"].parent / "sdm_ood_v1.csv", index=False)
d = pd.DataFrame(rows).pivot(index="tercile", columns="arm", values="r10")
print("\nSDM minus GBIF by GBIF-density tercile:")
for t in (0, 1, 2):
    print(f"  T{t}: {d.loc[t,'base+SDM'] - d.loc[t,'base+GBIF']:+.4f}")
print("\nDan's hypothesis predicts a POSITIVE difference in T0 (GBIF-sparse) shrinking toward T2.")
