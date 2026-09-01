import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from antheia import models
from antheia.config import load_config
from antheia.pairs import load_split
from antheia.store import FeatureStore
from antheia.taxonomy import TaxonomyAffinity

# Falsification test for "detection is pair-dependent" (which breaks ULTR rank-1).
# If the curated/iNat gap vanishes WITHIN pollinator families, detection is taxon-dependent
# but not otherwise pair-dependent -> rank-1 holds conditional on taxon -> debiasing identifiable.
GEO = ["vf", "vp", "n", "delta", "delta_local", "frs", "prs"]
LINEAR = {"rank_n": ["n"], "rank_vp_only": ["vp"], "rank_n_tax_ldelta": ["n", "tax", "delta_local"]}
MIN_PLANTS = 25

cfg = load_config()
store = FeatureStore(cfg)
edges = pd.read_parquet(cfg["edges"])
prov = pd.read_csv(cfg["edges"].parent / "edge_provenance_v1.csv")
e = edges.merge(prov, on=["plant", "pollinator"], how="left")
e["curated"] = e["n_inat"].notna() & (e["n_inat"] < e["n"])

g = pd.read_csv(cfg["paths"]["globi"], usecols=["sourceTaxonName", "sourceTaxonFamilyName"]).dropna()
fam = g.groupby("sourceTaxonName")["sourceTaxonFamilyName"].agg(lambda s: s.mode().iat[0])
e["family"] = e["pollinator"].map(fam).fillna("UNK")

known_pos = set(zip(e["plant"], e["pollinator"]))
seed = cfg["eval"]["seed"]
split = load_split(cfg["edges"].parent / "split_v1.json")
train_pos, test_pos = e[e["plant"].isin(split["train"])], e[e["plant"].isin(split["test"])]
aff = TaxonomyAffinity(store, train_pos, cfg["paths"]["globi"])

bp = lambda pi_, qi_, c: aff.pairs(pi_, qi_) if c == "tax" else store.assemble(pi_, qi_, (c,))
pi, qp, qn = models.rank_contrast_sets(train_pos, store, 10, seed, known_pos)
comps = sorted({c for s in LINEAR.values() for c in s} | set(GEO))
Xp = {c: bp(pi, qp, c) for c in comps}
Xn = {c: bp(pi, qn, c) for c in comps}
scorers = {}
for name, spec in LINEAR.items():
    d = np.hstack([Xp[c] for c in spec]) - np.hstack([Xn[c] for c in spec])
    pipe = Pipeline([("s", StandardScaler(with_mean=False)),
                     ("lr", LogisticRegression(max_iter=1000, random_state=seed, fit_intercept=False))])
    pipe.fit(np.vstack([d, -d]), np.concatenate([np.ones(len(d)), np.zeros(len(d))]))
    scorers[name] = (spec, lambda M, pp=pipe: pp.decision_function(M))
pos_rows = np.arange(0, len(pi), 10)
gbm = HistGradientBoostingClassifier(random_state=seed, max_iter=800, learning_rate=0.05,
                                     min_samples_leaf=100, l2_regularization=5.0)
gbm.fit(np.vstack([np.hstack([Xp[c][pos_rows] for c in GEO]), np.hstack([Xn[c] for c in GEO])]),
        np.concatenate([np.ones(len(pos_rows)), np.zeros(len(pi))]))
scorers["gbm_geo"] = (GEO, lambda M: gbm.predict_proba(M)[:, 1])

fams = [f for f in test_pos["family"].unique() if f != "UNK"]
fam_idx = {f: {store.q2i[s] for s in store.polls if fam.get(s, "UNK") == f} for f in fams}
acc = {(f, c): {n: [] for n in scorers} for f in fams for c in (True, False)}

for sp, grp in test_pos.groupby("plant"):
    p = store.p2i[sp]
    blk = {c: (aff.plant(p) if c == "tax" else store.assemble_plant(p, (c,))) for c in comps}
    svec = {n: fn(np.hstack([blk[c] for c in spec])) for n, (spec, fn) in scorers.items()}
    recs = grp.to_dict("records")
    all_partners = {store.q2i[r["pollinator"]] for r in recs}
    for f in {r["family"] for r in recs} & set(fams):
        cand = fam_idx[f]
        if len(cand) < 20:
            continue
        for cur in (True, False):
            targets = {store.q2i[r["pollinator"]] for r in recs if r["family"] == f and r["curated"] == cur}
            if not targets:
                continue
            for n, s in svec.items():
                sm = s.copy()
                block = np.ones(len(sm), bool)
                block[list(cand)] = False
                sm[block] = -np.inf
                sm[list(all_partners - targets)] = -np.inf
                t10 = set(np.argpartition(-sm, 10)[:10].tolist())
                acc[(f, cur)][n].append(len(targets & t10) / len(targets))

rows = []
for f in fams:
    n_cur = len(acc[(f, True)]["rank_n"])
    n_inat = len(acc[(f, False)]["rank_n"])
    if min(n_cur, n_inat) < MIN_PLANTS:
        continue
    r = {"family": f, "n_plants_curated": n_cur, "n_plants_inat": n_inat}
    for n in scorers:
        c, i = np.mean(acc[(f, True)][n]), np.mean(acc[(f, False)][n])
        r[f"{n}_ratio"] = c / i if i > 0 else np.nan
    rows.append(r)
df = pd.DataFrame(rows).sort_values("n_plants_curated", ascending=False)
df.to_csv(cfg["edges"].parent / "family_strata_v1.csv", index=False)
print(f"families with >={MIN_PLANTS} test plants in BOTH provenance strata: {len(df)}\n")
print(df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
print("\ncurated/iNat recall@10 ratio, WITHIN family (1.00 = detection not pair-dependent beyond taxon)")
for n in scorers:
    v = df[f"{n}_ratio"].dropna()
    print(f"  {n:<20} median {v.median():.2f}  range {v.min():.2f}-{v.max():.2f}  "
          f"|log2| mean {np.abs(np.log2(v)).mean():.2f}")
