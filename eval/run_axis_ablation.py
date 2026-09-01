import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from antheia import models
from antheia.config import load_config
from antheia.metrics import bootstrap_mean
from antheia.pairs import load_split
from antheia.store import FeatureStore
from antheia.taxonomy import TaxonomyAffinity

GEO = ["vf", "vp", "n", "delta", "delta_local", "frs", "prs"]
LINEAR = {"rank_n": ["n"], "rank_n_tax_ldelta": ["n", "tax", "delta_local"]}


def main():
    ap = argparse.ArgumentParser(description="Is the provenance axis distinct from taxonomy and geography?")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    store = FeatureStore(cfg)
    edges = pd.read_parquet(cfg["edges"])
    prov = pd.read_csv(cfg["edges"].parent / "edge_provenance_v1.csv")
    e = edges.merge(prov, on=["plant", "pollinator"], how="left")
    e["curated"] = e["n_inat"].notna() & (e["n_inat"] < e["n"])

    g = pd.read_csv(cfg["paths"]["globi"], usecols=["sourceTaxonName", "sourceTaxonOrderName"]).dropna()
    order = g.groupby("sourceTaxonName")["sourceTaxonOrderName"].agg(lambda s: s.mode().iat[0])
    e["order"] = e["pollinator"].map(order).fillna("UNK")

    known_pos = set(zip(e["plant"], e["pollinator"]))
    seed, nboot = cfg["eval"]["seed"], cfg["eval"]["bootstrap_n"]
    split = load_split(cfg["edges"].parent / "split_v1.json")
    train_pos = e[e["plant"].isin(split["train"])]
    test_pos = e[e["plant"].isin(split["test"])]
    aff = TaxonomyAffinity(store, train_pos, cfg["paths"]["globi"])

    def bp(p_idx, q_idx, c):
        return aff.pairs(p_idx, q_idx) if c == "tax" else store.assemble(p_idx, q_idx, (c,))

    pi, qp, qn = models.rank_contrast_sets(train_pos, store, 10, seed, known_pos)
    comps = sorted({c for s in LINEAR.values() for c in s} | set(GEO))
    Xp = {c: bp(pi, qp, c) for c in comps}
    Xn = {c: bp(pi, qn, c) for c in comps}

    scorers = {}
    for name, spec in LINEAR.items():
        d = np.hstack([Xp[c] for c in spec]) - np.hstack([Xn[c] for c in spec])
        X, y = np.vstack([d, -d]), np.concatenate([np.ones(len(d)), np.zeros(len(d))])
        pipe = Pipeline([("s", StandardScaler(with_mean=False)),
                         ("lr", LogisticRegression(max_iter=1000, random_state=seed, fit_intercept=False))])
        pipe.fit(X, y)
        scorers[name] = (spec, lambda M, pp=pipe: pp.decision_function(M))
    pos_rows = np.arange(0, len(pi), 10)
    gbm = HistGradientBoostingClassifier(random_state=seed, max_iter=800, learning_rate=0.05,
                                         min_samples_leaf=100, l2_regularization=5.0)
    gbm.fit(np.vstack([np.hstack([Xp[c][pos_rows] for c in GEO]), np.hstack([Xn[c] for c in GEO])]),
            np.concatenate([np.ones(len(pos_rows)), np.zeros(len(pi))]))
    scorers["gbm_geo"] = (GEO, lambda M: gbm.predict_proba(M)[:, 1])

    # Pollinator-side masks define each evaluation stratum.
    hym = {store.q2i[s] for s in store.polls if order.get(s, "UNK") == "Hymenoptera"}
    lep = {store.q2i[s] for s in store.polls if order.get(s, "UNK") == "Lepidoptera"}
    strata = {
        "all partners": lambda r: True,
        "curated only": lambda r: r["curated"],
        "iNat-only": lambda r: not r["curated"],
        "Hymenoptera, curated": lambda r: r["curated"] and r["order"] == "Hymenoptera",
        "Hymenoptera, iNat-only": lambda r: (not r["curated"]) and r["order"] == "Hymenoptera",
        "Lepidoptera, all": lambda r: r["order"] == "Lepidoptera",
    }
    cand_mask = {"Hymenoptera, curated": hym, "Hymenoptera, iNat-only": hym, "Lepidoptera, all": lep}

    rows = []
    per_stratum = {s: {n: [] for n in scorers} for s in strata}
    for sp, grp in test_pos.groupby("plant"):
        p = store.p2i[sp]
        blk = {c: (aff.plant(p) if c == "tax" else store.assemble_plant(p, (c,))) for c in comps}
        svec = {n: fn(np.hstack([blk[c] for c in spec])) for n, (spec, fn) in scorers.items()}
        recs = grp.to_dict("records")
        for sname, keep in strata.items():
            targets = {store.q2i[r["pollinator"]] for r in recs if keep(r)}
            if not targets:
                continue
            allowed = cand_mask.get(sname)
            for n, s in svec.items():
                sm = s.copy()
                if allowed is not None:
                    block = np.ones(len(sm), bool)
                    block[list(allowed)] = False
                    sm[block] = -np.inf                      # rank only within this pollinator order
                other = {store.q2i[r["pollinator"]] for r in recs} - targets
                sm[list(other)] = -np.inf                     # other true partners are not distractors
                t10 = set(np.argpartition(-sm, 10)[:10].tolist())
                per_stratum[sname][n].append(len(targets & t10) / len(targets))

    print(f"\n{'stratum':<26}{'n_plants':>9}" + "".join(f"{n:>22}" for n in scorers))
    for sname in strata:
        vals = per_stratum[sname]
        n_pl = len(next(iter(vals.values())))
        if n_pl == 0:
            continue
        line = f"{sname:<26}{n_pl:>9}"
        rec = {"stratum": sname, "n_plants": n_pl}
        for n in scorers:
            m, lo, hi, _ = bootstrap_mean(np.array(vals[n]), nboot, seed)
            line += f"{m:>13.4f} [{lo:.2f}]"
            rec[n] = m
            rec[f"{n}_lo"], rec[f"{n}_hi"] = lo, hi
        rows.append(rec)
        print(line, flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(cfg["edges"].parent / "axis_ablation_v1.csv", index=False)

    print("\nKey contrast — within Hymenoptera only (taxon composition held fixed):")
    h = df.set_index("stratum")
    if "Hymenoptera, curated" in h.index and "Hymenoptera, iNat-only" in h.index:
        for n in scorers:
            a, b = h.loc["Hymenoptera, curated", n], h.loc["Hymenoptera, iNat-only", n]
            print(f"  {n:<20} curated {a:.4f} vs iNat-only {b:.4f}  (ratio {a / max(b, 1e-9):.2f}x)")
    print("\nIf the curated/iNat gap survives within one order, provenance is not a relabeled taxonomy split.")


if __name__ == "__main__":
    main()
