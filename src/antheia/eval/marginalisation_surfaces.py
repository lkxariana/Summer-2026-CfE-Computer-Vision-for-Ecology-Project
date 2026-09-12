"""The marginalisation test on the production surfaces, full universe.

`run_marginalisation_test.py` makes the claim on one joint field model but only within the trained
sub-universe (379 plants, 5,354 candidates). The production surfaces -- PPE for plants, SDM for
pollinators -- cover every taxon, so the same four statistics can be scored under the full protocol:
663 validation plants, all 13,124 candidates, tier A. The two sides come from different model
families here, which is a caveat for the absolute numbers but not for the contrast: every statistic
is built from the same two surfaces and differs only in what is summed out first.

  joint   sum_{c,w} P Q                (via the shared SVD basis; pearson 1.000 against the exact product)
  space   sum_c (sum_w P)(sum_w Q) / W
  time    sum_w (sum_c P)(sum_c Q) / C
  scalar  (sum P)(sum Q) / (C W)

Each is scored alone and as a re-ranker of popularity's top 200, with paired bootstraps of joint
against each marginal, overall and on the unseen-genus stratum.
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from antheia.paths import REPO_ROOT as ROOT
from antheia.metrics import paired_bootstrap, retrieval_metrics
from antheia.store import UniverseStore


def metrics(S, tp, part, Y):
    pp = pd.DataFrame([retrieval_metrics(S[i], part[sp], ks=(10, 50)) for i, sp in enumerate(tp)])
    return pp["nrecall@10"].to_numpy(), pp["nrecall@50"].to_numpy(), pp["ap"].mean(), \
        average_precision_score(Y.ravel(), S.ravel().astype(np.float64))


def main():
    store = UniverseStore(curves="modelled")
    F = ROOT / "data/features"
    split = json.load(open(ROOT / "data/splits/plants_75_10_15.json"))
    e = pd.read_parquet(ROOT / "data/network/edges.parquet")
    e = e[e.plant.isin(store.p2i) & e.pollinator.isin(store.q2i)]
    train = e[e.plant.isin(set(split["train"]))]
    test = e[e.plant.isin(set(split["val"]))]; test = test[test.tier == "A"]
    tp = sorted(set(test.plant)); pi = store.idx_plants(tp)
    part = {sp: set(store.idx_polls(g.pollinator)) for sp, g in test.groupby("plant")}
    train_gen = {s.split()[0] for s in set(train.plant)}
    seen = np.array([sp.split()[0] in train_gen for sp in tp])
    Y = np.zeros((len(tp), len(store.polls)), np.int8)
    for i, sp in enumerate(tp):
        Y[i, list(part[sp])] = 1
    pop = np.bincount(store.idx_polls(train.pollinator), minlength=len(store.polls)).astype(np.float64)
    C, W = 3335, 52
    print(f"[data] {len(tp)} val plants (unseen genus {(~seen).sum()}), {len(store.polls)} candidates, "
          f"{int(Y.sum()):,} tier-A partners", flush=True)

    Ps, Qs = np.load(F / "plant_surf_space.npy")[pi].astype(np.float64), np.load(F / "poll_surf_space.npy").astype(np.float64)
    Pt, Qt = np.load(F / "plant_surf_time.npy")[pi].astype(np.float64), np.load(F / "poll_surf_time.npy").astype(np.float64)
    Pm, Qm = np.load(F / "plant_surf_mass.npy")[pi].astype(np.float64), np.load(F / "poll_surf_mass.npy").astype(np.float64)
    stats = {
        "joint":  np.maximum(store.plant_proj[pi].astype(np.float64) @ store.poll_proj.T.astype(np.float64), 0),
        "space":  (Ps @ Qs.T) / W,
        "time":   (Pt @ Qt.T) / C,
        "scalar": Pm[:, None] * Qm[None, :] / (C * W),
    }
    stats = {k: np.log1p(v) for k, v in stats.items()}
    top = np.argsort(-pop)[:200]
    in_top = np.isin(np.arange(len(store.polls)), top)[None, :]

    rows, res = [], {}
    nr, nr50, mp, pr = metrics(np.tile(np.log1p(pop), (len(tp), 1)), tp, part, Y)
    print(f"  {'popularity':<24} nR@10 {nr.mean():.4f}  unseen {nr[~seen].mean():.4f}  nR@50 {nr50.mean():.4f}  PR {pr:.4f}")
    rows.append(dict(statistic="popularity", mode="alone", nrecall10=nr.mean(), unseen=nr[~seen].mean(), nrecall50=nr50.mean(), map=mp, pr_auc=pr))
    for mode in ("alone", "rerank top-200"):
        for k, L in stats.items():
            S = L if mode == "alone" else np.where(in_top, L, L.min() - 1.0)
            nr, nr50, mp, pr = metrics(S, tp, part, Y); res[(k, mode)] = (nr, nr50)
            line = f"  {k + ' | ' + mode:<24} nR@10 {nr.mean():.4f}  unseen {nr[~seen].mean():.4f}  nR@50 {nr50.mean():.4f}  MAP {mp:.4f}  PR {pr:.4f}"
            r = dict(statistic=k, mode=mode, nrecall10=nr.mean(), unseen=nr[~seen].mean(), nrecall50=nr50.mean(), map=mp, pr_auc=pr)
            if k != "joint":
                a, b = res[("joint", mode)], res[(k, mode)]
                d10, _, _, p10 = paired_bootstrap(a[0], b[0], 5000, 42)
                d50, lo, hi, p50 = paired_bootstrap(a[1], b[1], 5000, 42)
                du, _, _, pu = paired_bootstrap(a[0][~seen], b[0][~seen], 5000, 42)
                line += f"   joint-vs-{k}: @10 {d10:+.4f} p={p10:.3f}  @50 {d50:+.4f} [{lo:+.4f},{hi:+.4f}] p={p50:.3f}  unseen@10 {du:+.4f} p={pu:.3f}"
                r.update(joint_minus_10=d10, p10=p10, joint_minus_50=d50, p50=p50, joint_minus_unseen10=du, p_unseen=pu)
            print(line, flush=True); rows.append(r)

    out = ROOT / "results/marginalisation_surfaces_val_tierA.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[wrote] {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
