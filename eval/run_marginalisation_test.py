"""Does marginalising the spatio-temporal field destroy the signal? Same model, same presence grid.

From one joint field model take each species' presence over the (cell, week) grid,
P[c, w] = sigma(u . h(c, w)). Four overlap statistics of the SAME two surfaces, differing only in
what is collapsed before the product:

  joint     sum_{c,w} P_p[c,w] Q_q[c,w]                       per-cell, per-week  (nothing collapsed)
  space     sum_c  (sum_w P_p[c,w]) (sum_w Q_q[c,w]) / 52     weeks collapsed first  -- range overlap
  time      sum_w  (sum_c P_p[c,w]) (sum_c Q_q[c,w]) / C      cells collapsed first  -- phenology overlap
  scalar    (sum P_p) (sum Q_q) / (52 C)                      both collapsed         -- prevalence product

Every statistic is an expected co-presence under a different independence assumption, so they share
units and differ only in the assumption. Each is scored alone and as a re-ranker of popularity's top
200, within the trained sub-universe, with paired bootstraps of joint against each marginal.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from antheia.metrics import paired_bootstrap, retrieval_metrics
from antheia.store import UniverseStore


def presence(U, H, dev, chunk=512):
    """sigma(U H^T) -> [n, C*52] on device, float16 to fit the pollinator side."""
    out = torch.empty(len(U), H.shape[0], device=dev, dtype=torch.float16)
    Ut = torch.from_numpy(U).to(dev)
    for s in range(0, len(U), chunk):
        out[s:s + chunk] = torch.sigmoid(Ut[s:s + chunk] @ H.T).half()
    return out


def metrics(S, tp, part, seen, Y):
    pp = pd.DataFrame([retrieval_metrics(S[i], part[sp], ks=(10, 50)) for i, sp in enumerate(tp)])
    pr = average_precision_score(Y.ravel(), S.ravel().astype(np.float64))
    return pp["nrecall@10"].to_numpy(), pp["nrecall@50"].to_numpy(), pp["ap"].mean(), pr


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"

    store = UniverseStore(curves="modelled")
    split = json.load(open(ROOT / "data/splits/plants_75_10_15.json"))
    e = pd.read_parquet(ROOT / "data/network/edges.parquet")
    e = e[e.plant.isin(store.p2i) & e.pollinator.isin(store.q2i)]
    train = e[e.plant.isin(set(split["train"]))]
    test = e[e.plant.isin(set(split["val"]))]; test = test[test.tier == "A"]
    tp_all = sorted(set(test.plant))
    part_all = {sp: set(store.idx_polls(g.pollinator)) for sp, g in test.groupby("plant")}
    train_gen = {s.split()[0] for s in set(train.plant)}
    pop = np.bincount(store.idx_polls(train.pollinator), minlength=len(store.polls)).astype(np.float64)

    rows = []
    for run in args.runs:
        run = Path(run); tag = run.name.replace("joint_field_", "")
        U_p = np.load(run / "plant_field.npy"); U_q = np.load(run / "poll_field.npy")
        d_p = np.load(run / "plant_field_direct.npy"); d_q = np.load(run / "poll_field_direct.npy")
        Hg = np.load(run / "grid_h.npy"); C = Hg.shape[0]; W = Hg.shape[1]
        H = torch.from_numpy(Hg.reshape(C * W, -1)).to(dev)

        pi_all = store.idx_plants(tp_all)
        cand = np.flatnonzero(d_q)
        tp = [sp for sp, i in zip(tp_all, pi_all) if d_p[i]]
        part = {sp: {int(np.searchsorted(cand, q)) for q in part_all[sp] if d_q[q]} for sp in tp}
        tp = [sp for sp in tp if part[sp]]
        pi = store.idx_plants(tp)
        seen = np.array([sp.split()[0] in train_gen for sp in tp])
        Y = np.zeros((len(tp), len(cand)), np.int8)
        for i, sp in enumerate(tp):
            Y[i, list(part[sp])] = 1
        print(f"\n[{tag}] {len(tp)} val plants (unseen genus {(~seen).sum()}), {len(cand)} candidates, "
              f"{int(Y.sum()):,} partners", flush=True)

        with torch.no_grad():
            Pg = presence(U_p[pi], H, dev)                          # [n_p, C*W]
            Qg = presence(U_q[cand], H, dev)                        # [n_q, C*W]
            P3 = Pg.view(len(pi), C, W).float(); Q3 = Qg.view(len(cand), C, W).float()
            stats = {
                "joint":  (Pg.float() @ Qg.float().T) / (C * W),
                "space":  (P3.sum(2) @ Q3.sum(2).T) / (C * W * W),
                "time":   (P3.sum(1) @ Q3.sum(1).T) / (C * W * C),
                "scalar": (P3.sum((1, 2))[:, None] * Q3.sum((1, 2))[None, :]) / (C * W) ** 2,
            }
            del P3, Q3, Pg, Qg; torch.cuda.empty_cache()
        stats = {k: torch.log(v + 1e-12).cpu().numpy() for k, v in stats.items()}
        top = np.argsort(-pop[cand])[:200]
        popS = np.tile(np.log1p(pop[cand]), (len(tp), 1))

        res = {}
        nr, nr50, mp, pr = metrics(popS, tp, part, seen, Y); res["popularity"] = (nr, nr50)
        print(f"  {'popularity':<26} nR@10 {nr.mean():.4f}  unseen {nr[~seen].mean():.4f}  nR@50 {nr50.mean():.4f}  PR {pr:.4f}")
        rows.append(dict(scheme=tag, statistic="popularity", mode="alone", nrecall10=nr.mean(), unseen=nr[~seen].mean(),
                         nrecall50=nr50.mean(), map=mp, pr_auc=pr))
        for mode in ("alone", "rerank top-200"):
            for k, L in stats.items():
                S = L if mode == "alone" else np.where(np.isin(np.arange(len(cand)), top)[None, :], L, L.min() - 1.0)
                nr, nr50, mp, pr = metrics(S, tp, part, seen, Y); res[(k, mode)] = (nr, nr50)
                line = f"  {k + ' | ' + mode:<26} nR@10 {nr.mean():.4f}  unseen {nr[~seen].mean():.4f}  nR@50 {nr50.mean():.4f}  MAP {mp:.4f}  PR {pr:.4f}"
                r = dict(scheme=tag, statistic=k, mode=mode, nrecall10=nr.mean(), unseen=nr[~seen].mean(),
                         nrecall50=nr50.mean(), map=mp, pr_auc=pr)
                if k != "joint":
                    a, b = res[("joint", mode)], res[(k, mode)]
                    d10, lo, hi, p10 = paired_bootstrap(a[0], b[0], 5000, 42)
                    d50, lo5, hi5, p50 = paired_bootstrap(a[1], b[1], 5000, 42)
                    line += f"   joint-vs-{k}: @10 {d10:+.4f} p={p10:.3f}  @50 {d50:+.4f} [{lo5:+.4f},{hi5:+.4f}] p={p50:.3f}"
                    r.update(joint_minus_10=d10, p10=p10, joint_minus_50=d50, p50=p50)
                print(line, flush=True); rows.append(r)

    out = ROOT / "results/marginalisation_test_val_tierA.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n[wrote] {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
