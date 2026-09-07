"""Parameter-free interaction scores from a joint field model, one row per sampling scheme.

Given the joint model's grid embedding H [n_cells*52, d] and species vectors u, two scores need
no training at all and so credit nothing but the representation:

  encounter   mean over the grid of sigma(u_p.h) * sigma(u_q.h)   -- expected co-presence
  cosine      cos(u_p, u_q)                                        -- niche similarity in encoder space

Both are computed exactly over the 0.5-degree x week grid (173k points) rather than by Monte Carlo,
since H is cached. Scored on the validation plants, tier A, all 13,124 candidates, split by
whether the plant's genus appears in training. Popularity and N are printed for reference.
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


def evaluate(S, tp, part, seen, Y, label, ref=None):
    pp = pd.DataFrame([retrieval_metrics(S[i], part[sp], ks=(10, 50)) for i, sp in enumerate(tp)])
    nr = pp["nrecall@10"].to_numpy()
    pr = average_precision_score(Y.ravel(), S.ravel().astype(np.float64))
    line = (f"  {label:<28} nR@10 {nr.mean():.4f}  seen {nr[seen].mean():.4f}  unseen {nr[~seen].mean():.4f}  "
            f"nR@50 {pp['nrecall@50'].mean():.4f}  MAP {pp['ap'].mean():.4f}  PR {pr:.4f}")
    if ref is not None:
        d, lo, hi, pv = paired_bootstrap(nr, ref, 2000, 42)
        line += f"   vs popularity {d:+.4f} p={pv:.3f}"
    print(line, flush=True)
    return nr, dict(arm=label, nrecall10=nr.mean(), nrecall10_seen=nr[seen].mean(),
                    nrecall10_unseen=nr[~seen].mean(), nrecall50=pp["nrecall@50"].mean(),
                    map=pp["ap"].mean(), pr_auc=pr)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", required=True, help="joint-field output directories")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"

    store = UniverseStore(curves="modelled")
    split = json.load(open(ROOT / "data/splits/plants_75_10_15.json"))
    e = pd.read_parquet(ROOT / "data/network/edges.parquet")
    e = e[e.plant.isin(store.p2i) & e.pollinator.isin(store.q2i)]
    train = e[e.plant.isin(set(split["train"]))]
    test = e[e.plant.isin(set(split["val"]))]; test = test[test.tier == "A"]
    tp = sorted(set(test.plant))
    pi = store.idx_plants(tp)
    part = {sp: set(store.idx_polls(g.pollinator)) for sp, g in test.groupby("plant")}
    train_gen = {s.split()[0] for s in set(train.plant)}
    seen = np.array([sp.split()[0] in train_gen for sp in tp])
    Y = np.zeros((len(tp), len(store.polls)), np.int8)
    for i, sp in enumerate(tp):
        Y[i, list(part[sp])] = 1
    print(f"[data] {len(tp)} val plants, tier A; genus seen {seen.sum()} / unseen {(~seen).sum()}", flush=True)

    rows = []
    pop = np.bincount(store.idx_polls(train.pollinator), minlength=len(store.polls)).astype(np.float64)
    ref, r = evaluate(np.tile(np.log1p(pop), (len(tp), 1)), tp, part, seen, Y, "popularity"); rows.append(r)
    _, r = evaluate(np.log1p(store.N_full[pi].astype(np.float64)), tp, part, seen, Y, "N (co-occurrence)", ref); rows.append(r)

    for run in args.runs:
        run = Path(run)
        U_p = np.load(run / "plant_field.npy"); U_q = np.load(run / "poll_field.npy")
        d_p = np.load(run / "plant_field_direct.npy"); d_q = np.load(run / "poll_field_direct.npy")
        H = torch.from_numpy(np.load(run / "grid_h.npy").reshape(-1, U_p.shape[1])).to(dev)   # [M, d]
        b = float(np.load(run / "head_bias.npy")) if (run / "head_bias.npy").exists() else 0.0
        M = H.shape[0]
        Up = torch.from_numpy(U_p[pi]).to(dev); Uq = torch.from_numpy(U_q).to(dev)
        with torch.no_grad():
            A = torch.sigmoid(Up @ H.T + b)                       # [n_val, M]  plant presence over the grid
            enc = torch.empty(len(pi), len(U_q), device=dev)
            for s in range(0, len(U_q), 1024):                   # [n_val, Q]  expected co-presence
                B = torch.sigmoid(Uq[s:s + 1024] @ H.T + b)      # [q, M]
                enc[:, s:s + 1024] = A @ B.T / M
            cos = torch.nn.functional.normalize(Up, dim=1) @ torch.nn.functional.normalize(Uq, dim=1).T
        tag = run.name.replace("joint_field_", "")
        cov = f"(plants {d_p[pi].mean():.0%} / polls {d_q.mean():.0%} trained)"
        print(f"[{tag}] {cov}", flush=True)
        _, r = evaluate(torch.log(enc + 1e-12).cpu().numpy(), tp, part, seen, Y, f"{tag}: encounter", ref); r["scheme"] = tag; rows.append(r)
        _, r = evaluate(cos.cpu().numpy(), tp, part, seen, Y, f"{tag}: cosine", ref); r["scheme"] = tag; rows.append(r)
        # the encounter term is prevalence-loaded by construction; the cheapest conditional read is
        # to re-rank popularity's own top 200 by the encounter score
        top = np.argsort(-pop)[:200]
        L = torch.log(enc + 1e-12).cpu().numpy()
        S2 = np.full((len(tp), len(U_q)), L.min() - 1.0); S2[:, top] = L[:, top]   # finite floor: PR-AUC rejects -inf
        _, r = evaluate(S2, tp, part, seen, Y, f"{tag}: encounter | pop top-200", ref); r["scheme"] = tag; rows.append(r)

    out = ROOT / "results/field_encounter_val_tierA.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[wrote] {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
