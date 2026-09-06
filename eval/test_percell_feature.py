"""Is there per-cell phenological information, independent of whether a network can learn it?

The surface encoder underperforms the plain two-tower, which leaves two possibilities: the per-cell
signal is not there, or the encoder cannot extract it. This computes the signal directly --
sum over cells and weeks of f(p,c,w) * a(q,c,w), exactly on GPU -- and adds it to the strongest
retrieval model as one column. A species-permutation control separates real matching from capacity.
"""
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))
from antheia.metrics import paired_bootstrap, retrieval_metrics
from antheia.store import UniverseStore
from run_encoding_ablation import run_row, _pca


class LocalOverlap:
    """Exact per-cell co-activity on GPU: both surfaces flattened, dot product per pair."""

    def __init__(self, store, device="cuda", permute=False):
        self.dev = device
        P = np.array(store.plant_surfaces).reshape(len(store.plants), -1)
        Q = np.array(store.poll_surfaces).reshape(len(store.polls), -1)
        if permute:
            rng = np.random.default_rng(0)
            P = P[rng.permutation(len(P))]
            Q = Q[rng.permutation(len(Q))]
        self.P = torch.from_numpy(P).to(device)
        self.Q = torch.from_numpy(Q).to(device)

    def pairs(self, pi, qi, chunk=4096):
        out = np.empty(len(pi), np.float32)
        for s in range(0, len(pi), chunk):
            a = self.P[torch.from_numpy(np.asarray(pi[s:s + chunk])).long().to(self.dev)].float()
            b = self.Q[torch.from_numpy(np.asarray(qi[s:s + chunk])).long().to(self.dev)].float()
            out[s:s + chunk] = (a * b).sum(1).cpu().numpy()
        return out

    def plant(self, p):
        with torch.no_grad():
            v = self.P[p].float()
            return (self.Q.float() @ v).cpu().numpy()


def main():
    store = UniverseStore(curves="modelled")
    split = json.load(open(ROOT / "data/splits/plants_75_10_15.json"))
    e = pd.read_parquet(ROOT / "data/network/edges.parquet")
    e = e[e.plant.isin(store.p2i) & e.pollinator.isin(store.q2i)]
    train = e[e.plant.isin(set(split["train"]))]
    test = e[e.plant.isin(set(split["val"]))]; test = test[test.tier == "A"]
    tp = sorted(set(test.plant))
    part = {sp: set(store.idx_polls(g.pollinator)) for sp, g in test.groupby("plant")}

    gen = np.array([s.split()[0] for s in store.plants])
    fam = np.array([store.family.get(s, "UNK") for s in store.plants])
    gi = {g: i for i, g in enumerate(sorted(set(gen)))}
    fi = {f: i for i, f in enumerate(sorted(set(fam)))}
    GI = np.array([gi[g] for g in gen]); FI = np.array([fi[f] for f in fam])
    Cg = np.zeros((len(store.polls), len(gi)), np.float32)
    Cf = np.zeros((len(store.polls), len(fi)), np.float32)
    for pl, po in zip(train.plant, train.pollinator):
        Cg[store.q2i[po], GI[store.p2i[pl]]] += 1
        Cf[store.q2i[po], FI[store.p2i[pl]]] += 1
    Fp, Pp = _pca(store)
    N = store.N_full

    def base(pi, qi):
        return np.hstack([
            np.log1p(store.Prs[qi])[:, None],
            (Cg[qi, GI[pi]] + 1e-3 * Cf[qi, FI[pi]])[:, None],
            np.log1p(np.asarray(N[pi, qi], dtype=np.float64))[:, None],
            Fp[pi] * Pp[qi], store.FC[pi], store.AC[qi]])

    ref = None
    for label, permute in [(None, None), ("+ per-cell co-activity", False),
                           ("  same, species-permuted", True)]:
        if label is None:
            f = base
            name = "ours_gbm (marginals only)"
        else:
            lo = LocalOverlap(store, permute=permute)
            f = lambda pi, qi, L=lo: np.hstack([base(pi, qi), np.log1p(L.pairs(pi, qi))[:, None]])
            name = label
        t0 = time.time()
        pp, r = run_row(store, f, train, tp, part, 42, 10, 1000)
        line = f"  {name:<28} nR@10 {r['nrecall@10']:.4f}  MAP {r['map']:.4f}  PR {r['pr_auc']:.4f}"
        if ref is not None:
            d, l, h, pv = paired_bootstrap(pp["nrecall@10"].to_numpy(), ref["nrecall@10"].to_numpy(), 2000, 42)
            line += f"   vs base {d:+.4f} [{l:+.4f},{h:+.4f}] p={pv:.3f}"
        print(line + f"  ({time.time()-t0:.0f}s)", flush=True)
        if ref is None:
            ref = pp
        if label is not None:
            del lo
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
