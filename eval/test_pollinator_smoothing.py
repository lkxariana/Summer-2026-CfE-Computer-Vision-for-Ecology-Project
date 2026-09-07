"""Kernel-smoothed taxonomic affinity on the pollinator side.

Taxonomic affinity is the strongest signal available, and it is applied asymmetrically. The plant
side is smoothed -- a held-out plant inherits the interactions of its genus -- while the pollinator
side is an identity lookup: Cg[q, genus(p)] counts only the edges of pollinator q itself. Half the
pollinators in the training set have fewer than three edges, so that row is almost empty for exactly
the taxa cold-start retrieval misses.

BioCLIP-2 text embeddings supply the missing smoother. They place congeners close (same-genus cosine
0.716 against 0.412 for random pairs) without being told the taxonomy -- prompts are bare binomials.
Affinity is propagated across the k nearest pollinators in that space,

    A_smooth[q, g] = sum_q' k(q, q') * C[q', g] / sum_q' k(q, q'),

which is Nadaraya-Watson regression on the pollinator manifold. A pollinator with one recorded edge
now inherits an affinity profile from its relatives.

Controls: a species-permuted neighbour graph, which keeps the smoothing arithmetic and destroys the
taxonomy, and a random-neighbour graph.
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
from antheia.metrics import paired_bootstrap
from antheia.store import UniverseStore
from run_encoding_ablation import run_row, _pca

TEXT = Path("/scratch/cher/antheia-data/text_embeddings")
EPS = 1e-12


def knn_smoother(T, k, device, permute_rng=None):
    """Row-stochastic [Q, Q] smoother from the top-k cosine neighbours, self excluded."""
    X = torch.from_numpy(T).to(device)
    X = X / (X.norm(dim=1, keepdim=True) + EPS)
    if permute_rng is not None:
        X = X[torch.from_numpy(permute_rng.permutation(len(X))).to(device)]
    n = len(X)
    idx = torch.empty(n, k, dtype=torch.long, device=device)
    val = torch.empty(n, k, device=device)
    for s in range(0, n, 1024):                                   # chunked to bound memory
        sim = X[s:s + 1024] @ X.T                                 # [b, Q]
        sim[torch.arange(len(sim), device=device), torch.arange(s, s + len(sim), device=device)] = -2
        v, i = torch.topk(sim, k, dim=1)
        idx[s:s + 1024], val[s:s + 1024] = i, v
    val = torch.clamp(val, min=0.0)
    val = val / (val.sum(1, keepdim=True) + EPS)
    return idx, val


def smooth(C, idx, val, device):
    """A_smooth[q] = sum_j val[q,j] * C[idx[q,j]]  -> [Q, G]."""
    Ct = torch.from_numpy(C).to(device)
    out = torch.zeros_like(Ct)
    for s in range(0, len(Ct), 2048):
        nb = Ct[idx[s:s + 2048]]                                  # [b, k, G]
        out[s:s + 2048] = (nb * val[s:s + 2048].unsqueeze(-1)).sum(1)
    return out.cpu().numpy()


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
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
    g2i = {g: i for i, g in enumerate(sorted(set(gen)))}
    f2i = {f: i for i, f in enumerate(sorted(set(fam)))}
    GI = np.array([g2i[g] for g in gen]); FI = np.array([f2i[f] for f in fam])
    pi = store.idx_plants(train.plant); qi = store.idx_polls(train.pollinator)
    Cg = np.zeros((len(store.polls), len(g2i)), np.float32)
    Cf = np.zeros((len(store.polls), len(f2i)), np.float32)
    np.add.at(Cg, (qi, GI[pi]), 1.0)
    np.add.at(Cf, (qi, FI[pi]), 1.0)
    deg = np.bincount(qi, minlength=len(store.polls))
    print(f"[data] {int((deg < 3).sum())}/{len(store.polls)} pollinators have fewer than 3 training edges",
          flush=True)

    TQ = torch.load(TEXT / "polls_bioclip2.pt", weights_only=False)["embeddings"].numpy().astype(np.float32)
    Fp, Pp = _pca(store)
    N = store.N_full
    base = lambda p, q: np.hstack([
        np.log1p(store.Prs[q])[:, None],
        (Cg[q, GI[p]] + 1e-3 * Cf[q, FI[p]])[:, None],
        np.log1p(np.asarray(N[p, q], dtype=np.float64))[:, None],
        Fp[p] * Pp[q], store.FC[p], store.AC[q],
        np.log1p(np.maximum(store.local_overlap(p, q), 0))[:, None]])

    variants = [("reference (no pollinator smoothing)", None, None)]
    for k in (10, 25, 50):
        variants.append((f"+ smoothed affinity, k={k}", k, None))
    variants.append(("  same, permuted neighbour graph", 25, np.random.default_rng(0)))

    ref, rows = None, []
    for label, k, perm in variants:
        if k is None:
            f = base
        else:
            idx, val = knn_smoother(TQ, k, dev, permute_rng=perm)
            Sg = smooth(Cg, idx, val, dev)
            Sf = smooth(Cf, idx, val, dev)
            f = lambda p, q, A=Sg, B=Sf: np.hstack([base(p, q), (A[q, GI[p]] + 1e-3 * B[q, FI[p]])[:, None]])
        t0 = time.time()
        pp, r = run_row(store, f, train, tp, part, 42, 10, 1000)
        line = f"  {label:<38} nR@10 {r['nrecall@10']:.4f}  MAP {r['map']:.4f}  PR {r['pr_auc']:.4f}"
        if ref is not None:
            d, lo, hi, pv = paired_bootstrap(pp["nrecall@10"].to_numpy(), ref["nrecall@10"].to_numpy(), 2000, 42)
            line += f"   vs ref {d:+.4f} [{lo:+.4f},{hi:+.4f}] p={pv:.3f}"
        print(line + f"  ({time.time() - t0:.0f}s)", flush=True)
        rows.append(dict(config=label, k=k, **{a: b for a, b in r.items() if a != "lo"}))
        if ref is None:
            ref = pp
    pd.DataFrame(rows).to_csv(ROOT / "results/pollinator_smoothing_val.csv", index=False)
    print("[wrote] results/pollinator_smoothing_val.csv")


if __name__ == "__main__":
    main()
