"""Table 3 Panel C: explicit taxonomy against a learned species representation.

Taxonomic affinity is the strongest single signal in this network, and it is also the most brittle
form identity can take: a genus absent from training is a row of zeros, and nothing carries over from
its relatives beyond a coarse family fallback. A species-name embedding is the implicit counterpart --
BioCLIP-2 places related taxa near each other whether or not they co-occur in the training edges.

The implicit row scores a pair by how close the plant sits to the *prototype* of the plants a
pollinator is recorded visiting: the mean unit text vector of its training partners. That is the
text-space analogue of congeneric transfer, and unlike the affinity table it degrades gracefully --
an unseen genus still lands near its relatives.

Prompts are the bare binomial, with no order or family string, so the embedding cannot smuggle in the
taxonomy the explicit row is built from.
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


def unit(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + EPS)


def main():
    store = UniverseStore(curves="modelled")
    split = json.load(open(ROOT / "data/splits/plants_75_10_15.json"))
    e = pd.read_parquet(ROOT / "data/network/edges.parquet")
    e = e[e.plant.isin(store.p2i) & e.pollinator.isin(store.q2i)]
    train = e[e.plant.isin(set(split["train"]))]
    test = e[e.plant.isin(set(split["val"]))]; test = test[test.tier == "A"]
    tp = sorted(set(test.plant))
    part = {sp: set(store.idx_polls(g.pollinator)) for sp, g in test.groupby("plant")}

    TP = unit(torch.load(TEXT / "plants_bioclip2.pt", weights_only=False)["embeddings"].numpy().astype(np.float32))
    TQ = unit(torch.load(TEXT / "polls_bioclip2.pt", weights_only=False)["embeddings"].numpy().astype(np.float32))

    gen = np.array([s.split()[0] for s in store.plants])
    fam = np.array([store.family.get(s, "UNK") for s in store.plants])
    gi = {g: i for i, g in enumerate(sorted(set(gen)))}
    fi = {f: i for i, f in enumerate(sorted(set(fam)))}
    GI = np.array([gi[g] for g in gen]); FI = np.array([fi[f] for f in fam])
    Cg = np.zeros((len(store.polls), len(gi)), np.float32)
    Cf = np.zeros((len(store.polls), len(fi)), np.float32)
    pi_tr = store.idx_plants(train.plant); qi_tr = store.idx_polls(train.pollinator)
    np.add.at(Cg, (qi_tr, GI[pi_tr]), 1.0)
    np.add.at(Cf, (qi_tr, FI[pi_tr]), 1.0)

    def prototypes(text_p, pi, qi):
        """Mean unit text vector of each pollinator's training partner plants."""
        P = np.zeros((len(store.polls), text_p.shape[1]), np.float32)
        np.add.at(P, qi, text_p[pi])
        return unit(P)

    proto = prototypes(TP, pi_tr, qi_tr)
    rng = np.random.default_rng(0)
    proto_perm = proto[rng.permutation(len(proto))]

    pop = lambda p, q: np.log1p(store.Prs[q])[:, None]
    comp = lambda p, q: (Cg[q, GI[p]] + 1e-3 * Cf[q, FI[p]])[:, None]
    txt = lambda p, q, M=proto: (TP[p] * M[q]).sum(1)[:, None]
    Fp, Pp = _pca(store)
    N = store.N_full
    other = lambda p, q: np.hstack([
        np.log1p(np.asarray(N[p, q], dtype=np.float64))[:, None], Fp[p] * Pp[q],
        store.FC[p], store.AC[q], np.log1p(np.maximum(store.local_overlap(p, q), 0))[:, None]])

    rows = [
        ("explicit", "taxonomic affinity (genus, family)", lambda p, q: np.hstack([pop(p, q), comp(p, q)])),
        ("implicit", "species-name embedding prototype", lambda p, q: np.hstack([pop(p, q), txt(p, q)])),
        ("control", "  same, species-permuted", lambda p, q: np.hstack([pop(p, q), txt(p, q, proto_perm)])),
        ("both", "affinity + name embedding", lambda p, q: np.hstack([pop(p, q), comp(p, q), txt(p, q)])),
        ("full", "ours + name embedding", lambda p, q: np.hstack([pop(p, q), comp(p, q), other(p, q), txt(p, q)])),
        ("full", "ours (reference)", lambda p, q: np.hstack([pop(p, q), comp(p, q), other(p, q)])),
    ]
    out, ref = [], {}
    for kind, label, f in rows:
        t0 = time.time()
        pp, r = run_row(store, f, train, tp, part, 42, 10, 1000)
        line = f"  {kind:<9} {label:<36} nR@10 {r['nrecall@10']:.4f}  MAP {r['map']:.4f}  PR {r['pr_auc']:.4f}"
        if label.startswith("  same") and "implicit" in ref:
            d, lo, hi, pv = paired_bootstrap(ref["implicit"]["nrecall@10"].to_numpy(),
                                             pp["nrecall@10"].to_numpy(), 2000, 42)
            line += f"   real margin {d:+.4f} p={pv:.3f}"
        print(line + f"  ({time.time()-t0:.0f}s)", flush=True)
        ref[kind] = pp
        out.append(dict(kind=kind, representation=label, **{k: v for k, v in r.items() if k != "lo"}))
    if "full" in ref:
        pass
    pd.DataFrame(out).to_csv(ROOT / "results/identity_panel_val_tierA.csv", index=False)
    print(f"[wrote] results/identity_panel_val_tierA.csv")


if __name__ == "__main__":
    main()
