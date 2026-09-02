import argparse
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as Fn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from antheia.config import load_config
from antheia.localmatch import LocalMatch, sample_shared_bins
from antheia.metrics import bootstrap_mean
from antheia.pairs import load_split
from antheia.store import FeatureStore
from antheia.taxonomy import TaxonomyAffinity, family_map, genus
from antheia.twotower import Tower

# Symmetric ladder, restricted to SDM-covered pollinators so BOTH sides have a
# (location x week) surface. Each arm adds one level of temporal representation.
ARMS = ["spatial", "curves", "curves_sdm", "global_delta", "global_delta_sdm",
        "local_delta_hand", "local_learned"]
N_BINS = 8


class Model(torch.nn.Module):
    def __init__(self, p_dim, q_dim, ng_p, nf_p, ng_q, nf_q, n_wide, local_learned):
        super().__init__()
        self.pt, self.qt = Tower(p_dim, ng_p, nf_p), Tower(q_dim, ng_q, nf_q)
        self.tau = torch.nn.Parameter(torch.tensor(10.0))
        self.wide = torch.nn.Sequential(torch.nn.Linear(n_wide, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))
        self.local = LocalMatch() if local_learned else None

    def forward(self, pv, qv, wide, pc=None, qc=None, mask=None):
        s = self.tau * (pv * qv).sum(-1) + self.wide(wide).squeeze(-1)
        if self.local is not None and pc is not None:
            s = s + self.local(pc, qc, mask)
        return s


def main():
    ap = argparse.ArgumentParser(description="Symmetric bilateral temporal ladder on the SDM universe")
    ap.add_argument("--config", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seeds", default="42,0")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--tag", default="bilateral_ladder_v2")
    args = ap.parse_args()

    cfg = load_config(args.config)
    store = FeatureStore(cfg)
    edges = pd.read_parquet(cfg["edges"])
    split = load_split(cfg["edges"].parent / "split_v1.json")
    seeds = [int(x) for x in args.seeds.split(",")]
    nboot = cfg["eval"]["bootstrap_n"]
    dev = args.device
    t0 = time.time()

    sdm_names = list(np.load(cfg["cache_dir"] / "sdm_species.npy", allow_pickle=True))
    sdm_surf = np.load(cfg["cache_dir"] / "sdm_surfaces_1275x3162x52.npy", mmap_mode="r")
    keep = [(i, store.q2i[s]) for i, s in enumerate(sdm_names) if s in store.q2i]
    sdm_rows = np.array([a for a, _ in keep])
    poll_idx = np.array([b for _, b in keep])                    # store index of each universe member
    n_q = len(poll_idx)
    print(f"restricted universe: {n_q} pollinators ({time.time()-t0:.0f}s)", flush=True)

    Q_SURF = np.asarray(sdm_surf[sdm_rows], dtype=np.float16)    # (n_q, bins, 52)
    P_SURF = store.surfaces                                       # memmap (n_plants, bins, 52)
    F_bin = store.F.astype(bool)
    P_bin = store.P[poll_idx].astype(bool)

    edges = edges[edges["pollinator"].isin({store.polls[i] for i in poll_idx})]
    q_pos = {store.polls[q]: j for j, q in enumerate(poll_idx)}
    train_pos = edges[edges["plant"].isin(split["train"])].reset_index(drop=True)
    test_pos = edges[edges["plant"].isin(split["test"])]
    print(f"edges in universe: train {len(train_pos):,} test {len(test_pos):,}", flush=True)

    aff = TaxonomyAffinity(store, train_pos, cfg["paths"]["globi"])
    pfam, qfam = family_map(cfg["paths"]["globi"], set(store.plants)), family_map(cfg["paths"]["globi"], set(store.polls))
    cat = lambda names, fam: (lambda g, f: (np.array([{x: i for i, x in enumerate(sorted(set(g)))}[v] for v in g]),
                                            np.array([{x: i for i, x in enumerate(sorted(set(f)))}[v] for v in f]),
                                            len(set(g)), len(set(f))))([genus(n) for n in names],
                                                                       [fam.get(n, "UNK") for n in names])
    p_gi, p_fi, ngp, nfp = cat(store.plants, pfam)
    q_gi, q_fi, ngq, nfq = cat([store.polls[i] for i in poll_idx], qfam)

    emb_p = np.load(cfg["cache_dir"] / "bioclip_text_plants.npy").astype(np.float32)
    emb_q = np.load(cfg["cache_dir"] / "bioclip_text_polls.npy").astype(np.float32)[poll_idx]
    FC, AC = store.FC.astype(np.float32), store.AC.astype(np.float32)[poll_idx]
    # SDM curves marginalized over space: the same granularity as GBIF a_curves, different source.
    AC_SDM = Q_SURF.astype(np.float32).sum(1)
    AC_SDM /= np.clip(AC_SDM.sum(1, keepdims=True), 1e-9, None)
    N_sub = store.N_full[:, poll_idx].astype(np.float32)
    Frs, Prs = store.Frs.astype(np.float32), store.Prs.astype(np.float32)[poll_idx]

    def local_delta_hand(pi, qj):
        """Hand-computed bilateral per-bin overlap: mean over shared bins of sum-min."""
        out = np.zeros(len(pi), np.float32)
        for k, (p, q) in enumerate(zip(pi, qj)):
            sh = np.flatnonzero(F_bin[p] & P_bin[q])
            if not len(sh):
                continue
            a = np.asarray(P_SURF[p][sh], np.float32)
            b = Q_SURF[q][sh].astype(np.float32)
            a /= np.clip(a.sum(1, keepdims=True), 1e-9, None)
            b /= np.clip(b.sum(1, keepdims=True), 1e-9, None)
            out[k] = np.minimum(a, b).sum(1).mean()
        return out

    def wide_block(pi, qj, arm):
        b = [np.log1p(N_sub[pi, qj])[:, None], aff.pairs(pi, np.array([poll_idx[j] for j in qj]))[:, [1, 3]].astype(np.float32)]
        if arm == "global_delta":
            b.append(np.minimum(FC[pi], AC[qj]).sum(1)[:, None].astype(np.float32))
        elif arm == "global_delta_sdm":
            b.append(np.minimum(FC[pi], AC_SDM[qj]).sum(1)[:, None].astype(np.float32))
        elif arm == "local_delta_hand":
            b.append(local_delta_hand(pi, qj)[:, None])
        return np.hstack(b).astype(np.float32)

    partners = {sp: {q_pos[x] for x in g["pollinator"]} for sp, g in test_pos.groupby("plant")}
    plant_order = list(partners)
    pos_pi = store.idx_plants(train_pos["plant"])
    pos_qj = np.array([q_pos[s] for s in train_pos["pollinator"]])
    known = set(zip(pos_pi.tolist(), pos_qj.tolist()))

    rows, per_plant = [], {}
    for arm in ARMS:
        n_wide = 3 + (1 if arm in ("global_delta", "global_delta_sdm", "local_delta_hand") else 0)
        use_curves = arm != "spatial"
        q_curve = AC_SDM if arm == "curves_sdm" else AC
        pdense = np.hstack([FC, np.log1p(Frs)[:, None], emb_p]) if use_curves else np.hstack([np.log1p(Frs)[:, None], emb_p])
        qdense = np.hstack([q_curve, np.log1p(Prs)[:, None], emb_q]) if use_curves else np.hstack([np.log1p(Prs)[:, None], emb_q])
        seed_r = []
        tv = time.time()
        for sd in seeds:
            rng = np.random.default_rng(sd)
            torch.manual_seed(sd)
            m = Model(pdense.shape[1], qdense.shape[1], ngp, nfp, ngq, nfq, n_wide,
                      arm == "local_learned").to(dev)
            opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
            P = torch.from_numpy(pdense.astype(np.float32)).to(dev)
            Q = torch.from_numpy(qdense.astype(np.float32)).to(dev)
            tg = {k: torch.from_numpy(v).to(dev) for k, v in
                  dict(pg=p_gi, pf=p_fi, qg=q_gi, qf=q_fi).items()}
            for ep in range(args.epochs):
                order = rng.permutation(len(pos_pi))
                for s in range(0, len(order), 256):
                    idx = order[s:s + 256]
                    pi = pos_pi[idx]
                    negs = rng.integers(0, n_q, (len(idx), 15))
                    qc = np.concatenate([pos_qj[idx][:, None], negs], 1)
                    flat_p = np.repeat(pi, qc.shape[1])
                    flat_q = qc.ravel()
                    w = torch.from_numpy(wide_block(flat_p, flat_q, arm)).to(dev).view(len(idx), qc.shape[1], n_wide)
                    pv = m.pt(P[pi], tg["pg"][pi], tg["pf"][pi])[:, None, :]
                    qt = torch.from_numpy(flat_q).to(dev)
                    qv = m.qt(Q[qt], tg["qg"][qt], tg["qf"][qt]).view(len(idx), qc.shape[1], -1)
                    pc = qcv = mk = None
                    if m.local is not None:
                        bi, mask = sample_shared_bins(F_bin, P_bin, flat_p, flat_q, N_BINS, rng)
                        pc = torch.from_numpy(np.asarray(P_SURF[flat_p[:, None], bi], np.float32)).to(dev).view(len(idx), qc.shape[1], N_BINS, 52)
                        qcv = torch.from_numpy(Q_SURF[flat_q[:, None], bi].astype(np.float32)).to(dev).view(len(idx), qc.shape[1], N_BINS, 52)
                        mk = torch.from_numpy(mask).to(dev).view(len(idx), qc.shape[1], N_BINS)
                    logits = m(pv, qv, w, pc, qcv, mk)
                    loss = Fn.cross_entropy(logits, torch.zeros(len(idx), dtype=torch.long, device=dev))
                    opt.zero_grad(); loss.backward(); opt.step()
            m.eval()
            r10 = []
            with torch.no_grad():
                allq = torch.arange(n_q, device=dev)
                qv_all = m.qt(Q, tg["qg"], tg["qf"])
                for sp in plant_order:
                    p = store.p2i[sp]
                    pv = m.pt(P[p:p + 1], tg["pg"][p:p + 1], tg["pf"][p:p + 1])[0]
                    pj = np.full(n_q, p)
                    qj = np.arange(n_q)
                    w = torch.from_numpy(wide_block(pj, qj, arm)).to(dev)
                    sc = m.tau * (qv_all @ pv) + m.wide(w).squeeze(-1)
                    if m.local is not None:
                        bi, mask = sample_shared_bins(F_bin, P_bin, pj, qj, N_BINS, rng)
                        pc = torch.from_numpy(np.asarray(P_SURF[pj[:, None], bi], np.float32)).to(dev)
                        qcv = torch.from_numpy(Q_SURF[qj[:, None], bi].astype(np.float32)).to(dev)
                        sc = sc + m.local(pc, qcv, torch.from_numpy(mask).to(dev))
                    s = sc.cpu().numpy()
                    top = set(np.argpartition(-s, 10)[:10].tolist())
                    r10.append(len(partners[sp] & top) / len(partners[sp]))
            seed_r.append(np.array(r10))
            print(f"  [{arm}] seed {sd} R@10 {seed_r[-1].mean():.4f} ({time.time()-tv:.0f}s)", flush=True)
        per_plant[arm] = np.mean(seed_r, axis=0)
        mm, lo, hi, _ = bootstrap_mean(per_plant[arm], nboot, seeds[0])
        rows.append({"arm": arm, "recall@10": mm, "lo": lo, "hi": hi})
        print(f"  {arm:<18} R@10 {mm:.4f} [{lo:.4f},{hi:.4f}]", flush=True)

    # Baselines in the SAME restricted universe.
    tr_pi, tr_qj = pos_pi, pos_qj
    p_gen = np.array([genus(s) for s in store.plants])
    gen_mat = {}
    for g, q in zip(p_gen[tr_pi], tr_qj):
        gen_mat.setdefault(g, np.zeros(n_q))[q] += 1
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import svds
    A = csr_matrix((np.ones(len(tr_pi)), (tr_pi, tr_qj)), shape=(len(store.plants), n_q))
    U, S, Vt = svds(A.astype(np.float64), k=32)
    P_lat, Q_lat = (U * S)[:, ::-1], Vt[::-1].T
    seen = np.zeros(len(store.plants), bool); seen[np.unique(tr_pi)] = True
    gl = {g: P_lat[seen & (p_gen == g)].mean(0) for g in np.unique(p_gen[seen])}
    glob = P_lat[seen].mean(0)
    base = {"N only": lambda p: N_sub[p],
            "congeneric transfer": lambda p: gen_mat.get(p_gen[p], np.zeros(n_q)),
            "latent-trait SVD": lambda p: Q_lat @ gl.get(p_gen[p], glob)}
    for nm, fn in base.items():
        v = []
        for sp in plant_order:
            p = store.p2i[sp]
            s = np.asarray(fn(p), float) + np.random.default_rng(p).uniform(0, 1e-9, n_q)
            v.append(len(partners[sp] & set(np.argpartition(-s, 10)[:10].tolist())) / len(partners[sp]))
        per_plant[nm] = np.array(v)
        mm, lo, hi, _ = bootstrap_mean(per_plant[nm], nboot, seeds[0])
        rows.append({"arm": nm, "recall@10": mm, "lo": lo, "hi": hi})
        print(f"  {nm:<18} R@10 {mm:.4f} [{lo:.4f},{hi:.4f}]", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(cfg["edges"].parent / f"{args.tag}.csv", index=False)
    print("\n" + df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    rng = np.random.default_rng(seeds[0])
    ix = rng.integers(0, len(plant_order), size=(10000, len(plant_order)))
    print("\npaired bootstrap vs spatial:")
    for a in [x for x in per_plant if x != "spatial"]:
        d = per_plant[a] - per_plant["spatial"]
        bs = d[ix].mean(1)
        pv = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
        print(f"  {a:<20} {d.mean():+.4f} [{np.percentile(bs,2.5):+.4f},{np.percentile(bs,97.5):+.4f}] p={max(pv,1e-4):.4f}")


if __name__ == "__main__":
    main()
