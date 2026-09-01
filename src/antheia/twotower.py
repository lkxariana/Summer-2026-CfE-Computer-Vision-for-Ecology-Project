import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from antheia.taxonomy import TaxonomyAffinity, family_map, genus

WIDE_DIM = 5  # log1p(N), delta, delta_local, tax genus share, tax family share


class Tower(nn.Module):
    """Dense features + genus/family embeddings -> unit-norm vector."""

    def __init__(self, in_dim, n_gen, n_fam, emb_dim=32, hid=256, out=128, p_drop=0.2):
        super().__init__()
        self.eg = nn.Embedding(n_gen, emb_dim)
        self.ef = nn.Embedding(n_fam, emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim + 2 * emb_dim, hid), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(hid, hid), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(hid, out))

    def forward(self, dense, gi, fi):
        h = torch.cat([dense, self.eg(gi), self.ef(fi)], -1)
        return F.normalize(self.mlp(h), dim=-1)


class WideDeep(nn.Module):
    """Two-tower ranker, optionally with a shallow observation-process head.

    The bias head consumes only species-level documentation-process descriptors (never the pair's
    ecology) and is ADDED to the score during training, then DISCARDED at inference — the shallow-tower
    construction from position-bias correction in recommenders, transplanted to documentation bias.
    """

    def __init__(self, p_dim, q_dim, n_pgen, n_pfam, n_qgen, n_qfam, bias_dim=0):
        super().__init__()
        self.pt = Tower(p_dim, n_pgen, n_pfam)
        self.qt = Tower(q_dim, n_qgen, n_qfam)
        self.wide = nn.Sequential(nn.Linear(WIDE_DIM, 64), nn.ReLU(), nn.Linear(64, 1))
        self.tau = nn.Parameter(torch.tensor(10.0))
        self.bias_head = (nn.Sequential(nn.Linear(bias_dim, 32), nn.ReLU(), nn.Linear(32, 1))
                          if bias_dim else None)

    def score(self, p_vec, q_vec, wide, bias_feats=None):
        s = self.tau * (p_vec * q_vec).sum(-1) + self.wide(wide).squeeze(-1)
        if self.bias_head is not None and bias_feats is not None:
            s = s + self.bias_head(bias_feats).squeeze(-1)
        return s


class Data:
    """Precomputes all tensors: tower inputs, per-positive negative pools, wide features, eval caches."""

    def __init__(self, store, edges, split, cfg, emb=None, pool_size=200, seed=42, proc=None):
        self.store = store
        rng = np.random.default_rng(seed)
        known = set(zip(edges["plant"], edges["pollinator"]))
        train_pos = edges[edges["plant"].isin(split["train"])].reset_index(drop=True)
        self.aff = TaxonomyAffinity(store, train_pos, cfg["paths"]["globi"])

        pfam = family_map(cfg["paths"]["globi"], set(store.plants))
        qfam = family_map(cfg["paths"]["globi"], set(store.polls))
        self.p_gi, self.p_fi, self.n_pgen, self.n_pfam = self._cats(store.plants, pfam)
        self.q_gi, self.q_fi, self.n_qgen, self.n_qfam = self._cats(store.polls, qfam)

        pd_blocks = [store.FC.astype(np.float32), np.log1p(store.Frs)[:, None].astype(np.float32)]
        qd_blocks = [store.AC.astype(np.float32), np.log1p(store.Prs)[:, None].astype(np.float32)]
        if emb is not None:
            pd_blocks.append(emb[0].astype(np.float32))
            qd_blocks.append(emb[1].astype(np.float32))
        self.p_dense = np.hstack(pd_blocks)
        self.q_dense = np.hstack(qd_blocks)

        self.pos_pi = store.idx_plants(train_pos["plant"])
        self.pos_qi = store.idx_polls(train_pos["pollinator"])
        n_po = len(store.polls)
        pools = np.empty((len(train_pos), pool_size), dtype=np.int64)
        cooc_cache = {}
        for i, (p, sp) in enumerate(zip(self.pos_pi, train_pos["plant"])):
            if p not in cooc_cache:
                cooc_cache[p] = np.flatnonzero(store.N_full[p] > 0)
            cooc = cooc_cache[p]
            half = pool_size // 2
            cand = np.concatenate([rng.integers(0, n_po, half + 20),
                                   cooc[rng.integers(0, len(cooc), half + 20)] if len(cooc) else
                                   rng.integers(0, n_po, half + 20)])
            keep = [q for q in cand if (sp, store.polls[q]) not in known][:pool_size]
            while len(keep) < pool_size:
                keep.append(int(rng.integers(0, n_po)))
            pools[i] = keep
        self.pools = pools
        self.pos_wide = self._wide(np.repeat(self.pos_pi, 1), self.pos_qi)
        flat_pi = np.repeat(self.pos_pi, pool_size)
        self.pool_wide = self._wide(flat_pi, pools.ravel()).reshape(len(train_pos), pool_size, WIDE_DIM)

        # Documentation-process descriptors, defined for EVERY species (so negatives are covered too).
        self.proc_p, self.proc_q = (proc if proc is not None else (None, None))
        self.bias_dim = 0 if proc is None else self.proc_p.shape[1] + self.proc_q.shape[1]
        if proc is not None:
            self.pos_bias = self._bias(self.pos_pi, self.pos_qi)
            self.pool_bias = self._bias(flat_pi, pools.ravel()).reshape(len(train_pos), pool_size, self.bias_dim)

    def _bias(self, pi, qi):
        return np.hstack([self.proc_p[pi], self.proc_q[qi]]).astype(np.float32)

    def _cats(self, names, fam):
        gens = sorted({genus(n) for n in names})
        fams = sorted({fam.get(n, "UNK") for n in names} | {"UNK"})
        g2i = {g: i for i, g in enumerate(gens)}
        f2i = {f: i for i, f in enumerate(fams)}
        gi = np.array([g2i[genus(n)] for n in names])
        fi = np.array([f2i[fam.get(n, "UNK")] for n in names])
        return gi, fi, len(gens), len(fams)

    def _wide(self, pi, qi):
        st = self.store
        n = np.log1p(st.N_full[pi, qi].astype(np.float32))[:, None]
        d = np.minimum(st.FC[pi], st.AC[qi]).sum(1).astype(np.float32)[:, None]
        dl = st.delta_local_pairs(pi, qi).astype(np.float32)[:, None]
        t = self.aff.pairs(pi, qi).astype(np.float32)[:, [1, 3]]
        return np.hstack([n, d, dl, t]).astype(np.float32)

    def wide_plant(self, p):
        st = self.store
        n = np.log1p(st.N_full[p].astype(np.float32))[:, None]
        d = np.minimum(st.FC[p][None, :], st.AC).sum(1).astype(np.float32)[:, None]
        masks = (st.P * st.F[p][None, :]).astype(np.float32)
        fl = masks @ st.surfaces[p].astype(np.float32)
        sums = fl.sum(1, keepdims=True)
        fl = np.divide(fl, sums, out=np.zeros_like(fl), where=sums > 0)
        dl = np.minimum(fl, st.AC).sum(1).astype(np.float32)[:, None]
        t = self.aff.plant(p).astype(np.float32)[:, [1, 3]]
        return np.hstack([n, d, dl, t]).astype(np.float32)


def train_model(data, device, seed=42, batch=512, k_neg=63, max_epochs=50, patience=6,
                lr=1e-3, val_eval=None, log=print):
    """Trains the wide&deep two-tower ranker with sampled-softmax over within-plant pools; returns (model, history)."""
    torch.manual_seed(seed)
    m = WideDeep(data.p_dense.shape[1], data.q_dense.shape[1],
                 data.n_pgen, data.n_pfam, data.n_qgen, data.n_qfam,
                 bias_dim=getattr(data, "bias_dim", 0)).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    P = torch.from_numpy(data.p_dense).to(device)
    Q = torch.from_numpy(data.q_dense).to(device)
    pgi = torch.from_numpy(data.p_gi).to(device)
    pfi = torch.from_numpy(data.p_fi).to(device)
    qgi = torch.from_numpy(data.q_gi).to(device)
    qfi = torch.from_numpy(data.q_fi).to(device)
    pos_pi = torch.from_numpy(data.pos_pi).to(device)
    pos_qi = torch.from_numpy(data.pos_qi).to(device)
    pools = torch.from_numpy(data.pools).to(device)
    pos_w = torch.from_numpy(data.pos_wide).to(device)
    pool_w = torch.from_numpy(data.pool_wide).to(device)
    use_bias = getattr(data, "bias_dim", 0) > 0
    if use_bias:
        pos_b = torch.from_numpy(data.pos_bias).to(device)
        pool_b = torch.from_numpy(data.pool_bias).to(device)
    n_pos, pool_size = pools.shape
    rng = np.random.default_rng(seed)
    best, best_state, bad, hist = -1.0, None, 0, []

    for ep in range(max_epochs):
        m.train()
        order = torch.from_numpy(rng.permutation(n_pos)).to(device)
        tot = 0.0
        for s in range(0, n_pos, batch):
            idx = order[s:s + batch]
            cols = torch.from_numpy(rng.integers(0, pool_size, (len(idx), k_neg))).to(device)
            qneg = torch.gather(pools[idx], 1, cols)
            qcand = torch.cat([pos_qi[idx, None], qneg], 1)                     # B x (1+K)
            wide = torch.cat([pos_w[idx, None, :], torch.gather(
                pool_w[idx], 1, cols[..., None].expand(-1, -1, WIDE_DIM))], 1)  # B x (1+K) x 5
            pv = m.pt(P[pos_pi[idx]], pgi[pos_pi[idx]], pfi[pos_pi[idx]])
            qv = m.qt(Q[qcand.reshape(-1)], qgi[qcand.reshape(-1)],
                      qfi[qcand.reshape(-1)]).view(len(idx), 1 + k_neg, -1)
            bfeat = None
            if use_bias:
                bfeat = torch.cat([pos_b[idx, None, :], torch.gather(
                    pool_b[idx], 1, cols[..., None].expand(-1, -1, data.bias_dim))], 1)
            logits = m.score(pv[:, None, :], qv, wide, bfeat)
            loss = F.cross_entropy(logits, torch.zeros(len(idx), dtype=torch.long, device=device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        msg = f"epoch {ep} loss {tot / n_pos:.4f}"
        if val_eval is not None:
            r10 = val_eval(m)
            hist.append(r10)
            msg += f" val R@10 {r10:.4f}"
            if r10 > best:
                best, bad = r10, 0
                best_state = {k: v.detach().clone() for k, v in m.state_dict().items()}
            else:
                bad += 1
        log(msg)
        if bad >= patience:
            break
    if best_state is not None:
        m.load_state_dict(best_state)
    return m, hist


def make_ranker(m, data, device):
    """Returns scorer(plant_index, wide_block) -> score vector over all pollinators, using cached tower outputs."""
    m.eval()
    with torch.no_grad():
        Q = torch.from_numpy(data.q_dense).to(device)
        qv = torch.cat([m.qt(Q[i:i + 4096], torch.from_numpy(data.q_gi[i:i + 4096]).to(device),
                             torch.from_numpy(data.q_fi[i:i + 4096]).to(device))
                        for i in range(0, len(data.q_dense), 4096)])

    def scorer(p, wide_np):
        with torch.no_grad():
            pv = m.pt(torch.from_numpy(data.p_dense[p:p + 1]).to(device),
                      torch.from_numpy(data.p_gi[p:p + 1]).to(device),
                      torch.from_numpy(data.p_fi[p:p + 1]).to(device))
            wide = torch.from_numpy(wide_np).to(device)
            s = m.tau * (qv @ pv[0]) + m.wide(wide).squeeze(-1)
        return s.cpu().numpy()

    return scorer
