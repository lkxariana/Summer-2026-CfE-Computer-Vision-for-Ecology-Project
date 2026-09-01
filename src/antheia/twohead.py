import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from antheia.twotower import WIDE_DIM, Tower

CTX_DIM = 2  # log plant range size, flowering-curve breadth (entropy)


def plant_context(store):
    """Per-plant scalars that are constant across candidates: range size and flowering breadth."""
    fc = store.FC / np.clip(store.FC.sum(1, keepdims=True), 1e-12, None)
    ent = -(fc * np.log(np.clip(fc, 1e-12, None))).sum(1) / np.log(fc.shape[1])
    return np.stack([np.log1p(store.Frs), ent], 1).astype(np.float32)


class TwoHead(nn.Module):
    """Shared towers, two objective heads.

    Both heads share the plant/pollinator encoders and the unit-norm dot product, so the
    representation is trained by both objectives. They differ in what else they may use:

    - `head_r` (retrieval) sees only pair features that VARY across candidates for a fixed plant.
    - `head_c` (compatibility) additionally sees plant-only context, which is constant within a
      plant and therefore cannot affect a within-plant ranking at all. This is the two-objective
      asymmetry made architectural: plant-side signal can only ever move the pooled objective.
    """

    def __init__(self, p_dim, q_dim, n_pgen, n_pfam, n_qgen, n_qfam):
        super().__init__()
        self.pt = Tower(p_dim, n_pgen, n_pfam)
        self.qt = Tower(q_dim, n_qgen, n_qfam)
        self.tau = nn.Parameter(torch.tensor(10.0))
        self.head_r = nn.Sequential(nn.Linear(WIDE_DIM, 64), nn.ReLU(), nn.Linear(64, 1))
        self.head_c = nn.Sequential(nn.Linear(WIDE_DIM + CTX_DIM, 64), nn.ReLU(), nn.Linear(64, 1))

    def interaction(self, p_vec, q_vec):
        return self.tau * (p_vec * q_vec).sum(-1)

    def score_r(self, p_vec, q_vec, wide):
        return self.interaction(p_vec, q_vec) + self.head_r(wide).squeeze(-1)

    def score_c(self, p_vec, q_vec, wide, ctx):
        return self.interaction(p_vec, q_vec) + self.head_c(torch.cat([wide, ctx], -1)).squeeze(-1)


def mask_delta(wide, mode):
    """wide = [log N, delta, delta_local, tax_genus, tax_family]; zero the temporal columns per ablation."""
    if mode == "both":
        return wide
    w = wide.clone()
    if mode in ("none", "local"):
        w[..., 1] = 0.0
    if mode in ("none", "global"):
        w[..., 2] = 0.0
    return w


def train_twohead(data, ctx, device, seed=42, batch=512, k_neg=63, max_epochs=40, patience=6,
                  lr=1e-3, lam_rank=1.0, lam_comp=1.0, delta_mode="both", val_eval=None, log=print):
    """Joint training: sampled softmax (retrieval head) + BCE (compatibility head) on the same candidates."""
    torch.manual_seed(seed)
    m = TwoHead(data.p_dense.shape[1], data.q_dense.shape[1],
                data.n_pgen, data.n_pfam, data.n_qgen, data.n_qfam).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    P = torch.from_numpy(data.p_dense).to(device)
    Q = torch.from_numpy(data.q_dense).to(device)
    CTX = torch.from_numpy(ctx).to(device)
    pgi, pfi = (torch.from_numpy(data.p_gi).to(device), torch.from_numpy(data.p_fi).to(device))
    qgi, qfi = (torch.from_numpy(data.q_gi).to(device), torch.from_numpy(data.q_fi).to(device))
    pos_pi = torch.from_numpy(data.pos_pi).to(device)
    pos_qi = torch.from_numpy(data.pos_qi).to(device)
    pools = torch.from_numpy(data.pools).to(device)
    pos_w = torch.from_numpy(data.pos_wide).to(device)
    pool_w = torch.from_numpy(data.pool_wide).to(device)
    n_pos, pool_size = pools.shape
    rng = np.random.default_rng(seed)
    best, best_state, bad, hist = -1e9, None, 0, []

    for ep in range(max_epochs):
        m.train()
        order = torch.from_numpy(rng.permutation(n_pos)).to(device)
        tot_r = tot_c = 0.0
        for s in range(0, n_pos, batch):
            idx = order[s:s + batch]
            cols = torch.from_numpy(rng.integers(0, pool_size, (len(idx), k_neg))).to(device)
            qcand = torch.cat([pos_qi[idx, None], torch.gather(pools[idx], 1, cols)], 1)
            wide = torch.cat([pos_w[idx, None, :], torch.gather(
                pool_w[idx], 1, cols[..., None].expand(-1, -1, WIDE_DIM))], 1)
            wide = mask_delta(wide, delta_mode)
            pv = m.pt(P[pos_pi[idx]], pgi[pos_pi[idx]], pfi[pos_pi[idx]])
            qv = m.qt(Q[qcand.reshape(-1)], qgi[qcand.reshape(-1)],
                      qfi[qcand.reshape(-1)]).view(len(idx), 1 + k_neg, -1)
            pvb = pv[:, None, :]

            logits_r = m.score_r(pvb, qv, wide)
            loss_r = F.cross_entropy(logits_r, torch.zeros(len(idx), dtype=torch.long, device=device))

            ctx_b = CTX[pos_pi[idx]][:, None, :].expand(-1, 1 + k_neg, -1)
            logits_c = m.score_c(pvb, qv, wide, ctx_b)
            labels = torch.zeros_like(logits_c)
            labels[:, 0] = 1.0
            loss_c = F.binary_cross_entropy_with_logits(logits_c, labels)

            loss = lam_rank * loss_r + lam_comp * loss_c
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_r += float(loss_r) * len(idx)
            tot_c += float(loss_c) * len(idx)

        msg = f"epoch {ep} rank {tot_r / n_pos:.4f} comp {tot_c / n_pos:.4f}"
        if val_eval is not None:
            score, detail = val_eval(m)
            hist.append(detail)
            msg += f" | {detail}"
            if score > best:
                best, bad = score, 0
                best_state = {k: v.detach().clone() for k, v in m.state_dict().items()}
            else:
                bad += 1
        log(msg)
        if bad >= patience:
            break
    if best_state is not None:
        m.load_state_dict(best_state)
    return m, hist


def make_scorers(m, data, ctx, device, delta_mode="both"):
    """Returns (rank_scorer, comp_scorer): plant_index, wide_block -> scores over all pollinators."""
    m.eval()
    with torch.no_grad():
        Q = torch.from_numpy(data.q_dense).to(device)
        qv = torch.cat([m.qt(Q[i:i + 4096], torch.from_numpy(data.q_gi[i:i + 4096]).to(device),
                             torch.from_numpy(data.q_fi[i:i + 4096]).to(device))
                        for i in range(0, len(data.q_dense), 4096)])

    def _both(p, wide_np):
        with torch.no_grad():
            pv = m.pt(torch.from_numpy(data.p_dense[p:p + 1]).to(device),
                      torch.from_numpy(data.p_gi[p:p + 1]).to(device),
                      torch.from_numpy(data.p_fi[p:p + 1]).to(device))[0]
            wide = mask_delta(torch.from_numpy(wide_np).to(device), delta_mode)
            inter = m.tau * (qv @ pv)
            c = torch.from_numpy(ctx[p:p + 1]).to(device).expand(len(qv), -1)
            r = inter + m.head_r(wide).squeeze(-1)
            comp = inter + m.head_c(torch.cat([wide, c], -1)).squeeze(-1)
        return r.cpu().numpy(), comp.cpu().numpy()

    return (lambda p, w: _both(p, w)[0]), (lambda p, w: _both(p, w)[1])
