"""A neural pair ranker: the booster's features, learned embeddings, and the two objectives.

The two-tower plateaus near 0.28 nrecall@10 on this data whatever it is given -- marginal curves,
per-cell surfaces through a random basis, the same through a fitted one -- while a gradient-boosted
ranker on the same information reaches 0.326. The gap is the dot-product bottleneck: every
interaction between the taxonomic, spatial and temporal signals has to survive being squeezed
through one inner product, with only a handful of scalars bypassing it.

That factorisation exists so a retrieval system can index millions of items with an approximate
nearest-neighbour search. Here the candidate set is 13,124 and every candidate is scored anyway, so
the constraint costs accuracy and buys nothing.

This model drops it: the pair's full feature vector goes through one network. What it keeps from the
neural side is what the booster cannot do --

  * a learned embedding per pollinator, and per plant genus and family, so identity is a trained
    vector rather than a lookup that returns zeros for an unseen genus;
  * continuous scores, where the booster's binning cost about 0.04 on the taxonomic signal;
  * the two-term objective, a sampled softmax for ordering within a plant and a binary term for
    comparability between plants, which took two-tower PR-AUC from 0.054 to 0.098.

Plant-side parameters are genus and family only. A per-plant embedding would be untrained for every
held-out plant, which is the whole task.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PairNet(nn.Module):
    def __init__(self, n_feat, n_q, n_pgen, n_pfam, emb=32, hid=128, p_drop=0.2):
        super().__init__()
        self.eq = nn.Embedding(n_q, emb)
        self.eg = nn.Embedding(n_pgen, emb)
        self.ef = nn.Embedding(n_pfam, emb)
        self.norm = nn.LayerNorm(n_feat)
        self.mlp = nn.Sequential(
            nn.Linear(n_feat + 3 * emb, hid), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(hid, hid), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(hid, 1))

    def forward(self, feats, qi, gi, fi):
        e = torch.cat([self.eq(qi), self.eg(gi), self.ef(fi)], -1)
        return self.mlp(torch.cat([self.norm(feats), e], -1)).squeeze(-1)


class PairRanker:
    """Baseline-interface wrapper. All feature construction happens on GPU."""

    name = "Neural pair ranker (ours)"
    reference = "this work"
    cold_start = True

    def __init__(self, epochs=20, batch=256, n_cand=512, in_batch=128, lr=2e-3,
                 bce_weight=0.5, family_weight=1e-3, device="cuda", seed=42, pca_dim=15):
        self.__dict__.update(epochs=epochs, batch=batch, n_cand=n_cand, in_batch=in_batch, lr=lr,
                             bce_weight=bce_weight, family_weight=family_weight, seed=seed,
                             pca_dim=pca_dim)
        self.device = device if torch.cuda.is_available() else "cpu"

    def _feats(self, pi, qi):
        """[B, C, D] pair features for plant rows pi and candidate columns qi, built on GPU."""
        B, C = len(pi), len(qi)
        pop = self.Prs[qi].view(1, C, 1).expand(B, C, 1)
        n = torch.log1p(self.N[pi][:, qi].float()).unsqueeze(-1)
        tx = (self.Cg[qi][:, self.GI[pi]].T + self.family_weight * self.Cf[qi][:, self.FI[pi]].T).unsqueeze(-1)
        pca = self.Fp[pi].unsqueeze(1) * self.Pp[qi].unsqueeze(0)
        loc = torch.log1p(torch.clamp(self.Pj[pi] @ self.Qj[qi].T, min=0)).unsqueeze(-1)
        fc = self.FC[pi].unsqueeze(1).expand(B, C, 52)
        ac = self.AC[qi].unsqueeze(0).expand(B, C, 52)
        return torch.cat([pop, n, tx, pca, loc, fc, ac], -1)

    def fit(self, edges, store):
        self.store = store
        dev = self.device
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        from sklearn.decomposition import TruncatedSVD

        gen = np.array([s.split()[0] for s in store.plants])
        fam = np.array([store.family.get(s, "UNK") for s in store.plants])
        g2i = {g: i for i, g in enumerate(sorted(set(gen)))}
        f2i = {f: i for i, f in enumerate(sorted(set(fam)))}
        GI = np.array([g2i[g] for g in gen]); FI = np.array([f2i[f] for f in fam])
        pi = store.idx_plants(edges["plant"]); qi = store.idx_polls(edges["pollinator"])
        Cg = np.zeros((len(store.polls), len(g2i)), np.float32)
        Cf = np.zeros((len(store.polls), len(f2i)), np.float32)
        np.add.at(Cg, (qi, GI[pi]), 1.0)
        np.add.at(Cf, (qi, FI[pi]), 1.0)

        f = TruncatedSVD(self.pca_dim, random_state=self.seed).fit_transform(store.F.astype(np.float32))
        q = TruncatedSVD(self.pca_dim, random_state=self.seed).fit_transform(store.P.astype(np.float32))
        norm = lambda x: x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)

        T = lambda a, d=torch.float32: torch.from_numpy(np.ascontiguousarray(a)).to(dev, d)
        self.GI, self.FI = T(GI, torch.long), T(FI, torch.long)
        self.Cg, self.Cf = T(Cg), T(Cf)
        self.Fp, self.Pp = T(norm(f)), T(norm(q))
        self.FC, self.AC = T(store.FC), T(store.AC)
        self.Prs = T(np.log1p(store.Prs))
        self.N = T(np.asarray(store.N_full, dtype=np.float32))
        sc = float(np.abs(store.plant_proj).mean()) + 1e-12
        self.Pj, self.Qj = T(store.plant_proj / sc), T(store.poll_proj / sc)

        n_feat = 1 + 1 + 1 + self.pca_dim + 1 + 52 + 52
        self.model = PairNet(n_feat, len(store.polls), len(g2i), len(f2i)).to(dev)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)

        partners = {}
        for a, b in zip(pi.tolist(), qi.tolist()):
            partners.setdefault(a, set()).add(b)
        cnt = np.bincount(qi, minlength=len(store.polls)).astype(np.float64)
        logQ_pop = T(np.log(np.maximum(cnt / cnt.sum(), 1e-12)))
        logQ_uni = float(np.log(1.0 / len(store.polls)))
        n_uni = self.n_cand - self.in_batch

        for ep in range(self.epochs):
            perm = rng.permutation(len(pi))
            tot = 0.0
            for s in range(0, len(pi), self.batch):
                b = perm[s:s + self.batch]
                bp, bq = pi[b], qi[b]
                sub = rng.choice(len(b), min(self.in_batch, len(b)), replace=False)
                cand = np.concatenate([bq[sub], rng.integers(0, len(store.polls), n_uni)])
                tp = torch.from_numpy(bp).long().to(dev)
                tc = torch.from_numpy(cand).long().to(dev)
                feats = self._feats(tp, tc)
                B, C = len(bp), len(cand)
                logits = self.model(feats.view(B * C, -1),
                                    tc.repeat(B), self.GI[tp].repeat_interleave(C),
                                    self.FI[tp].repeat_interleave(C)).view(B, C)

                # the row's own positive, and every other candidate that is also its partner
                pos_col = np.full(B, -1)
                where = {int(c): j for j, c in enumerate(cand)}
                for i, qq in enumerate(bq):
                    pos_col[i] = where.get(int(qq), -1)
                hit = np.zeros((B, C), bool)
                for i, pl in enumerate(bp):
                    ps = partners.get(int(pl))
                    if ps:
                        hit[i] = np.fromiter((c in ps for c in cand), bool, C)
                valid = pos_col >= 0
                hit[np.arange(B), np.clip(pos_col, 0, None)] = False
                hitT = torch.from_numpy(hit).to(dev)
                tgt = torch.from_numpy(np.clip(pos_col, 0, None)).long().to(dev)
                vm = torch.from_numpy(valid).to(dev)

                corr = torch.cat([logQ_pop[torch.from_numpy(bq[sub]).long().to(dev)],
                                  torch.full((n_uni,), logQ_uni, device=dev)])
                ce = F.cross_entropy((logits - corr[None, :]).masked_fill(hitT, -1e4)[vm], tgt[vm])
                lab = torch.zeros_like(logits)
                lab[torch.arange(B, device=dev)[vm], tgt[vm]] = 1.0
                bce = F.binary_cross_entropy_with_logits(logits, lab, weight=(~hitT).float())
                loss = ce + self.bce_weight * bce
                opt.zero_grad(); loss.backward(); opt.step()
                tot += float(ce) * len(b)
            sched.step()
            if ep % 4 == 0 or ep == self.epochs - 1:
                print(f"    epoch {ep + 1}/{self.epochs} ce {tot / len(pi):.4f}", flush=True)
        self.model.eval()
        return self

    def score_plant(self, p, chunk=2048):
        nq = len(self.store.polls)
        dev = self.device
        tp = torch.tensor([p], device=dev, dtype=torch.long)
        out = np.empty(nq, np.float32)
        with torch.no_grad():
            for s in range(0, nq, chunk):
                tc = torch.arange(s, min(s + chunk, nq), device=dev)
                feats = self._feats(tp, tc).view(len(tc), -1)
                out[s:s + len(tc)] = self.model(
                    feats, tc, self.GI[tp].expand(len(tc)), self.FI[tp].expand(len(tc))
                ).float().cpu().numpy()
        return out

    def score_pairs(self, pi, qi):
        out = np.empty(len(pi), np.float64)
        for p in np.unique(pi):
            m = pi == p
            out[m] = self.score_plant(int(p))[qi[m]]
        return out
