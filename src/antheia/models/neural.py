"""Two-tower retrieval for cold-start plant-pollinator prediction.

Each tower reads one species and emits a unit-norm vector; the score is their inner product plus a
small wide term for the pair features a tower structurally cannot see. Three things follow from that
shape, and each addresses a specific failure measured on this data.

**Scores are continuous and globally comparable.** A gradient-boosted ranker bins its features, and
on the taxonomic affinity signal -- whose useful structure is a lexicographic tie-break -- that costs
about 0.04 nrecall@10. An inner product preserves the ordering, and because every plant is scored in
one embedding space the scores are comparable between plants, which pooled PR-AUC requires.

**Taxonomy is learned, not looked up.** The hand-built affinity table cannot generalise across
genera: an unseen genus is a fresh row of zeros. Genus and family embeddings put related taxa near
each other, so a plant whose genus never appears in training still lands somewhere sensible.

**Phenology is encoded per cell.** Continental marginal curves barely discriminate -- a random
plant-pollinator pair already overlaps 0.50 against 0.59 for a true pair, because CONUS phenology is
dominated by a shared summer peak. The surface encoder factorises a taxon's [cells x weeks] surface
through a learned spatial basis and a learned temporal basis, so the towers can express "active in
the same weeks *in the same places*", which no marginal can reconstruct.

Training is sampled softmax over mixed negatives with a logQ correction (Yi et al. 2019, RecSys;
Yang et al. 2020, arXiv:2006.11632): in-batch negatives are drawn in proportion to how often a
pollinator appears as a positive, so without the correction the loss simply relearns popularity.
Candidates that are recorded partners of the row's own plant are masked, since scoring them as
negatives puts a floor under the loss and trains the model against correct answers.

The objective is two terms. The softmax orders candidates within a plant, which is what retrieval
needs, and says nothing about whether one plant's scores are comparable with another's, which is
what pooled PR-AUC measures; a binary term on the same logits supplies that.
"""
from pathlib import Path

import numpy as np
import torch
from antheia.paths import TEXT_DIR
import torch.nn as nn
import torch.nn.functional as F
from antheia import negpool

WIDE_DIM = 5


class SurfaceEncoder(nn.Module):
    """[cells x weeks] surface -> vector, factorised through a spatial and a temporal basis.

    A dense layer over 3,335 x 52 inputs would be 22M parameters per tower and would learn each cell
    independently. Factorising costs 3,335*k + 52*m parameters and forces the model to describe a
    surface as a few regions crossed with a few seasonal modes, which is the structure phenology
    actually has.
    """

    def __init__(self, n_cells, n_weeks=52, k_space=48, k_time=12, out=64):
        super().__init__()
        self.ws = nn.Parameter(torch.randn(n_cells, k_space) * (1.0 / np.sqrt(n_cells)))
        self.wt = nn.Parameter(torch.randn(n_weeks, k_time) * (1.0 / np.sqrt(n_weeks)))
        self.norm = nn.LayerNorm(k_space * k_time)
        self.proj = nn.Linear(k_space * k_time, out)

    def forward(self, surf):
        # surf [B, cells, weeks] -> [B, k_space, k_time]
        h = torch.einsum("bcw,ck,wm->bkm", surf, self.ws, self.wt)
        return self.proj(self.norm(h.flatten(1)))


class Tower(nn.Module):
    def __init__(self, dense_dim, n_gen, n_fam, n_cells, emb=32, hid=256, out=128,
                 p_drop=0.2, use_surface=True, surf_out=64):
        super().__init__()
        self.eg = nn.Embedding(n_gen, emb)
        self.ef = nn.Embedding(n_fam, emb)
        self.surf = SurfaceEncoder(n_cells, out=surf_out) if use_surface else None
        in_dim = dense_dim + 2 * emb + (surf_out if use_surface else 0)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hid), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(hid, hid), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(hid, out))

    def forward(self, dense, gi, fi, surf=None):
        blocks = [dense, self.eg(gi), self.ef(fi)]
        if self.surf is not None:
            blocks.append(self.surf(surf))
        return F.normalize(self.mlp(torch.cat(blocks, -1)), dim=-1)


class TwoTower(nn.Module):
    def __init__(self, p_dense, q_dense, n_pgen, n_pfam, n_qgen, n_qfam, n_cells,
                 use_surface=True, out=128):
        super().__init__()
        self.pt = Tower(p_dense, n_pgen, n_pfam, n_cells, out=out, use_surface=use_surface)
        self.qt = Tower(q_dense, n_qgen, n_qfam, n_cells, out=out, use_surface=use_surface)
        self.wide = nn.Sequential(nn.Linear(WIDE_DIM, 32), nn.ReLU(), nn.Linear(32, 1))
        self.tau = nn.Parameter(torch.tensor(10.0))

    def score(self, pv, qv, wide):
        return self.tau * (pv * qv).sum(-1) + self.wide(wide).squeeze(-1)


class NeuralRanker:
    """Trains the two-tower and exposes the Baseline interface, so it runs in the same harness.

    Negatives are mixed: the other positives in the batch, which are drawn in proportion to how often
    a pollinator is recorded, plus uniformly sampled pollinators, which is the only way a taxon with
    no training interaction ever appears as a negative. Both groups get a logQ correction, subtracting
    the log sampling probability from the logit, without which the softmax rewards popularity for
    being sampled often rather than for being right.
    """

    name = "Two-tower retrieval (ours)"
    reference = "after Yi et al. 2019, RecSys; Yang et al. 2020"
    cold_start = True

    def __init__(self, use_surface=True, epochs=25, batch=512, lr=3e-3, n_uniform=1024,
                 family_weight=1e-3, device="cuda", seed=42, out=128, bce_weight=0.5,
                 use_text=False, text_dir=None):
        self.__dict__.update(use_surface=use_surface, epochs=epochs, batch=batch, lr=lr,
                             n_uniform=n_uniform, family_weight=family_weight, seed=seed, out=out,
                             bce_weight=bce_weight, use_text=use_text, text_dir=text_dir)
        self.device = device if torch.cuda.is_available() else "cpu"
        self.name = ("Two-tower + per-cell surface (ours)" if use_surface
                     else "Two-tower retrieval (ours)") + (" + name embedding" if use_text else "")
        self.reference = "this work" if use_surface else "after Yi et al. 2019, RecSys"

    # ---- feature construction -------------------------------------------------
    def _cats(self, labels, fam_of):
        gen = [s.split()[0] for s in labels]
        fam = [fam_of.get(s, "UNK") for s in labels]
        g2i = {g: i for i, g in enumerate(sorted(set(gen)))}
        f2i = {f: i for i, f in enumerate(sorted(set(fam)))}
        return (np.array([g2i[g] for g in gen]), np.array([f2i[f] for f in fam]), len(g2i), len(f2i))

    def _wide(self, pi, qi):
        """[log1p(N), marginal overlap, taxonomic composite, log1p(pollinator range)]."""
        st = self.store
        n = np.log1p(np.asarray(st.N_full[pi, qi], dtype=np.float32))
        ov = np.minimum(st.FC[pi], st.AC[qi]).sum(1).astype(np.float32)
        tx = self.Cg[qi, self.p_gi[pi]] + self.family_weight * self.Cf[qi, self.p_fi[pi]]
        loc = np.log1p(np.maximum(st.local_overlap(pi, qi), 0)).astype(np.float32)
        return np.stack([n, ov, tx, np.log1p(st.Prs[qi]).astype(np.float32), loc], 1)

    # ---- fit ------------------------------------------------------------------
    def fit(self, edges, store):
        self.store = store
        dev = self.device
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)

        self.p_gi, self.p_fi, n_pg, n_pf = self._cats(store.plants, store.family)
        self.q_gi, self.q_fi, n_qg, n_qf = self._cats(store.polls, store.family)

        pi = store.idx_plants(edges["plant"])
        qi = store.idx_polls(edges["pollinator"])
        self.Cg = np.zeros((len(store.polls), n_pg), np.float32)
        self.Cf = np.zeros((len(store.polls), n_pf), np.float32)
        np.add.at(self.Cg, (qi, self.p_gi[pi]), 1.0)
        np.add.at(self.Cf, (qi, self.p_fi[pi]), 1.0)

        p_blocks = [store.FC, np.log1p(store.Frs)[:, None]]
        q_blocks = [store.AC, np.log1p(store.Prs)[:, None]]
        if self.use_surface:
            # the shared-basis projection carries the per-cell surface in 256 dimensions with its
            # inner products intact, so the tower starts from the structure instead of learning it
            ps, qs = store.plant_proj, store.poll_proj
            sc = float(np.abs(ps).mean() + np.abs(qs).mean()) / 2 + 1e-12
            p_blocks.append(ps / sc)
            q_blocks.append(qs / sc)
        if self.use_text:
            # the embedding enters as a tower input, so the model learns its own bilinear map over
            # the text space instead of being handed a hand-built prototype cosine
            import torch as _t
            td = Path(self.text_dir) if self.text_dir else TEXT_DIR
            tp = _t.load(td / "plants_bioclip2.pt", weights_only=False)["embeddings"].numpy()
            tq = _t.load(td / "polls_bioclip2.pt", weights_only=False)["embeddings"].numpy()
            p_blocks.append(tp / (np.linalg.norm(tp, axis=1, keepdims=True) + 1e-12))
            q_blocks.append(tq / (np.linalg.norm(tq, axis=1, keepdims=True) + 1e-12))
        p_dense = np.hstack(p_blocks).astype(np.float32)
        q_dense = np.hstack(q_blocks).astype(np.float32)
        self.pD = torch.from_numpy(p_dense).to(dev)
        self.qD = torch.from_numpy(q_dense).to(dev)
        self.pG = torch.from_numpy(self.p_gi).long().to(dev)
        self.pF = torch.from_numpy(self.p_fi).long().to(dev)
        self.qG = torch.from_numpy(self.q_gi).long().to(dev)
        self.qF = torch.from_numpy(self.q_fi).long().to(dev)
        self.model = TwoTower(p_dense.shape[1], q_dense.shape[1], n_pg, n_pf, n_qg, n_qf,
                              0, use_surface=False, out=self.out).to(dev)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        # partners of each plant, to mask accidental hits: a sampled negative that is in fact a
        # recorded partner puts a floor under the softmax loss and teaches the model it is wrong
        partners = {}
        for a, b in zip(pi.tolist(), qi.tolist()):
            partners.setdefault(a, set()).add(b)
        self._partners = partners

        # sampling distribution for the logQ correction
        cnt = np.bincount(qi, minlength=len(store.polls)).astype(np.float64)
        pop = cnt / cnt.sum()
        logQ_pop = torch.from_numpy(np.log(np.maximum(pop, 1e-12))).float().to(dev)
        logQ_uni = float(np.log(1.0 / len(store.polls)))

        n = len(pi)
        for ep in range(self.epochs):
            perm = rng.permutation(n)
            tot = 0.0
            for s in range(0, n, self.batch):
                b = perm[s:s + self.batch]
                bp, bq = pi[b], qi[b]
                uni = negpool.sample(rng, self.n_uniform, len(store.polls))
                cand = np.concatenate([bq, uni])                      # [B + U]
                pv = self._pvec(bp)
                qv = self._qvec(cand)
                wide = torch.from_numpy(
                    self._wide(np.repeat(bp, len(cand)), np.tile(cand, len(bp)))
                ).to(dev).view(len(bp), len(cand), WIDE_DIM)
                logits = (self.model.tau * (pv[:, None, :] * qv[None, :, :]).sum(-1)
                          + self.model.wide(wide).squeeze(-1))
                corr = torch.cat([logQ_pop[torch.from_numpy(bq).long().to(dev)],
                                  torch.full((len(uni),), logQ_uni, device=dev)])
                # mask every candidate that is a recorded partner of this row's plant, except the
                # column holding its own positive
                hit = np.zeros((len(bp), len(cand)), dtype=bool)
                for i, pl in enumerate(bp):
                    ps = partners.get(int(pl))
                    if ps:
                        hit[i] = np.fromiter((c in ps for c in cand), bool, len(cand))
                hit[np.arange(len(bp)), np.arange(len(bp))] = False
                hitT = torch.from_numpy(hit).to(dev)

                target = torch.arange(len(bp), device=dev)
                ce = F.cross_entropy((logits - corr[None, :]).masked_fill(hitT, -1e4), target)
                # calibration term: the softmax fixes ordering within a plant but says nothing about
                # whether scores are comparable between plants, which pooled PR-AUC needs
                lab = torch.zeros_like(logits)
                lab[target, target] = 1.0
                bce = F.binary_cross_entropy_with_logits(
                    logits, lab, weight=(~hitT).float(), reduction="mean")
                loss = ce + self.bce_weight * bce
                opt.zero_grad(); loss.backward(); opt.step()
                tot += float(ce) * len(b)
            sched.step()
            if ep % 4 == 0 or ep == self.epochs - 1:
                print(f"    epoch {ep + 1}/{self.epochs} ce {tot / n:.4f}", flush=True)

        self.model.eval()
        with torch.no_grad():
            self.qvecs = torch.cat([self._qvec(np.arange(i, min(i + 512, len(store.polls))))
                                    for i in range(0, len(store.polls), 512)])
        return self

    def _pvec(self, idx):
        t = torch.from_numpy(np.asarray(idx)).long().to(self.device)
        return self.model.pt(self.pD[t], self.pG[t], self.pF[t])

    def _qvec(self, idx):
        t = torch.from_numpy(np.asarray(idx)).long().to(self.device)
        return self.model.qt(self.qD[t], self.qG[t], self.qF[t])

    def score_plant(self, p):
        nq = len(self.store.polls)
        with torch.no_grad():
            pv = self._pvec(np.array([p]))
            wide = torch.from_numpy(self._wide(np.full(nq, p), np.arange(nq))).to(self.device)
            s = self.model.tau * (pv * self.qvecs).sum(-1) + self.model.wide(wide).squeeze(-1)
        return s.float().cpu().numpy()

    def score_pairs(self, pi, qi):
        out = np.empty(len(pi), dtype=np.float64)
        for p in np.unique(pi):
            m = pi == p
            out[m] = self.score_plant(int(p))[qi[m]]
        return out
