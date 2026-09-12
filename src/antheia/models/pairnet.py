"""Neural pair ranker for cold-start plant-pollinator retrieval.

Design rationale
----------------
**Why a pair model rather than two towers.** A two-tower scorer factorises as <f(plant), g(poll)>,
so every interaction between the taxonomic, spatial and temporal signals must survive projection
through a single inner product. That factorisation exists to permit approximate nearest-neighbour
search over item catalogues too large to score exhaustively. Here the candidate set is 13,124 and
every candidate is scored anyway, so the constraint costs accuracy and buys nothing: on this data
the two-tower plateaus at 0.28 nrecall@10 under every feature set tried, against 0.326 for a
gradient-boosted ranker on identical inputs.

**Why per-feature standardisation, not LayerNorm.** The pair features are heterogeneous physical
quantities: a co-occurrence count (skew 5.1), a taxonomic affinity count (skew 20.8), two
probability curves in [0, 0.24], and a per-cell overlap with mean 247 and maximum 16,481. LayerNorm
normalises each sample across the feature axis, which presumes the features are exchangeable draws
from a shared distribution -- true for token embeddings, false here. Under it, the meaning of a
given affinity value depends on the magnitudes of that pair's unrelated features. Statistics are
instead estimated once per feature over training pairs and frozen as buffers, so train and inference
apply an identical affine map.

**Why log1p on the count-like features.** Affinity, co-occurrence and per-cell overlap are
non-negative and heavy-tailed. A linear layer on a variable with skew 20 is dominated by its tail;
log1p is the standard variance-stabilising transform for counts and makes the subsequent
standardisation meaningful. Trees are invariant to monotone transforms and need none of this, which
is one reason the booster tolerated the raw features.

**Why the objective has two terms.** A sampled softmax orders candidates within a plant, which is
what retrieval measures, and constrains nothing about comparability between plants, which is what
pooled PR-AUC measures. A binary term on the same logits supplies the second. Positives are weighted
by the negative-to-positive ratio so the binary term is not swamped at 1:511.

**Why the logQ correction is optional.** In-batch negatives are drawn in proportion to how often a
pollinator is recorded, so an uncorrected softmax partly relearns popularity. But popularity is
genuinely predictive here -- ranking by it alone reaches 0.244 nrecall@10 -- so removing it is not
free, and the correction is exposed as a switch rather than assumed.

**Cold start.** Plant-side parameters are genus and family embeddings only. A per-plant embedding
would be untrained for every held-out plant, which is precisely the evaluation setting. Pollinators
are all seen during training, so a per-pollinator embedding is admissible.

**The niche bilinear term.** Both sides are projected onto one orthonormal basis of the
[cells x weeks] surface space, so each taxon is a 256-dimensional vector whose inner product with
another taxon's *is* their per-cell co-activity. Passing only that inner product fixes the metric at
the identity: every cell and every week counts equally toward compatibility. The term generalises it
to phi_p^T (diag(d) + A B^T) phi_q with d initialised at one and A, B near zero, so training begins
at exactly the hand-computed feature and learns a correction to it. What the correction can express
is ecological -- that overlap in one region and season predicts interaction more strongly than
overlap in another -- which a scalar overlap cannot represent, and which factorisation of the
interaction matrix cannot reach for a species absent from training.

**The text bilinear term.** BioCLIP-2 embeds a bare binomial into a 768-dimensional space in which
congeners sit close (same-genus cosine 0.716 against 0.412 for random pairs) without ever being shown
the taxonomy. The taxonomic affinity table is the sparse counterpart of that: an identity lookup that
is empty for the 7,491 of 13,124 pollinators with fewer than three training edges, and empty for any
plant genus absent from training. The term learns two projections and scores compatibility as
(T_p W_p) . (T_q W_q), which is a plant-to-pollinator compatibility map in species space -- supervised
by interactions, defined for any taxon that has a name, and smooth where the lookup is sparse.

Hand-built reductions of the same embeddings both failed: a prototype cosine scored below its own
permuted control, and kernel smoothing of the affinity table degraded the model as much when the
neighbour graph was permuted as when it was real. Those results bound the reduction, not the
representation, which is why the projections are learned here. This is also the one place a neural
model should be expected to beat a boosted tree, which cannot exploit a dense 768-dimensional input.

**Numerical control.** Masked logits use a large finite negative rather than -inf, so a fully masked
row cannot produce NaN. Gradients are clipped to unit norm: early in training the masked softmax
places most mass on few candidates and produces large updates. Embeddings are initialised at
N(0, 0.01) so they do not dominate standardised features before they carry signal.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from antheia.paths import TEXT_DIR
import torch.nn as nn
import torch.nn.functional as F
from antheia import negpool

# Feature block layout, in construction order. LOG marks blocks given log1p before standardisation.
BLOCKS = [("pop", 1, False), ("cooc", 1, True), ("taxo", 1, True),
          ("pca", 15, False), ("local", 1, True), ("fcurve", 52, False), ("acurve", 52, False)]
N_FEAT = sum(n for _, n, _ in BLOCKS)
MASK_FILL = -1e4


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class PairConfig:
    """Every knob the ablation varies, with the defaults justified in the module docstring."""
    epochs: int = 20
    batch: int = 256            # plants per step
    n_cand: int = 512           # candidates scored per plant per step
    in_batch: int = 128         # of which drawn from the batch's own positives
    lr: float = 2e-3
    weight_decay: float = 1e-4
    hidden: int = 256
    depth: int = 2
    dropout: float = 0.1
    emb_dim: int = 32
    grad_clip: float = 1.0
    bce_weight: float = 0.5
    family_weight: float = 1e-3
    pca_dim: int = 15
    seed: int = 42
    norm: str = "feature"       # "feature" | "layer" | "none"
    logq: bool = True
    loss: str = "both"          # "both" | "softmax" | "bce"
    use_q_emb: bool = True
    use_tax_emb: bool = True
    stat_pairs: int = 200_000   # sample size for the standardisation statistics
    bilinear_rank: int = 0      # 0 disables the niche term; >0 gives the low-rank correction
    niche_dim: int = 256        # width of the shared surface basis
    text_rank: int = 0          # 0 disables the text term; >0 gives the joint species-space rank
    text_dim: int = 768         # BioCLIP-2 text embedding width
    text_dir: str = str(TEXT_DIR)
    device: str = "cuda"
    name: str = "Neural pair ranker (ours)"


class PairNet(nn.Module):
    """Scores one (plant, pollinator) pair. Standardisation lives in frozen buffers."""

    def __init__(self, cfg: PairConfig, n_q: int, n_gen: int, n_fam: int):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("mu", torch.zeros(N_FEAT))     # [D]
        self.register_buffer("sd", torch.ones(N_FEAT))      # [D]
        self.ln = nn.LayerNorm(N_FEAT) if cfg.norm == "layer" else None

        emb_total = 0
        if cfg.use_q_emb:
            self.eq = nn.Embedding(n_q, cfg.emb_dim); emb_total += cfg.emb_dim
        if cfg.use_tax_emb:
            self.eg = nn.Embedding(n_gen, cfg.emb_dim)
            self.ef = nn.Embedding(n_fam, cfg.emb_dim); emb_total += 2 * cfg.emb_dim
        for m in (getattr(self, a, None) for a in ("eq", "eg", "ef")):
            if m is not None:
                nn.init.normal_(m.weight, std=0.01)

        if cfg.bilinear_rank > 0:
            r, k = cfg.bilinear_rank, cfg.niche_dim
            self.bil_d = nn.Parameter(torch.ones(k))              # [K] identity at init
            self.bil_A = nn.Parameter(torch.randn(k, r) * 1e-3)   # [K,r]
            self.bil_B = nn.Parameter(torch.randn(k, r) * 1e-3)   # [K,r]
            self.bil_scale = nn.Parameter(torch.tensor(1.0))

        if cfg.text_rank > 0:
            r = cfg.text_rank
            self.txt_P = nn.Linear(cfg.text_dim, r, bias=False)   # plant  -> joint space [768,r]
            self.txt_Q = nn.Linear(cfg.text_dim, r, bias=False)   # pollin -> joint space [768,r]
            nn.init.normal_(self.txt_P.weight, std=1.0 / np.sqrt(cfg.text_dim))
            nn.init.normal_(self.txt_Q.weight, std=1.0 / np.sqrt(cfg.text_dim))
            self.txt_scale = nn.Parameter(torch.tensor(1.0))

        dims = [N_FEAT + emb_total] + [cfg.hidden] * cfg.depth
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            lin = nn.Linear(a, b)
            nn.init.kaiming_normal_(lin.weight, nonlinearity="relu")
            nn.init.zeros_(lin.bias)
            layers += [lin, nn.ReLU(inplace=True), nn.Dropout(cfg.dropout)]
        head = nn.Linear(dims[-1], 1)
        nn.init.zeros_(head.bias)
        nn.init.normal_(head.weight, std=0.01)
        self.mlp = nn.Sequential(*layers, head)

    def set_stats(self, mu: torch.Tensor, sd: torch.Tensor) -> None:
        self.mu.copy_(mu)
        self.sd.copy_(torch.clamp(sd, min=1e-6))

    def niche(self, phi_p, phi_q):
        """phi_p [B,K], phi_q [C,K] -> [B,C]. Identity metric at initialisation."""
        d = self.bil_d.unsqueeze(0) * phi_p                       # [B,K]
        core = d @ phi_q.T                                        # [B,C]
        low = (phi_p @ self.bil_A) @ (phi_q @ self.bil_B).T       # [B,C]
        return self.bil_scale * (core + low)

    def text(self, tp, tq):
        """tp [B,768], tq [C,768] -> [B,C] compatibility in the learned joint species space."""
        return self.txt_scale * (self.txt_P(tp) @ self.txt_Q(tq).T)

    def forward(self, feats, qi, gi, fi):
        """feats [M, D]; qi/gi/fi [M] -> logits [M]."""
        x = feats
        if self.cfg.norm == "feature":
            x = (x - self.mu) / self.sd
        elif self.cfg.norm == "layer":
            x = self.ln(x)
        parts = [x]
        if self.cfg.use_q_emb:
            parts.append(self.eq(qi))
        if self.cfg.use_tax_emb:
            parts += [self.eg(gi), self.ef(fi)]
        return self.mlp(torch.cat(parts, -1)).squeeze(-1)


class PairRanker:
    """Baseline-interface wrapper: fit(edges, store) then score_plant(p)."""

    reference = "this work"
    cold_start = True

    def __init__(self, **kw):
        self.cfg = PairConfig(**kw)
        self.name = self.cfg.name
        if not torch.cuda.is_available():
            self.cfg.device = "cpu"

    # ---- feature construction (all on GPU) ------------------------------------
    def _feats(self, pi, qi):
        """pi [B], qi [C] -> [B, C, D] in the BLOCKS order."""
        B, C = len(pi), len(qi)
        pop = self.Prs[qi].view(1, C, 1).expand(B, C, 1)                       # [B,C,1]
        cooc = torch.log1p(self.N[pi][:, qi]).unsqueeze(-1)                    # [B,C,1]
        taxo = torch.log1p(self.Cg[qi][:, self.GI[pi]].T
                           + self.cfg.family_weight * self.Cf[qi][:, self.FI[pi]].T).unsqueeze(-1)
        pca = self.Fp[pi].unsqueeze(1) * self.Pp[qi].unsqueeze(0)              # [B,C,15]
        local = torch.log1p(torch.clamp(self.Pj[pi] @ self.Qj[qi].T, min=0)).unsqueeze(-1)
        fc = self.FC[pi].unsqueeze(1).expand(B, C, 52)                         # [B,C,52]
        ac = self.AC[qi].unsqueeze(0).expand(B, C, 52)                         # [B,C,52]
        return torch.cat([pop, cooc, taxo, pca, local, fc, ac], -1)            # [B,C,D]

    def _feats_pairs(self, pi, qi):
        """Matched pairs: pi [M], qi [M] -> [M, D], same block order as _feats."""
        pop = self.Prs[qi].unsqueeze(-1)                                       # [M,1]
        cooc = torch.log1p(self.N[pi, qi]).unsqueeze(-1)                       # [M,1]
        taxo = torch.log1p(self.Cg[qi, self.GI[pi]]
                           + self.cfg.family_weight * self.Cf[qi, self.FI[pi]]).unsqueeze(-1)
        pca = self.Fp[pi] * self.Pp[qi]                                        # [M,15]
        local = torch.log1p(torch.clamp((self.Pj[pi] * self.Qj[qi]).sum(-1), min=0)).unsqueeze(-1)
        return torch.cat([pop, cooc, taxo, pca, local, self.FC[pi], self.AC[qi]], -1)

    def _standardisation_stats(self, pi, qi, rng):
        """Per-feature mean and sd over a sample of training pairs. Training edges only."""
        n = min(self.cfg.stat_pairs, len(pi) * 4)
        a = pi[rng.integers(0, len(pi), n)]
        b = rng.integers(0, len(self.store.polls), n)
        tot = torch.zeros(N_FEAT, device=self.dev, dtype=torch.float64)
        sq = torch.zeros(N_FEAT, device=self.dev, dtype=torch.float64)
        seen = 0
        for s in range(0, n, 4096):
            ta = torch.from_numpy(np.ascontiguousarray(a[s:s + 4096])).long().to(self.dev)
            tb = torch.from_numpy(np.ascontiguousarray(b[s:s + 4096])).long().to(self.dev)
            with torch.no_grad():
                x = self._feats_pairs(ta, tb).double()                          # [m, D]
            tot += x.sum(0); sq += (x ** 2).sum(0); seen += len(x)
        mu = tot / seen
        var = torch.clamp(sq / seen - mu ** 2, min=0.0)
        return mu.float(), var.sqrt().float()

    # ---- fit ------------------------------------------------------------------
    def fit(self, edges, store):
        from sklearn.decomposition import TruncatedSVD
        cfg = self.cfg
        seed_everything(cfg.seed)
        self.store = store
        self.dev = dev = cfg.device
        rng = np.random.default_rng(cfg.seed)

        gen = np.array([s.split()[0] for s in store.plants])
        fam = np.array([store.family.get(s, "UNK") for s in store.plants])
        g2i = {g: i for i, g in enumerate(sorted(set(gen)))}
        f2i = {f: i for i, f in enumerate(sorted(set(fam)))}
        GI = np.array([g2i[g] for g in gen]); FI = np.array([f2i[f] for f in fam])
        pi = store.idx_plants(edges["plant"]); qi = store.idx_polls(edges["pollinator"])

        # affinity tables from training edges only -- no held-out plant contributes to its own score
        Cg = np.zeros((len(store.polls), len(g2i)), np.float32)
        Cf = np.zeros((len(store.polls), len(f2i)), np.float32)
        np.add.at(Cg, (qi, GI[pi]), 1.0)
        np.add.at(Cf, (qi, FI[pi]), 1.0)

        svd = lambda X: TruncatedSVD(cfg.pca_dim, random_state=cfg.seed).fit_transform(X.astype(np.float32))
        unit = lambda x: x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
        T = lambda a, d=torch.float32: torch.from_numpy(np.ascontiguousarray(a)).to(dev, d)
        self.GI, self.FI = T(GI, torch.long), T(FI, torch.long)
        self.Cg, self.Cf = T(Cg), T(Cf)
        self.Fp, self.Pp = T(unit(svd(store.F))), T(unit(svd(store.P)))
        self.FC, self.AC = T(store.FC), T(store.AC)
        self.Prs = T(np.log1p(store.Prs))
        self.N = T(np.asarray(store.N_full, dtype=np.float32))
        sc = float(np.abs(store.plant_proj).mean()) + 1e-12
        self.Pj, self.Qj = T(store.plant_proj / sc), T(store.poll_proj / sc)
        if cfg.bilinear_rank > 0:
            # scale so <phi_p, phi_q> has unit standard deviation over random pairs: the bilinear
            # term then starts as a well-conditioned logit rather than one of magnitude ~250
            rp, rq = store.plant_proj, store.poll_proj
            g = np.random.default_rng(0)
            samp = (rp[g.integers(0, len(rp), 20000)] * rq[g.integers(0, len(rq), 20000)]).sum(1)
            nsc = float(np.sqrt(np.abs(samp).std()) + 1e-12)
            self.Pn, self.Qn = T(rp / nsc), T(rq / nsc)

        if cfg.text_rank > 0:
            td = Path(cfg.text_dir)
            tp_ = torch.load(td / "plants_bioclip2.pt", weights_only=False)["embeddings"].numpy()
            tq_ = torch.load(td / "polls_bioclip2.pt", weights_only=False)["embeddings"].numpy()
            unit_ = lambda x: x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
            self.Tp, self.Tq = T(unit_(tp_)), T(unit_(tq_))       # [P,768], [Q,768]

        self.model = PairNet(cfg, len(store.polls), len(g2i), len(f2i)).to(dev)
        if cfg.norm == "feature":
            mu, sd = self._standardisation_stats(pi, qi, rng)
            self.model.set_stats(mu, sd)

        opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

        partners = {}
        for a, b in zip(pi.tolist(), qi.tolist()):
            partners.setdefault(a, set()).add(b)
        cnt = np.bincount(qi, minlength=len(store.polls)).astype(np.float64)
        logQ_pop = T(np.log(np.maximum(cnt / cnt.sum(), 1e-12)))
        logQ_uni = float(np.log(1.0 / len(store.polls)))
        n_uni = cfg.n_cand - cfg.in_batch
        pos_weight = torch.tensor(float(cfg.n_cand - 1), device=dev)   # class imbalance in the BCE term

        for ep in range(cfg.epochs):
            self.model.train()
            perm = rng.permutation(len(pi))
            tot, nb = 0.0, 0
            for s in range(0, len(pi), cfg.batch):
                b = perm[s:s + cfg.batch]
                bp, bq = pi[b], qi[b]
                B = len(bp)
                sub = rng.choice(B, min(cfg.in_batch, B), replace=False)
                cand = np.concatenate([bq[sub], negpool.sample(rng, n_uni, len(store.polls))])
                C = len(cand)
                tp = torch.from_numpy(np.ascontiguousarray(bp)).long().to(dev)
                tc = torch.from_numpy(np.ascontiguousarray(cand)).long().to(dev)

                feats = self._feats(tp, tc)                                    # [B,C,D]
                logits = self.model(feats.view(B * C, N_FEAT), tc.repeat(B),
                                    self.GI[tp].repeat_interleave(C),
                                    self.FI[tp].repeat_interleave(C)).view(B, C)
                if cfg.bilinear_rank > 0:
                    logits = logits + self.model.niche(self.Pn[tp], self.Qn[tc])
                if cfg.text_rank > 0:
                    logits = logits + self.model.text(self.Tp[tp], self.Tq[tc])

                # column of each row's own positive; -1 when it was not sampled
                where = {}
                for j, c in enumerate(cand):
                    where.setdefault(int(c), j)
                pos_col = np.fromiter((where.get(int(q), -1) for q in bq), int, B)
                valid = pos_col >= 0
                # every candidate that is a recorded partner of the row's plant, target excepted
                hit = np.zeros((B, C), bool)
                for i, pl in enumerate(bp):
                    ps = partners.get(int(pl))
                    if ps:
                        hit[i] = np.fromiter((c in ps for c in cand), bool, C)
                hit[np.arange(B), np.clip(pos_col, 0, None)] = False
                hitT = torch.from_numpy(hit).to(dev)
                tgt = torch.from_numpy(np.clip(pos_col, 0, None)).long().to(dev)
                vm = torch.from_numpy(valid).to(dev)
                if not bool(vm.any()):
                    continue

                loss = logits.new_zeros(())
                if cfg.loss in ("both", "softmax"):
                    z = logits
                    if cfg.logq:
                        corr = torch.cat([logQ_pop[torch.from_numpy(np.ascontiguousarray(bq[sub])).long().to(dev)],
                                          torch.full((n_uni,), logQ_uni, device=dev)])
                        z = z - corr[None, :]
                    loss = loss + F.cross_entropy(z.masked_fill(hitT, MASK_FILL)[vm], tgt[vm])
                if cfg.loss in ("both", "bce"):
                    lab = torch.zeros_like(logits)
                    lab[torch.arange(B, device=dev)[vm], tgt[vm]] = 1.0
                    w = cfg.bce_weight if cfg.loss == "both" else 1.0
                    loss = loss + w * F.binary_cross_entropy_with_logits(
                        logits, lab, weight=(~hitT).float(), pos_weight=pos_weight)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                opt.step()
                tot += float(loss); nb += 1
            sched.step()
            if ep % 5 == 0 or ep == cfg.epochs - 1:
                print(f"    epoch {ep + 1}/{cfg.epochs} loss {tot / max(nb, 1):.4f}", flush=True)
        self.model.eval()
        return self

    # ---- inference ------------------------------------------------------------
    def score_plant(self, p, chunk=2048):
        nq = len(self.store.polls)
        tp = torch.tensor([p], device=self.dev, dtype=torch.long)
        out = np.empty(nq, np.float32)
        with torch.no_grad():
            for s in range(0, nq, chunk):
                tc = torch.arange(s, min(s + chunk, nq), device=self.dev)
                feats = self._feats(tp, tc).view(len(tc), N_FEAT)
                z = self.model(feats, tc, self.GI[tp].expand(len(tc)),
                               self.FI[tp].expand(len(tc)))
                if self.cfg.bilinear_rank > 0:
                    z = z + self.model.niche(self.Pn[tp], self.Qn[tc]).squeeze(0)
                if self.cfg.text_rank > 0:
                    z = z + self.model.text(self.Tp[tp], self.Tq[tc]).squeeze(0)
                out[s:s + len(tc)] = z.float().cpu().numpy()
        return out

    def score_pairs(self, pi, qi):
        out = np.empty(len(pi), np.float64)
        for p in np.unique(pi):
            m = pi == p
            out[m] = self.score_plant(int(p))[qi[m]]
        return out
