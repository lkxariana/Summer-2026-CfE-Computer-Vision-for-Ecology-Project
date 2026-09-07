"""Embedding-first interaction model: encoders per side, a pair head, and a discarded bias head.

Every neural model tried before this one was handed the same hand-computed tabular quantities the
gradient-boosted ranker uses -- shared-cell counts, phenological overlap scalars, an affinity lookup.
Those are summaries a person chose, and a boosted tree fits them better than a network does, which is
the documented behaviour for tabular inputs (Grinsztajn, Oyallon & Varoquaux 2022) and is what a
seven-run ablation of the pair network found: every implementation choice was neutral and the gap
persisted. This model changes the inputs rather than the vehicle.

Inputs are dense representations, each defined for every taxon in the universe:

  BioCLIP-2 text embedding   [768]  species identity; congeners at cosine 0.716 against 0.412 random
  surface projection         [256]  the [3,335 cells x 52 weeks] predicted surface on a shared basis
  occupancy PCA              [15]   truncated SVD of the observed cell occupancy
  log range size             [1]    scale of that occupancy
  observed flag              [1]    distinguishes "no records" from "records showing absence"

**Encoders.** One MLP per side. LayerNorm is applied per block rather than across the concatenation:
within a block the coordinates are exchangeable draws from one representation, which is the
assumption LayerNorm encodes; across blocks they are not, and normalising a text coordinate against a
range size is the error that made LayerNorm inappropriate for the tabular model.

**Pair head.** The score comes from an MLP over [h_p, h_q, h_p * h_q, |h_p - h_q|] rather than an
inner product. The elementwise product and absolute difference are the standard sentence-pair
interaction features (Conneau et al. 2017, EMNLP; Reimers & Gurevych 2019, EMNLP), and they let the
head express conjunctions -- compatible identity *and* spatial overlap -- that a dot product cannot.
On this data the dot-product form plateaus at 0.28 nrecall@10 under every input set tried, and the
factorisation it buys is worthless here because all 13,124 candidates are scored anyway.

**Observation-process head.** Records in GloBI are 71% iNaturalist, and transfer to expert-assembled
field networks is the weakest measured result (0.06-0.07 against 0.32 in domain). A shallow head over
species-level documentation descriptors only -- how heavily each taxon is recorded, never anything
about the pair's ecology -- is added to the logit during training and discarded at inference, so the
ecological towers are freed from explaining sampling effort rather than having to absorb it. This is
the shallow-tower construction used for position bias in recommenders (Zhao et al. 2019, RecSys),
applied to documentation bias.

**Objective.** Sampled softmax with a logQ correction orders candidates within a plant; a binary term
on the same logits makes scores comparable between plants. Ablation showed the correction is worth
0.089 nrecall@10 -- without it a popular pollinator appears as a sampled negative far more often than
a rare one and the model learns to penalise the taxa most likely to be true partners.

**Genus context (E4).** Taxonomic affinity is the strongest signal in this network and it enters
every other model as a sparse lookup: a count table indexed by pollinator identity, empty for the
7,491 of 13,124 pollinators with fewer than three training edges and empty for any plant genus absent
from training. Here the plant instead *attends* to the set of pollinators its genus was recorded
visiting, and the context vector that returns is concatenated to its representation. The set members
are encoded by the same pollinator encoder, so the model can shape the space in which the pooling
happens -- which is what separates this from the hand-built prototype cosine that scored below its
own permuted control.

The set is built leave-one-out: a training plant's own interactions are subtracted from its genus
profile, or the target would appear in the query. Held-out plants contribute nothing to any genus
profile by construction, since the profiles are built from training edges alone.

**Tier head (H1).** Interactions carry an evidence tier -- flower visitation against general
association, 99,033 and 81,316 edges. Predicting it as an auxiliary target asks the representation to
separate pollination from co-occurrence, which is the distinction the Tier A evaluation rests on.

**Cold start.** No per-plant parameters of any kind. A held-out plant is represented only by inputs
computable from its name and its predicted surfaces, so nothing about it is fitted during training.
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MASK_FILL = -1e4
TEXT_DIR = "/scratch/cher/antheia-data/text_embeddings"


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class EmbedConfig:
    epochs: int = 25
    batch: int = 128             # plants per step
    n_cand: int = 384            # candidates per plant per step
    in_batch: int = 128          # of which taken from the batch's own positives
    lr: float = 1e-3
    weight_decay: float = 1e-4
    d_model: int = 256           # encoder output width
    hidden: int = 512
    dropout: float = 0.2
    grad_clip: float = 1.0
    bce_weight: float = 0.5
    logq: bool = True
    use_bias_head: bool = True
    use_genus_context: bool = True   # E4: attend over the plant genus's recorded partners
    ctx_k: int = 64                  # partners kept per genus, by count
    ctx_heads: int = 4
    use_tier_head: bool = True       # H1: auxiliary evidence-tier classification
    tier_weight: float = 0.3
    blocks: tuple = ("text", "surface", "pca", "scale")
    seed: int = 42
    device: str = "cuda"
    text_dir: str = TEXT_DIR
    name: str = "Embedding two-encoder pair model (ours)"


class BlockEncoder(nn.Module):
    """Per-block LayerNorm, then one MLP over the concatenation. Blocks are [B, d_i]."""

    def __init__(self, dims, d_model, hidden, dropout):
        super().__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(d) for d in dims])
        total = sum(dims)
        self.mlp = nn.Sequential(
            nn.Linear(total, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, d_model))
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, blocks):
        return self.mlp(torch.cat([n(b) for n, b in zip(self.norms, blocks)], -1))  # [B, d_model]


class GenusContext(nn.Module):
    """Multi-head attention from a plant to its genus's recorded partners. Returns [B, d]."""

    def __init__(self, d_model, heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.empty = nn.Parameter(torch.zeros(d_model))     # learned fallback for an unseen genus
        self.norm = nn.LayerNorm(d_model)

    def forward(self, hp, ctx, mask):
        """hp [B, d]; ctx [B, K, d]; mask [B, K] True where the slot is padding."""
        allpad = mask.all(dim=1)                            # [B] genera with no recorded partners
        safe = mask.clone()
        safe[allpad, 0] = False                             # keep one slot so attention is defined
        out, _ = self.attn(hp.unsqueeze(1), ctx, ctx, key_padding_mask=safe)
        out = out.squeeze(1)                                # [B, d]
        return self.norm(torch.where(allpad.unsqueeze(-1), self.empty.expand_as(out), out))


class PairHead(nn.Module):
    """[h_p, h_q, h_p*h_q, |h_p-h_q|] -> logit."""

    def __init__(self, d_model, hidden, dropout, n_out=1, p_mult=1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear((2 + 2 * p_mult) * d_model, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_out))
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.mlp[-1].weight, std=0.01)

    def forward(self, hp, hq, ctx=None):
        """hp [B, d], hq [C, d], optional ctx [B, d] -> [B, C, n_out] squeezed when n_out == 1."""
        B, C, d = hp.shape[0], hq.shape[0], hp.shape[1]
        a = hp.unsqueeze(1).expand(B, C, d)
        b = hq.unsqueeze(0).expand(B, C, d)
        parts = [a, b, a * b, (a - b).abs()]
        if ctx is not None:
            c = ctx.unsqueeze(1).expand(B, C, d)
            parts += [c * b, (c - b).abs()]                       # candidate against the genus profile
        out = self.mlp(torch.cat(parts, -1))                      # [B, C, n_out]
        return out.squeeze(-1) if out.shape[-1] == 1 else out


class BiasHead(nn.Module):
    """Documentation effort only; added to the logit in training, never used at inference."""

    def __init__(self, n_feat, hidden=32):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(n_feat, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x):
        return self.mlp(x).squeeze(-1)


class EmbedRanker:
    name = "Embedding two-encoder pair model (ours)"
    reference = "this work"
    cold_start = True

    def __init__(self, **kw):
        self.cfg = EmbedConfig(**kw)
        self.name = self.cfg.name
        if not torch.cuda.is_available():
            self.cfg.device = "cpu"

    # ---- inputs ---------------------------------------------------------------
    def _blocks(self, side):
        return [b.to(self.dev) for b in (self.P_blocks if side == "p" else self.Q_blocks)]

    def _encode_all(self):
        """Encode every taxon once. Returns h_p [P, d], h_q [Q, d]."""
        with torch.no_grad():
            hp = torch.cat([self.enc_p([b[i:i + 4096] for b in self.P_blocks])
                            for i in range(0, self.n_p, 4096)])
            hq = torch.cat([self.enc_q([b[i:i + 4096] for b in self.Q_blocks])
                            for i in range(0, self.n_q, 4096)])
        return hp, hq

    def fit(self, edges, store):
        from sklearn.decomposition import TruncatedSVD
        cfg = self.cfg
        seed_everything(cfg.seed)
        self.store = store
        self.dev = dev = cfg.device
        rng = np.random.default_rng(cfg.seed)
        T = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dev, torch.float32)

        td = Path(cfg.text_dir)
        tp = torch.load(td / "plants_bioclip2.pt", weights_only=False)["embeddings"].numpy()
        tq = torch.load(td / "polls_bioclip2.pt", weights_only=False)["embeddings"].numpy()
        unit = lambda x: x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
        svd = lambda X: TruncatedSVD(15, random_state=cfg.seed).fit_transform(X.astype(np.float32))

        p_all = {"text": unit(tp), "surface": store.plant_proj,
                 "pca": svd(store.F),
                 "scale": np.stack([np.log1p(store.Frs), (store.Frs > 0).astype(np.float32)], 1)}
        q_all = {"text": unit(tq), "surface": store.poll_proj,
                 "pca": svd(store.P),
                 "scale": np.stack([np.log1p(store.Prs), (store.Prs > 0).astype(np.float32)], 1)}
        self.P_blocks = [T(p_all[b]) for b in cfg.blocks]
        self.Q_blocks = [T(q_all[b]) for b in cfg.blocks]
        dims = [b.shape[1] for b in self.P_blocks]
        self.n_p, self.n_q = len(store.plants), len(store.polls)

        self.enc_p = BlockEncoder(dims, cfg.d_model, cfg.hidden, cfg.dropout).to(dev)
        self.enc_q = BlockEncoder(dims, cfg.d_model, cfg.hidden, cfg.dropout).to(dev)
        n_out = 2 if cfg.use_tier_head else 1
        self.head = PairHead(cfg.d_model, cfg.hidden, cfg.dropout, n_out=n_out,
                             p_mult=2 if cfg.use_genus_context else 1).to(dev)
        mods = [self.enc_p, self.enc_q, self.head]
        if cfg.use_genus_context:
            self.ctx = GenusContext(cfg.d_model, cfg.ctx_heads, cfg.dropout).to(dev)
            mods.append(self.ctx)

        pi = store.idx_plants(edges["plant"]); qi = store.idx_polls(edges["pollinator"])

        if cfg.use_genus_context:
            gen = np.array([x.split()[0] for x in store.plants])
            g2i = {g: i for i, g in enumerate(sorted(set(gen)))}
            self.p_gi = np.array([g2i[g] for g in gen])
            counts = {}
            for a, b in zip(pi.tolist(), qi.tolist()):
                counts.setdefault(self.p_gi[a], {}).setdefault(b, 0)
                counts[self.p_gi[a]][b] += 1
            own = {}
            for a, b in zip(pi.tolist(), qi.tolist()):
                own.setdefault(a, {}).setdefault(b, 0)
                own[a][b] += 1
            K = cfg.ctx_k
            idx = np.zeros((self.n_p, K), np.int64)
            msk = np.ones((self.n_p, K), bool)
            for a in range(self.n_p):
                c = dict(counts.get(self.p_gi[a], {}))
                for b, n in own.get(a, {}).items():          # leave-one-out: remove this plant's own
                    c[b] = c.get(b, 0) - n
                    if c[b] <= 0:
                        c.pop(b, None)
                if not c:
                    continue
                top = sorted(c.items(), key=lambda kv: -kv[1])[:K]
                idx[a, :len(top)] = [b for b, _ in top]
                msk[a, :len(top)] = False
            self.ctx_idx = torch.from_numpy(idx).to(dev)
            self.ctx_msk = torch.from_numpy(msk).to(dev)
            print(f"    genus context: {int((~msk).any(1).sum())}/{self.n_p} plants have a non-empty "
                  f"leave-one-out genus profile", flush=True)

        if cfg.use_tier_head:
            tier = (edges["tier"].to_numpy() == "A").astype(np.float32)
            self.tier_of = {(int(a), int(b)): float(t) for a, b, t in zip(pi, qi, tier)}

        if cfg.use_bias_head:
            # species-level documentation effort only: never the pair's ecology
            self.bias_p = T(np.stack([np.log1p(store.Frs), (store.FCo.sum(1) > 0).astype(np.float32)], 1))
            self.bias_q = T(np.stack([np.log1p(store.Prs), (store.ACo.sum(1) > 0).astype(np.float32)], 1))
            self.bias = BiasHead(self.bias_p.shape[1] + self.bias_q.shape[1]).to(dev)
            mods.append(self.bias)

        params = [p for m in mods for p in m.parameters()]
        opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

        partners = {}
        for a, b in zip(pi.tolist(), qi.tolist()):
            partners.setdefault(a, set()).add(b)
        cnt = np.bincount(qi, minlength=self.n_q).astype(np.float64)
        logQ_pop = T(np.log(np.maximum(cnt / cnt.sum(), 1e-12)))
        logQ_uni = float(np.log(1.0 / self.n_q))
        n_uni = cfg.n_cand - cfg.in_batch
        pos_weight = torch.tensor(float(cfg.n_cand - 1), device=dev)

        for ep in range(cfg.epochs):
            for m in mods:
                m.train()
            perm = rng.permutation(len(pi))
            tot, nb = 0.0, 0
            for s in range(0, len(pi), cfg.batch):
                b = perm[s:s + cfg.batch]
                bp, bq = pi[b], qi[b]
                B = len(bp)
                sub = rng.choice(B, min(cfg.in_batch, B), replace=False)
                cand = np.concatenate([bq[sub], rng.integers(0, self.n_q, n_uni)])
                C = len(cand)
                tp_i = torch.from_numpy(np.ascontiguousarray(bp)).long().to(dev)
                tc_i = torch.from_numpy(np.ascontiguousarray(cand)).long().to(dev)

                hp = self.enc_p([blk[tp_i] for blk in self.P_blocks])      # [B, d]
                hq = self.enc_q([blk[tc_i] for blk in self.Q_blocks])      # [C, d]
                ctx = None
                if cfg.use_genus_context:
                    ci = self.ctx_idx[tp_i]                                # [B, K]
                    cm = self.ctx_msk[tp_i]                                # [B, K]
                    cflat = self.enc_q([blk[ci.reshape(-1)] for blk in self.Q_blocks])
                    ctx = self.ctx(hp, cflat.view(len(tp_i), cfg.ctx_k, -1), cm)   # [B, d]
                out = self.head(hp, hq, ctx)                               # [B,C] or [B,C,2]
                logits = out[..., 0] if cfg.use_tier_head else out
                if cfg.use_bias_head:
                    bp_f = self.bias_p[tp_i].unsqueeze(1).expand(B, C, -1)
                    bq_f = self.bias_q[tc_i].unsqueeze(0).expand(B, C, -1)
                    logits = logits + self.bias(torch.cat([bp_f, bq_f], -1))

                where = {}
                for j, c in enumerate(cand):
                    where.setdefault(int(c), j)
                pos_col = np.fromiter((where.get(int(q), -1) for q in bq), int, B)
                valid = pos_col >= 0
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

                z = logits
                if cfg.logq:
                    corr = torch.cat([logQ_pop[torch.from_numpy(np.ascontiguousarray(bq[sub])).long().to(dev)],
                                      torch.full((n_uni,), logQ_uni, device=dev)])
                    z = z - corr[None, :]
                loss = F.cross_entropy(z.masked_fill(hitT, MASK_FILL)[vm], tgt[vm])
                lab = torch.zeros_like(logits)
                lab[torch.arange(B, device=dev)[vm], tgt[vm]] = 1.0
                loss = loss + cfg.bce_weight * F.binary_cross_entropy_with_logits(
                    logits, lab, weight=(~hitT).float(), pos_weight=pos_weight)

                if cfg.use_tier_head:
                    # auxiliary: is this pair flower-visitation evidence, supervised on positives only
                    tv = np.array([self.tier_of.get((int(a), int(cand[c])), -1.0)
                                   for a, c in zip(bp, np.clip(pos_col, 0, None))], np.float32)
                    keep = torch.from_numpy((tv >= 0) & valid).to(dev)
                    if bool(keep.any()):
                        tl = out[torch.arange(B, device=dev), tgt, 1]
                        loss = loss + cfg.tier_weight * F.binary_cross_entropy_with_logits(
                            tl[keep], torch.from_numpy(np.clip(tv, 0, 1)).to(dev)[keep])

                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(params, cfg.grad_clip)
                opt.step()
                tot += float(loss); nb += 1
            sched.step()
            if ep % 5 == 0 or ep == cfg.epochs - 1:
                print(f"    epoch {ep + 1}/{cfg.epochs} loss {tot / max(nb, 1):.4f}", flush=True)

        for m in mods:
            m.eval()
        self.hp_all, self.hq_all = self._encode_all()      # bias head deliberately not applied
        if cfg.use_genus_context:
            with torch.no_grad():
                self.ctx_all = torch.cat([
                    self.ctx(self.hp_all[i:i + 512],
                             self.hq_all[self.ctx_idx[i:i + 512].reshape(-1)].view(
                                 len(self.hp_all[i:i + 512]), cfg.ctx_k, -1),
                             self.ctx_msk[i:i + 512])
                    for i in range(0, self.n_p, 512)])
        return self

    def score_plant(self, p, chunk=4096):
        out = np.empty(self.n_q, np.float32)
        with torch.no_grad():
            hp = self.hp_all[p:p + 1]
            ctx = self.ctx_all[p:p + 1] if self.cfg.use_genus_context else None
            for s in range(0, self.n_q, chunk):
                hq = self.hq_all[s:s + chunk]
                z = self.head(hp, hq, ctx)
                if self.cfg.use_tier_head:
                    z = z[..., 0]
                out[s:s + len(hq)] = z.squeeze(0).float().cpu().numpy()
        return out

    def score_pairs(self, pi, qi):
        out = np.empty(len(pi), np.float64)
        for p in np.unique(pi):
            m = pi == p
            out[m] = self.score_plant(int(p))[qi[m]]
        return out
