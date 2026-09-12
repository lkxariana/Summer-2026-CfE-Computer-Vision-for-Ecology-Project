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

**Score contrast.** The first version of this model produced a nearly uniform ranking: softmax
entropy 9.04 against a 9.48 maximum over 13,124 candidates, and a top-1-to-top-50 separation of 0.97
standard deviations where the boosted ranker reaches 14.35. Ordering was right and contrast was
absent, which is why it led on pooled PR-AUC and trailed on recall at ten -- a metric that depends
entirely on the head of the list being sharply separated. A learned temperature was tried and is
provably inert: scaling every logit by one positive constant is a monotone map, so it leaves every
within-plant ranking, and therefore every retrieval metric, bit-identical. Contrast and ordering are
separate properties, and only ordering drives recall@k. What does move the metric is the weight on
the binary term, which with 383 negatives per positive pulls every logit toward a common value, so
`bce_weight` is swept rather than fixed.

**The wide path.** Handing the taxonomic affinity count to the pair MLP left retrieval unchanged
(p=0.99) and collapsed pooled PR-AUC from 0.165 to 0.073: the deep path smooths exactly the sparsity
that makes the feature informative. Wide & Deep (Cheng et al. 2016) is explicit that a sparse
cross-product feature belongs on a *linear* path added to the logit, bypassing the network, because
"deep neural networks with embeddings can over-generalize when the interactions are sparse and
high-rank" -- which describes a network at 0.12% connectance exactly. `use_wide_affinity` puts the
affinity, co-occurrence and degree terms on that linear path instead.

**Cross network (DCN-V2).** The wide path memorises crosses a person chose. DCN-V2 (Wang et al.,
WWW 2021) makes the point that choosing them "falls back to the feature engineering problem for
linear models", and replaces the choice with explicit bounded-degree crossing: each layer computes
x_{l+1} = x_0 * (W_l x_l + b_l) + x_l, so l layers span interactions up to degree l+1 with a weight
*matrix* rather than DCN-V1's vector. It runs in parallel with the deep path and their outputs are
concatenated, the arrangement Google productionised in place of Wide & Deep.

Crossing is applied to the encoded pair, [h_p, h_q] at 512 dimensions, rather than to the raw 2,082
input dimensions: the latter is 4.3M parameters per layer and 2e11 multiply-adds per training step,
the former 262k and 1.3e10.

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
from antheia.paths import TEXT_DIR as _TEXT_DIR, FEATURES
TEXT_DIR = str(_TEXT_DIR)


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
    softmax_weight: float = 1.0      # within-plant softmax term; 0 = pooled BCE only (plan M1.1)
    use_wide_affinity: bool = False  # Wide & Deep: the taxonomic affinity counts on a linear path, added to the logit
    cross_layers: int = 0            # DCN-V2 depth over the encoded pair; 0 disables
    use_degree_offset: bool = False  # A3: log-degree as an explicit term with a learned coefficient
    blocks: tuple = ("text", "surface", "pca", "scale")
    blocks_q: tuple = None           # pollinator-side blocks; defaults to `blocks`
    field_impute: bool = True        # "field": keep text-imputed rows, or zero them (trained vectors only)
    field_path: str = str(FEATURES / "poll_field.npy")
    seed: int = 42
    device: str = "cuda"
    text_dir: str = TEXT_DIR
    text_variant: str = "bioclip2"   # "bioclip2" = bare binomial; "bioclip2_hier" = full Linnaean
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


class PairHead(nn.Module):
    """[h_p, h_q, h_p*h_q, |h_p-h_q|] -> logit.

    `head_type` and `bilinear_rank` are accepted and ignored: the concatenation-plus-bilinear variant
    was an ablation and the elementwise form is the only one used by a reported run.
    """

    def __init__(self, d_model, hidden, dropout, n_extra=0, cross_layers=0,
                 head_type="elementwise", bilinear_rank=64):
        super().__init__()
        self.cross = CrossNet(2 * d_model, cross_layers) if cross_layers else None
        cross_dim = 2 * d_model if cross_layers else 0
        in_dim = 4 * d_model + n_extra + cross_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1))
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.mlp[-1].weight, std=0.01)
        self.logit_scale = nn.Parameter(torch.tensor(0.0))   # exp(0)=1 at init

    def forward(self, hp, hq, extra=None):
        """hp [B, d], hq [C, d], optional extra [B, C, n_extra] -> [B, C]."""
        B, C, d = hp.shape[0], hq.shape[0], hp.shape[1]
        a = hp.unsqueeze(1).expand(B, C, d)
        b = hq.unsqueeze(0).expand(B, C, d)
        parts = [a, b, a * b, (a - b).abs()]
        if self.cross is not None:
            pair = torch.cat([a, b], -1)                          # [B, C, 2d]
            parts.append(self.cross(pair.reshape(-1, pair.shape[-1])).view_as(pair))
        if extra is not None:
            parts.append(extra)                                   # [B, C, n_extra]
        out = self.mlp(torch.cat(parts, -1)) * self.logit_scale.exp()   # [B, C, 1]
        return out.squeeze(-1)


class CrossNet(nn.Module):
    """DCN-V2 cross layers: x_{l+1} = x_0 * (W_l x_l + b_l) + x_l, weight matrix per layer."""

    def __init__(self, dim, depth):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(depth)])
        for lin in self.layers:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)

    def forward(self, x0):
        """x0 [M, dim] -> [M, dim]."""
        x = x0
        for lin in self.layers:
            x = x0 * lin(x) + x
        return x


class WidePath(nn.Module):
    """Linear model on the sparse cross-product features, added to the logit.

    Deliberately linear and deliberately not routed through the encoder: memorisation of
    "this pollinator was recorded on this plant genus" is the part a deep path degrades.
    """

    def __init__(self, n_feat):
        super().__init__()
        self.lin = nn.Linear(n_feat, 1)
        nn.init.zeros_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)

    def forward(self, x):
        return self.lin(x).squeeze(-1)


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
        tp = torch.load(td / f"plants_{cfg.text_variant}.pt", weights_only=False)["embeddings"].numpy()
        tq = torch.load(td / f"polls_{cfg.text_variant}.pt", weights_only=False)["embeddings"].numpy()
        unit = lambda x: x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
        svd = lambda X: TruncatedSVD(15, random_state=cfg.seed).fit_transform(X.astype(np.float32))

        p_all = {"text": unit(tp), "surface": store.plant_proj,
                 "pca": svd(store.F),
                 "scale": np.stack([np.log1p(store.Frs), (store.Frs > 0).astype(np.float32)], 1)}
        q_all = {"text": unit(tq), "surface": store.poll_proj,
                 "pca": svd(store.P),
                 "scale": np.stack([np.log1p(store.Prs), (store.Prs > 0).astype(np.float32)], 1)}
        blocks_q = cfg.blocks_q or cfg.blocks

        def load_field(side, n):
            # a species' learned spatio-temporal influence: the per-species head of a SINR-style
            # field model, not a projection of its output surface. `field_path` is either the
            # pollinator file (pipelines/features/build_field_embeddings.py) or a directory holding
            # plant_field.npy / poll_field.npy from pipelines/sdm/train_joint_field.py.
            fp = Path(cfg.field_path)
            f = fp / f"{side}_field.npy" if fp.is_dir() else fp
            fld = np.load(f).astype(np.float32)
            if not cfg.field_impute:
                fld = fld * np.load(f.with_name(f.stem + "_direct.npy"))[:, None]
            assert len(fld) == n, (f, fld.shape, n)
            return fld

        if "field" in cfg.blocks:
            p_all["field"] = load_field("plant", len(store.plants))
        if "field" in blocks_q:
            q_all["field"] = load_field("poll", len(store.polls))
        self.P_blocks = [T(p_all[b]) for b in cfg.blocks]
        self.Q_blocks = [T(q_all[b]) for b in blocks_q]
        dims = [b.shape[1] for b in self.P_blocks]
        dims_q = [b.shape[1] for b in self.Q_blocks]
        self.n_p, self.n_q = len(store.plants), len(store.polls)

        self.enc_p = BlockEncoder(dims, cfg.d_model, cfg.hidden, cfg.dropout).to(dev)
        self.enc_q = BlockEncoder(dims_q, cfg.d_model, cfg.hidden, cfg.dropout).to(dev)
        self.head = PairHead(cfg.d_model, cfg.hidden, cfg.dropout,
                             cross_layers=cfg.cross_layers).to(dev)
        mods = [self.enc_p, self.enc_q, self.head]

        pi = store.idx_plants(edges["plant"]); qi = store.idx_polls(edges["pollinator"])

        if cfg.use_wide_affinity:
            gen_a = np.array([x.split()[0] for x in store.plants])
            fam_a = np.array([store.family.get(x, "UNK") for x in store.plants])
            ga = {g: i for i, g in enumerate(sorted(set(gen_a)))}
            fa = {f: i for i, f in enumerate(sorted(set(fam_a)))}
            self.aff_GI = np.array([ga[g] for g in gen_a]); self.aff_FI = np.array([fa[f] for f in fam_a])
            Cg = np.zeros((self.n_q, len(ga)), np.float32); Cf = np.zeros((self.n_q, len(fa)), np.float32)
            np.add.at(Cg, (qi, self.aff_GI[pi]), 1.0); np.add.at(Cf, (qi, self.aff_FI[pi]), 1.0)
            self.Cg_t, self.Cf_t = T(Cg), T(Cf)
            self.N_t = T(np.asarray(store.N_full, dtype=np.float32))

        # species-level documentation effort only: never the pair's ecology
        self.bias_p = T(np.stack([np.log1p(store.Frs), (store.FCo.sum(1) > 0).astype(np.float32)], 1))
        self.bias_q = T(np.stack([np.log1p(store.Prs), (store.ACo.sum(1) > 0).astype(np.float32)], 1))
        self.bias = BiasHead(self.bias_p.shape[1] + self.bias_q.shape[1]).to(dev)
        mods.append(self.bias)

        if cfg.use_wide_affinity or cfg.use_degree_offset:
            n_wide = (3 if cfg.use_wide_affinity else 0) + (2 if cfg.use_degree_offset else 0)
            self.wide = WidePath(n_wide).to(dev)
            mods.append(self.wide)
            self.deg_q = T(np.log1p(np.bincount(qi, minlength=self.n_q).astype(np.float32)))
            self.deg_p = T(np.log1p(np.bincount(pi, minlength=self.n_p).astype(np.float32)))

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
                uni = rng.integers(0, self.n_q, n_uni)
                cand = np.concatenate([bq[sub], uni])
                C = len(cand)
                tp_i = torch.from_numpy(np.ascontiguousarray(bp)).long().to(dev)
                tc_i = torch.from_numpy(np.ascontiguousarray(cand)).long().to(dev)

                hp = self.enc_p([blk[tp_i] for blk in self.P_blocks])      # [B, d]
                hq = self.enc_q([blk[tc_i] for blk in self.Q_blocks])      # [C, d]
                logits = self.head(hp, hq)                                 # [B, C]
                if cfg.use_wide_affinity or cfg.use_degree_offset:
                    logits = logits + self.wide(self._wide_feats(tp_i, tc_i))
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

                # logQ correction: the in-batch candidates are drawn with probability proportional to
                # their recorded degree, the uniform ones uniformly
                corr = torch.cat([logQ_pop[torch.from_numpy(np.ascontiguousarray(bq[sub])).long().to(dev)],
                                  torch.full((n_uni,), logQ_uni, device=dev)])
                z = logits - corr[None, :]
                loss = cfg.softmax_weight * F.cross_entropy(z.masked_fill(hitT, MASK_FILL)[vm], tgt[vm])
                lab = torch.zeros_like(logits)
                lab[torch.arange(B, device=dev)[vm], tgt[vm]] = 1.0
                loss = loss + cfg.bce_weight * F.binary_cross_entropy_with_logits(
                    logits, lab, weight=(~hitT).float(), pos_weight=pos_weight)

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
        return self


    def _wide_feats(self, pi, qi):
        """pi [B], qi [C] -> [B, C, n_wide]; sparse cross-products kept linear."""
        B, C = len(pi), len(qi)
        parts = []
        if self.cfg.use_wide_affinity:
            parts += [torch.log1p(self.Cg_t[qi][:, self.aff_GI[pi.cpu().numpy()]].T),
                      torch.log1p(self.Cf_t[qi][:, self.aff_FI[pi.cpu().numpy()]].T),
                      torch.log1p(self.N_t[pi][:, qi])]
        if self.cfg.use_degree_offset:
            parts += [self.deg_q[qi].view(1, C).expand(B, C),
                      self.deg_p[pi].view(B, 1).expand(B, C)]
        return torch.stack(parts, -1)

    def score_plant(self, p, chunk=4096):
        out = np.empty(self.n_q, np.float32)
        with torch.no_grad():
            hp = self.hp_all[p:p + 1]
            for s in range(0, self.n_q, chunk):
                hq = self.hq_all[s:s + chunk]
                qs_all = torch.arange(s, s + len(hq), device=self.dev)
                z = self.head(hp, hq)
                if self.cfg.use_wide_affinity or self.cfg.use_degree_offset:
                    z = z + self.wide(self._wide_feats(
                        torch.tensor([p], device=self.dev, dtype=torch.long), qs_all))
                out[s:s + len(hq)] = z.squeeze(0).float().cpu().numpy()
        return out

    def score_pairs(self, pi, qi):
        out = np.empty(len(pi), np.float64)
        for p in np.unique(pi):
            m = pi == p
            out[m] = self.score_plant(int(p))[qi[m]]
        return out
