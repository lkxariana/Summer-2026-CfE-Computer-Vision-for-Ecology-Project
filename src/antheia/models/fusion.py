"""Single-stream fusion re-ranker over identity and field tokens (plan §3, model 2).

A pair (plant p, pollinator q) is one token sequence

    [CLS] [id_p] [f_p,1 .. f_p,k] [id_q] [f_q,1 .. f_q,k]

  identity token  projected frozen BioCLIP-2 text (768 -> d)
  field token     projected [h(c, w) ; log p_s(c, w)] (256 + 1 -> d), where h is the frozen joint-field
                  coordinate encoder over the grid and p_s the species' presence there; the k tokens are
                  the species' top-k (cell, week) positions with a per-month floor (train_field_v2.py)
  type embedding  one of four, added: plant/pollinator x identity/field. Role, not kingdom -- BioCLIP
                  already knows the kingdom.

No positional embeddings: a field token's coordinates are its content. L pre-norm transformer layers of
self-attention over the whole sequence (single-stream; ViLT, UNITER), then

    logit(p, q) = s_retriever(p, q) + MLP(CLS)

so the model learns a correction to the retriever and cannot do worse than it at initialisation.
Training pairs are the retriever's top-K per training plant, the plant's positives, and a few random
pollinators; the loss is pooled binary cross-entropy. At inference the retriever's top-K per query is
re-scored and everything outside keeps the retriever score, shifted below the re-ranked block.

Marginal-token controls (plan M2.7) use the same class with token caches built from the space or time
marginal of the field instead of the joint (pipelines/features/build_field_tokens.py).
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from antheia import negpool

from antheia.paths import REPO_ROOT as ROOT
from antheia.paths import TEXT_DIR, FIELD_DIR


def seed_everything(seed):
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


@dataclass
class FusionConfig:
    field_dir: str = str(FIELD_DIR)                       # joint_field/field_v2 (grid_h.npy, {plant,poll}_tokens*.npz)
    token_variant: str = "joint"        # "joint" | "space" | "time"  (marginal-token controls)
    k: int = 128                        # field tokens per species (<= cached top-k)
    d: int = 192
    layers: int = 3
    heads: int = 4
    dropout: float = 0.1
    identity_only: bool = False         # M2.1: CLS + two identity tokens, no field tokens
    presence_mode: str = "feature"      # "feature": log p in the token; "bias": additive attention bias instead
    epochs: int = 8
    lr: float = 3e-4
    weight_decay: float = 1e-2
    batch_pairs: int = 256
    pos_per_plant: int = 16             # positives sampled per plant per epoch
    hard_per_plant: int = 16            # negatives from the retriever's top-K
    rand_per_plant: int = 8             # uniform negatives
    cooc_per_plant: int = 0             # negatives among pollinators that co-occur with the plant (N > 0): the within-site regime
    genus_tokens: int = 0               # k_g tokens: pollinators recorded with the plant's genus, the trees' lookup as a set
    genus_loo: str = "pair"             # "plant": the plant's own edges removed from its profile (no warm information);
                                        # "pair": the scored candidate's token is masked entirely (hides congener evidence too);
                                        # "own": only the plant's OWN contribution to the candidate's count is subtracted --
                                        #        congener evidence stays, which is what the trees' affinity table sees
    base_affine: bool = False           # logit = a * s_retriever + b + delta, a,b learned (identity at init); needed when the
                                        # retriever's scores are not logits (trees / quantile-mapped scores)
    grad_clip: float = 1.0
    seed: int = 42
    device: str = "cuda"
    text_variant: str = "bioclip2"
    id_source: str = "text"             # "text": frozen BioCLIP-2 names (768) | "retriever_proj": the retriever's projected names (d_r) |
                                        # "retriever_h": the retriever's graph output vectors h_p, h_q (d_r); arrays passed via fit(id_override=)
    name: str = "Fusion re-ranker over identity and field tokens (ours)"


class Fusion(nn.Module):
    def __init__(self, cfg: FusionConfig, d_text=768, d_field=256):
        super().__init__()
        d = cfg.d
        self.cfg = cfg
        self.proj_id = nn.Linear(d_text, d)
        self.proj_field = nn.Linear(d_field + (1 if cfg.presence_mode == "feature" else 0), d)
        self.type_emb = nn.Embedding(5, d)              # 0 plant-id, 1 plant-field, 2 poll-id, 3 poll-field, 4 genus-profile
        self.proj_gcnt = nn.Linear(1, d)                # log count of the genus-profile pollinator
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        layer = nn.TransformerEncoderLayer(d, cfg.heads, 4 * d, cfg.dropout, activation="gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, cfg.layers)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(d, 1))
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)   # start as the retriever
        self.pres_scale = nn.Parameter(torch.tensor(1.0))
        self.base_a = nn.Parameter(torch.tensor(1.0)); self.base_b = nn.Parameter(torch.tensor(0.0))

    def forward(self, id_p, fld_p, lp_p, id_q, fld_q, lp_q, g_txt=None, g_cnt=None, g_pad=None):
        """id_*: [B, 768]; fld_*: [B, k, 256]; lp_*: [B, k] log presence; optional genus-profile tokens
        g_txt [B, kg, 768], g_cnt [B, kg] log counts, g_pad [B, kg] True where padded. Returns delta logit [B]."""
        B, k = fld_p.shape[0], fld_p.shape[1]
        te = self.type_emb.weight
        toks = [self.cls.expand(B, 1, -1), (self.proj_id(id_p) + te[0])[:, None], ]
        pad = None
        if g_txt is not None and g_txt.shape[1] > 0:
            toks.append(self.proj_id(g_txt) + self.proj_gcnt(g_cnt[..., None]) + te[4])
            pad = g_pad
        if k > 0:
            if self.cfg.presence_mode == "feature":
                fp = self.proj_field(torch.cat([fld_p, lp_p[..., None]], -1)) + te[1]
                fq = self.proj_field(torch.cat([fld_q, lp_q[..., None]], -1)) + te[3]
            else:
                fp = self.proj_field(fld_p) + te[1]; fq = self.proj_field(fld_q) + te[3]
            toks += [fp, (self.proj_id(id_q) + te[2])[:, None], fq]
        else:
            toks += [(self.proj_id(id_q) + te[2])[:, None]]
        x = torch.cat(toks, 1)                                                       # [B, 2k+3(+kg), d]
        key_pad = None
        if pad is not None:
            key_pad = torch.zeros(B, x.shape[1], dtype=torch.bool, device=x.device)
            key_pad[:, 2:2 + pad.shape[1]] = pad
        mask = None
        if k > 0 and self.cfg.presence_mode == "bias":
            # additive key bias: low-presence tokens are attended to less. Built as a float mask [B*heads, T, T].
            T = x.shape[1]
            bias = torch.zeros(B, T, device=x.device)
            bias[:, 2:2 + k] = lp_p * self.pres_scale; bias[:, 3 + k:3 + 2 * k] = lp_q * self.pres_scale
            mask = bias[:, None, :].expand(B, T, T).repeat_interleave(self.cfg.heads, 0)
        h = self.enc(x, mask=mask, src_key_padding_mask=key_pad)
        return self.head(self.norm(h[:, 0])).squeeze(-1)


class FusionReranker:
    name = "Fusion re-ranker (ours)"
    reference = "this work"
    cold_start = True

    def __init__(self, **kw):
        self.cfg = FusionConfig(**kw)
        if not torch.cuda.is_available():
            self.cfg.device = "cpu"

    # ---- caches -------------------------------------------------------------------------------
    def _load_inputs(self, store):
        cfg = self.cfg; dev = self.dev
        fd = Path(cfg.field_dir)
        tp = torch.load(TEXT_DIR / f"plants_{cfg.text_variant}.pt", weights_only=False)["embeddings"].float()
        tq = torch.load(TEXT_DIR / f"polls_{cfg.text_variant}.pt", weights_only=False)["embeddings"].float()
        self.text_p = F.normalize(tp, dim=1).to(dev); self.text_q = F.normalize(tq, dim=1).to(dev)
        if getattr(self, "_id_override", None) is not None:
            a, b = self._id_override
            self.text_p = F.normalize(torch.as_tensor(np.asarray(a, np.float32)), dim=1).to(dev)
            self.text_q = F.normalize(torch.as_tensor(np.asarray(b, np.float32)), dim=1).to(dev)
        H = np.load(fd / "grid_h.npy")                                              # [C, 52, 256]
        self.C, self.W, dh = H.shape
        suffix = "" if cfg.token_variant == "joint" else f"_{cfg.token_variant}"
        if cfg.token_variant == "joint":
            self.H = torch.from_numpy(H.reshape(-1, dh)).to(dev)                    # index = cell*52 + week
        elif cfg.token_variant == "space":
            self.H = torch.from_numpy(H.mean(1)).to(dev)                            # [C, 256]   weeks collapsed
        else:
            self.H = torch.from_numpy(H.mean(0)).to(dev)                            # [52, 256]  cells collapsed
        tok = {}
        for side in ("plant", "poll"):
            z = np.load(fd / f"{side}_tokens{suffix}.npz")
            k = min(cfg.k, z["logp"].shape[1])
            if cfg.token_variant == "joint":
                idx = z["cell"].astype(np.int64)[:, :k] * 52 + z["week"].astype(np.int64)[:, :k]
            elif cfg.token_variant == "space":
                idx = z["cell"].astype(np.int64)[:, :k]
            else:
                idx = z["week"].astype(np.int64)[:, :k]
            tok[side] = (torch.from_numpy(idx).to(dev), torch.from_numpy(z["logp"].astype(np.float32)[:, :k]).to(dev))
        self.tok_p, self.tok_q = tok["plant"], tok["poll"]
        assert len(self.text_p) == len(store.plants) and len(self.text_q) == len(store.polls)
        assert self.tok_p[0].shape[0] == len(store.plants) and self.tok_q[0].shape[0] == len(store.polls)

    def _genus(self, pi, qi=None):
        if self.cfg.genus_tokens <= 0:
            return ()
        gi = self.g_idx[pi]; gc = self.g_cnt[pi]                       # [B, kg], -1 where padded
        pad = gi < 0
        if qi is not None and self.cfg.genus_loo == "pair":
            pad = pad | (gi == qi[:, None])                            # candidate token masked entirely
        elif qi is not None and self.cfg.genus_loo == "own":
            # subtract the plant's own count for the candidate; mask only if nothing from congeners remains
            hit = gi == qi[:, None]                                    # [B, kg]
            if hit.any():
                own = self.own_cnt[pi, qi]                             # [B] own count of (p, q), dense lookup
                adj = torch.log1p(torch.clamp(torch.expm1(gc) - own[:, None], min=0.0))
                gc = torch.where(hit, adj, gc)
                pad = pad | (hit & (adj <= 0))
        return (self.text_q[gi.clamp_min(0)], gc, pad)

    def _batch(self, pi, qi):
        """Gather tensors for pairs (pi, qi): index tensors on device."""
        k = 0 if self.cfg.identity_only else self.tok_p[0].shape[1]
        if k == 0:
            e = torch.empty(len(pi), 0, self.H.shape[1], device=self.dev); z = torch.empty(len(pi), 0, device=self.dev)
            return (self.text_p[pi], e, z, self.text_q[qi], e, z) + self._genus(pi, qi)
        ip, lpp = self.tok_p[0][pi], self.tok_p[1][pi]; iq, lpq = self.tok_q[0][qi], self.tok_q[1][qi]
        return (self.text_p[pi], self.H[ip], lpp, self.text_q[qi], self.H[iq], lpq) + self._genus(pi, qi)

    # ---- training -------------------------------------------------------------------------------
    def fit(self, edges, store, retriever_scores_train, retriever_cands_train, train_plants_idx, id_override=None):
        """retriever_scores_train / cands_train: [n_train_plants, K] scores and pollinator indices from the
        retriever, aligned with train_plants_idx (plant indices into store)."""
        cfg = self.cfg; seed_everything(cfg.seed)
        self.dev = dev = cfg.device
        self.store = store
        self._id_override = id_override
        self._load_inputs(store)
        self.net = Fusion(cfg, d_text=self.text_p.shape[1], d_field=self.H.shape[1]).to(dev)
        opt = torch.optim.AdamW(self.net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        rng = np.random.default_rng(cfg.seed)

        pi = store.idx_plants(edges["plant"]); qi = store.idx_polls(edges["pollinator"])
        partners = {}
        for a, b in zip(pi.tolist(), qi.tolist()):
            partners.setdefault(a, set()).add(b)
        row_of = {int(p): i for i, p in enumerate(train_plants_idx)}
        # retriever score lookup for arbitrary (p, q): top-K table, else the plant's K-th score minus margin
        K = retriever_cands_train.shape[1]
        self.ret_floor = retriever_scores_train[:, -1] - 1.0
        cand_pos = {}
        for i, p in enumerate(train_plants_idx):
            cand_pos[int(p)] = {int(q): float(s) for q, s in zip(retriever_cands_train[i], retriever_scores_train[i])}
        plants = [p for p in train_plants_idx.tolist() if partners.get(p)]
        if cfg.genus_tokens > 0:
            # genus profile: pollinators recorded with the plant's genus in training, leave-one-out for training plants;
            # a held-out plant simply gets its genus's full profile (empty if the genus is unseen)
            gen = np.array([x.split()[0] for x in store.plants]); g2i = {g: i for i, g in enumerate(sorted(set(gen)))}
            pg = np.array([g2i[g] for g in gen])
            gcount = {}
            for a, b in zip(pi.tolist(), qi.tolist()):
                gcount.setdefault(pg[a], {}); gcount[pg[a]][b] = gcount[pg[a]].get(b, 0) + 1
            own = {}
            for a, b in zip(pi.tolist(), qi.tolist()):
                own.setdefault(a, {}); own[a][b] = own[a].get(b, 0) + 1
            kg = cfg.genus_tokens
            g_idx = np.full((len(store.plants), kg), -1, np.int64); g_cnt = np.zeros((len(store.plants), kg), np.float32)
            for a in range(len(store.plants)):
                c = dict(gcount.get(pg[a], {}))
                if cfg.genus_loo == "plant":
                    for b, n in own.get(a, {}).items():
                        c[b] = c.get(b, 0) - n
                        if c[b] <= 0: c.pop(b, None)
                top = sorted(c.items(), key=lambda kv: -kv[1])[:kg]
                for j, (b, n) in enumerate(top):
                    g_idx[a, j] = b; g_cnt[a, j] = np.log1p(n)
            self.g_idx = torch.from_numpy(g_idx).to(dev); self.g_cnt = torch.from_numpy(g_cnt).to(dev)
            if cfg.genus_loo == "own":
                oc = torch.zeros(len(store.plants), len(store.polls), dtype=torch.float16)
                for a, d_ in own.items():
                    for b, n in d_.items():
                        oc[a, b] = n
                self.own_cnt = oc.to(dev)                              # [P, Q] fp16 (~290 MB on GPU)
            print(f"    genus tokens: {int((g_idx[:, 0] >= 0).sum())}/{len(store.plants)} plants have a non-empty profile", flush=True)
        cooc = None
        if cfg.cooc_per_plant > 0:
            N = store.N_full                                              # [P, Q] shared cells
            cooc = {p: np.flatnonzero(N[p] > 0) for p in plants}
        steps_per_epoch = math.ceil(len(plants) * (cfg.pos_per_plant + cfg.hard_per_plant + cfg.rand_per_plant + cfg.cooc_per_plant) / cfg.batch_pairs)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg.lr, total_steps=cfg.epochs * steps_per_epoch, pct_start=0.1)
        n_q = len(store.polls)
        print(f"    fusion: {len(plants):,} train plants, K={K}, {steps_per_epoch} steps/epoch, tokens/pair "
              f"{3 if cfg.identity_only else 3 + 2 * min(cfg.k, self.tok_p[0].shape[1])}", flush=True)

        for ep in range(cfg.epochs):
            self.net.train(); t0 = time.time(); tot = 0.0; nb = 0
            # sample pairs
            P, Q, Yl, Sr = [], [], [], []
            for p in plants:
                ps = np.fromiter(partners[p], int)
                pos = rng.choice(ps, min(cfg.pos_per_plant, len(ps)), replace=False)
                cands = np.array([c for c in cand_pos[p] if c not in partners[p]], int)
                hard = rng.choice(cands, min(cfg.hard_per_plant, len(cands)), replace=False) if len(cands) else np.array([], int)
                rnd = negpool.sample(rng, cfg.rand_per_plant, n_q); rnd = rnd[~np.isin(rnd, ps)]
                if cooc is not None and len(cooc[p]):
                    cc = cooc[p][~np.isin(cooc[p], ps)]
                    co = rng.choice(cc, min(cfg.cooc_per_plant, len(cc)), replace=False) if len(cc) else np.array([], int)
                    rnd = np.concatenate([rnd, co])
                qs = np.concatenate([pos, hard, rnd]); ys = np.r_[np.ones(len(pos)), np.zeros(len(hard) + len(rnd))]
                sr = np.array([cand_pos[p].get(int(q), self.ret_floor[row_of[p]]) for q in qs])
                P.append(np.full(len(qs), p)); Q.append(qs); Yl.append(ys); Sr.append(sr)
            P, Q, Yl, Sr = map(np.concatenate, (P, Q, Yl, Sr))
            perm = rng.permutation(len(P))
            pos_w = torch.tensor(float((Yl == 0).sum() / max((Yl == 1).sum(), 1)), device=dev)
            for s in range(0, len(perm), cfg.batch_pairs):
                b = perm[s:s + cfg.batch_pairs]
                tpi = torch.from_numpy(P[b]).long().to(dev); tqi = torch.from_numpy(Q[b]).long().to(dev)
                delta = self.net(*self._batch(tpi, tqi))
                base = torch.from_numpy(Sr[b]).float().to(dev)
                if cfg.base_affine:
                    base = self.net.base_a * base + self.net.base_b
                logit = base + delta
                loss = F.binary_cross_entropy_with_logits(logit, torch.from_numpy(Yl[b]).float().to(dev), pos_weight=pos_w)
                opt.zero_grad(set_to_none=True); loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), cfg.grad_clip); opt.step(); sched.step()
                tot += float(loss); nb += 1
            print(f"    epoch {ep + 1}/{cfg.epochs} loss {tot / max(nb, 1):.4f} ({time.time() - t0:.0f}s)", flush=True)
        self.net.eval()
        return self

    # ---- inference -----------------------------------------------------------------------------
    @torch.no_grad()
    def rerank(self, plant_idx, ret_scores_row, ret_cands_row, n_cand):
        """Full score vector over n_cand candidates for one plant: re-ranked top-K, retriever elsewhere."""
        out = np.full(n_cand, -np.inf, np.float32)
        K = len(ret_cands_row)
        tpi = torch.full((K,), int(plant_idx), dtype=torch.long, device=self.dev)
        tqi = torch.from_numpy(np.asarray(ret_cands_row)).long().to(self.dev)
        delta = torch.cat([self.net(*self._batch(tpi[i:i + 512], tqi[i:i + 512])) for i in range(0, K, 512)]).cpu().numpy()
        base = np.asarray(ret_scores_row, np.float32)
        if self.cfg.base_affine:
            base = float(self.net.base_a) * base + float(self.net.base_b)
        new = base + delta
        return new, tqi.cpu().numpy()
