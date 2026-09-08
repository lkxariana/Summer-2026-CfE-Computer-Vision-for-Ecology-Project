"""Model 3: relation-typed message passing over species, taxa and cell x month nodes (plan §4).

The three axes as graph structure. Nodes: plants and pollinators (frozen BioCLIP-2 text, projected),
genus and family for both kingdoms (learned), cell x month (frozen joint-field coordinate encoder
h(c, month), projected). Relations, each with its own basis-decomposed weight (R-GCN, Schlichtkrull et
al. 2018) and its reverse:

  member      species -> genus, genus -> family
  occurs_in   species -> cell x month, weighted by the species' presence there (from the field's top-k tokens)
  interacts   plant <-> pollinator, TRAINING edges only

Two rounds of message passing. A cold plant has no `interacts` edges, so it is embedded through its
taxon and cell edges alone -- and so that training resembles that situation, every epoch a random 30%
of training plants ("anchors") have their own interaction edges removed before the forward pass
(leave-own-edges-out). The plant meets the pollinator inside the cell x month node, so the field is
integrated jointly, never as a marginal; `month_collapsed` (cell-only nodes) is the marginal control.

Head and loss are model 1's, so the only thing that differs from the retriever is the integration rule.
Plain-torch sparse ops; no PyG/DGL.
"""
from dataclasses import dataclass
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from antheia.embednet import PairHead, MASK_FILL, seed_everything

ROOT = Path(__file__).resolve().parents[2]
TEXT_DIR = Path("/scratch/cher/antheia-data/text_embeddings")


@dataclass
class RGCNConfig:
    field_dir: str
    d: int = 128
    layers: int = 2
    bases: int = 8
    dropout: float = 0.2
    anchor_drop: float = 0.3            # share of training plants whose own interaction edges are dropped each epoch
    cells_per_species: int = 64         # occurs_in edges kept per species
    use_cell_nodes: bool = True         # M3.2 control when False
    use_taxon_nodes: bool = True        # M3.3 control when False
    month_collapsed: bool = False       # M3.5 control: cell-only nodes (12 months summed)
    infer_with_edges: bool = True       # M3.4 on the warm split: use interaction edges at inference
    head_type: str = "concat_bilinear"
    bilinear_rank: int = 64
    use_degree_heads: bool = True
    softmax_weight: float = 0.0
    bce_weight: float = 1.0
    epochs: int = 20
    batch: int = 128
    n_cand: int = 384
    in_batch: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    seed: int = 42
    device: str = "cuda"
    text_variant: str = "bioclip2"
    name: str = "R-GCN over species, taxa and cell x month (ours)"


class RelLayer(nn.Module):
    """One R-GCN layer with basis decomposition: h' = act(LN(W_self h + sum_r A_r h W_r))."""

    def __init__(self, d, n_rel, bases, dropout):
        super().__init__()
        self.self_w = nn.Linear(d, d)
        self.basis = nn.Parameter(torch.randn(bases, d, d) * (1.0 / d ** 0.5))
        self.coef = nn.Parameter(torch.randn(n_rel, bases) * (1.0 / bases ** 0.5))
        self.norm = nn.LayerNorm(d); self.drop = nn.Dropout(dropout)

    def forward(self, h, adjs):
        W = torch.einsum("rb,bij->rij", self.coef, self.basis)            # [n_rel, d, d]
        out = self.self_w(h)
        for r, A in enumerate(adjs):
            if A is not None:
                out = out + torch.sparse.mm(A, h @ W[r])
        return self.drop(F.gelu(self.norm(out)))


def norm_adj(rows, cols, w, n, dev):
    """Sparse [n, n] with rows = targets, cols = sources, row-normalised by weighted in-degree."""
    if len(rows) == 0:
        return None
    w = torch.as_tensor(w, dtype=torch.float32)
    deg = torch.zeros(n).index_add_(0, torch.as_tensor(rows), w)
    vals = w / deg[rows].clamp_min(1e-9)
    idx = torch.stack([torch.as_tensor(rows), torch.as_tensor(cols)]).long()
    return torch.sparse_coo_tensor(idx, vals, (n, n)).coalesce().to(dev)


class RGCNRanker:
    name = "R-GCN (ours)"
    reference = "this work"
    cold_start = True

    def __init__(self, **kw):
        self.cfg = RGCNConfig(**kw)
        if not torch.cuda.is_available():
            self.cfg.device = "cpu"

    # ---- graph construction ----------------------------------------------------------------------
    def _build_graph(self, store, edges):
        cfg = self.cfg; dev = self.dev
        n_p, n_q = len(store.plants), len(store.polls)
        tax = pd.read_parquet(ROOT / "data/features/taxonomy.parquet").drop_duplicates("label").set_index("label")
        def rank_of(names, col):
            return [str(tax[col].get(n, "UNK")) if col in tax else "UNK" for n in names]
        gen_p = [s.split()[0] for s in store.plants]; gen_q = [s.split()[0] for s in store.polls]
        fam_p = rank_of(store.plants, "family"); fam_q = rank_of(store.polls, "family")
        taxa = sorted({("Pg", g) for g in gen_p} | {("Qg", g) for g in gen_q} | {("Pf", f) for f in fam_p} | {("Qf", f) for f in fam_q})
        t2i = {t: i for i, t in enumerate(taxa)}
        n_t = len(taxa) if cfg.use_taxon_nodes else 0
        # cell x month nodes
        H = np.load(Path(cfg.field_dir) / "grid_h.npy")                    # [C, 52, dh]
        C, W, dh = H.shape
        month_of_week = np.clip((np.arange(W) * 7 + 3) // 30, 0, 11)
        Hm = np.stack([H[:, month_of_week == m].mean(1) for m in range(12)], 1)   # [C, 12, dh]
        if cfg.month_collapsed:
            Hm = Hm.mean(1, keepdims=True)                                  # [C, 1, dh]
        n_m = Hm.shape[1]
        n_c = C * n_m if cfg.use_cell_nodes else 0
        off_q, off_t, off_c = n_p, n_p + n_q, n_p + n_q + n_t
        N = n_p + n_q + n_t + n_c
        self.N, self.off = N, (0, off_q, off_t, off_c)
        self.cell_feat = torch.from_numpy(Hm.reshape(-1, dh)).to(dev) if cfg.use_cell_nodes else None
        self.n_t = n_t

        rel = {}
        if cfg.use_taxon_nodes:
            r, c = [], []
            for i, g in enumerate(gen_p): r.append(off_t + t2i[("Pg", g)]); c.append(i)
            for i, g in enumerate(gen_q): r.append(off_t + t2i[("Qg", g)]); c.append(off_q + i)
            rel["member"] = (np.array(r), np.array(c), np.ones(len(r)))
            r2, c2 = [], []
            for g, f in set(zip(gen_p, fam_p)): r2.append(off_t + t2i[("Pf", f)]); c2.append(off_t + t2i[("Pg", g)])
            for g, f in set(zip(gen_q, fam_q)): r2.append(off_t + t2i[("Qf", f)]); c2.append(off_t + t2i[("Qg", g)])
            rel["genus_family"] = (np.array(r2), np.array(c2), np.ones(len(r2)))
        if cfg.use_cell_nodes:
            r, c, w = [], [], []
            for side, off in (("plant", 0), ("poll", off_q)):
                z = np.load(Path(cfg.field_dir) / f"{side}_tokens.npz")
                cell = z["cell"].astype(np.int64); week = z["week"].astype(np.int64); p = np.exp(z["logp"].astype(np.float32))
                month = month_of_week[week] if not cfg.month_collapsed else np.zeros_like(week)
                key = cell * n_m + month                                                 # [n, k]
                for i in range(key.shape[0]):
                    # aggregate token presence per cell-month, keep the strongest cells_per_species
                    agg = {}
                    for kk, pp in zip(key[i].tolist(), p[i].tolist()):
                        agg[kk] = agg.get(kk, 0.0) + pp
                    top = sorted(agg.items(), key=lambda kv: -kv[1])[:cfg.cells_per_species]
                    for kk, pp in top:
                        r.append(off_c + kk); c.append(off + i); w.append(pp)
            rel["occurs_in"] = (np.array(r), np.array(c), np.array(w))
        self.rel_static = rel
        self.rel_names = list(rel) + ["interacts"]
        self.n_rel = 2 * len(self.rel_names)                                  # each relation and its reverse
        pi = store.idx_plants(edges["plant"]); qi = store.idx_polls(edges["pollinator"])
        self.inter = (pi.astype(np.int64), qi.astype(np.int64))

    def _adjs(self, anchors=None, with_edges=True):
        """List of sparse adjacencies in relation order (forward then reverse), interaction edges optionally
        restricted (anchor plants removed) or absent."""
        dev = self.dev; N = self.N; off_q = self.off[1]
        adjs = []
        rels = dict(self.rel_static)
        pi, qi = self.inter
        if with_edges:
            keep = np.ones(len(pi), bool) if anchors is None else ~np.isin(pi, anchors)
            rels["interacts"] = (off_q + qi[keep], pi[keep], np.ones(int(keep.sum())))   # target pollinator <- plant
        else:
            rels["interacts"] = (np.array([], int), np.array([], int), np.array([]))
        for name in self.rel_names:
            r, c, w = rels[name]
            adjs.append(norm_adj(r, c, w, N, dev))
            adjs.append(norm_adj(c, r, w, N, dev))                            # reverse
        return adjs

    def _node_inputs(self):
        x = torch.zeros(self.N, self.cfg.d, device=self.dev)
        n_p, n_q = self.text_p.shape[0], self.text_q.shape[0]
        x[:n_p] = self.proj_text(self.text_p); x[n_p:n_p + n_q] = self.proj_text(self.text_q)
        if self.n_t:
            x[self.off[2]:self.off[2] + self.n_t] = self.tax_emb.weight
        if self.cell_feat is not None:
            x[self.off[3]:] = self.proj_cell(self.cell_feat)
        return x

    def _encode(self, adjs):
        h = self._node_inputs()
        for layer in self.layers:
            h = layer(h, adjs)
        return h

    # ---- training --------------------------------------------------------------------------------
    def fit(self, edges, store):
        cfg = self.cfg; seed_everything(cfg.seed)
        self.dev = dev = cfg.device; self.store = store
        tp = torch.load(TEXT_DIR / f"plants_{cfg.text_variant}.pt", weights_only=False)["embeddings"].float()
        tq = torch.load(TEXT_DIR / f"polls_{cfg.text_variant}.pt", weights_only=False)["embeddings"].float()
        self.text_p = F.normalize(tp, dim=1).to(dev); self.text_q = F.normalize(tq, dim=1).to(dev)
        self._build_graph(store, edges)
        self.proj_text = nn.Linear(tp.shape[1], cfg.d).to(dev)
        self.tax_emb = nn.Embedding(max(self.n_t, 1), cfg.d).to(dev)
        self.proj_cell = nn.Linear(self.cell_feat.shape[1] if self.cell_feat is not None else 1, cfg.d).to(dev)
        self.layers = nn.ModuleList([RelLayer(cfg.d, self.n_rel, cfg.bases, cfg.dropout) for _ in range(cfg.layers)]).to(dev)
        self.head = PairHead(cfg.d, 2 * cfg.d, cfg.dropout, head_type=cfg.head_type, bilinear_rank=cfg.bilinear_rank).to(dev)
        mods = [self.proj_text, self.tax_emb, self.proj_cell, self.layers, self.head]
        if cfg.use_degree_heads:
            self.deg_p = nn.Sequential(nn.Linear(cfg.d, 64), nn.GELU(), nn.Linear(64, 1)).to(dev)
            self.deg_q = nn.Sequential(nn.Linear(cfg.d, 64), nn.GELU(), nn.Linear(64, 1)).to(dev)
            mods += [self.deg_p, self.deg_q]
        params = [p for m in mods for p in m.parameters()]
        opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
        rng = np.random.default_rng(cfg.seed)
        pi, qi = self.inter; n_q = len(store.polls); off_q = self.off[1]
        partners = {}
        for a, b in zip(pi.tolist(), qi.tolist()):
            partners.setdefault(a, set()).add(b)
        train_plants = np.array(sorted(partners))
        cnt = np.bincount(qi, minlength=n_q).astype(np.float64)
        logQ_pop = torch.tensor(np.log(np.maximum(cnt / cnt.sum(), 1e-12)), dtype=torch.float32, device=dev)
        logQ_uni = float(np.log(1.0 / n_q)); n_uni = cfg.n_cand - cfg.in_batch
        pos_weight = torch.tensor(float(cfg.n_cand - 1), device=dev)
        print(f"    rgcn: {self.N:,} nodes ({len(store.plants)} plants, {n_q} polls, {self.n_t} taxa, "
              f"{self.N - self.off[3]} cell-month), {len(pi):,} interaction edges, {self.n_rel} relations", flush=True)

        for ep in range(cfg.epochs):
            for m in mods: m.train()
            anchors = rng.choice(train_plants, int(cfg.anchor_drop * len(train_plants)), replace=False)
            adjs = self._adjs(anchors=anchors, with_edges=True)
            perm = rng.permutation(len(pi)); tot = 0.0; nb = 0; t0 = time.time()
            for s in range(0, len(pi), cfg.batch):
                b = perm[s:s + cfg.batch]; bp, bq = pi[b], qi[b]; B = len(bp)
                sub = rng.choice(B, min(cfg.in_batch, B), replace=False)
                cand = np.concatenate([bq[sub], rng.integers(0, n_q, n_uni)]); Cn = len(cand)
                h = self._encode(adjs)                                            # full graph, [N, d]
                hp = h[torch.from_numpy(bp).long().to(dev)]; hq = h[off_q + torch.from_numpy(cand).long().to(dev)]
                logits = self.head(hp, hq)
                if cfg.use_degree_heads:
                    logits = logits + self.deg_p(hp) + self.deg_q(hq).T
                where = {}
                for j, c in enumerate(cand): where.setdefault(int(c), j)
                pos_col = np.fromiter((where.get(int(q), -1) for q in bq), int, B); valid = pos_col >= 0
                hit = np.zeros((B, Cn), bool)
                for i, pl in enumerate(bp):
                    ps = partners.get(int(pl))
                    if ps: hit[i] = np.fromiter((c in ps for c in cand), bool, Cn)
                hit[np.arange(B), np.clip(pos_col, 0, None)] = False
                hitT = torch.from_numpy(hit).to(dev); tgt = torch.from_numpy(np.clip(pos_col, 0, None)).long().to(dev)
                vm = torch.from_numpy(valid).to(dev)
                if not bool(vm.any()): continue
                loss = 0.0
                if cfg.softmax_weight > 0:
                    corr = torch.cat([logQ_pop[torch.from_numpy(bq[sub]).long().to(dev)], torch.full((n_uni,), logQ_uni, device=dev)])
                    loss = cfg.softmax_weight * F.cross_entropy((logits - corr[None]).masked_fill(hitT, MASK_FILL)[vm], tgt[vm])
                lab = torch.zeros_like(logits); lab[torch.arange(B, device=dev)[vm], tgt[vm]] = 1.0
                loss = loss + cfg.bce_weight * F.binary_cross_entropy_with_logits(logits, lab, weight=(~hitT).float(), pos_weight=pos_weight)
                opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(params, cfg.grad_clip); opt.step()
                tot += float(loss); nb += 1
            sched.step()
            if ep % 5 == 0 or ep == cfg.epochs - 1:
                print(f"    epoch {ep + 1}/{cfg.epochs} loss {tot / max(nb, 1):.4f} ({time.time() - t0:.0f}s)", flush=True)
        for m in mods: m.eval()
        with torch.no_grad():
            self.h_all = self._encode(self._adjs(anchors=None, with_edges=cfg.infer_with_edges))
            n_p = len(store.plants)
            self.hq_all = self.h_all[off_q:off_q + n_q]
            self.dq_all = self.deg_q(self.hq_all).squeeze(-1) if cfg.use_degree_heads else None
        return self

    @torch.no_grad()
    def score_plant(self, p):
        hp = self.h_all[p:p + 1]
        z = self.head(hp, self.hq_all).squeeze(0)
        if self.cfg.use_degree_heads:
            z = z + self.deg_p(hp).squeeze() + self.dq_all
        return z.float().cpu().numpy()

    def score_pairs(self, pi, qi):
        out = np.empty(len(pi), np.float64)
        for p in np.unique(pi):
            m = pi == p; out[m] = self.score_plant(int(p))[qi[m]]
        return out
