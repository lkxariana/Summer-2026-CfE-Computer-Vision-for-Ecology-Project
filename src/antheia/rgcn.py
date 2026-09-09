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
from antheia import negpool

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
    anchor_side: str = "plant"          # "plant" (M3.1: plants only) | "both" (R3: pollinator anchors too, so cold pollinators are rehearsed)
    cells_per_species: int = 64         # occurs_in edges kept per species
    use_cell_nodes: bool = True         # M3.2 control when False
    use_taxon_nodes: bool = True        # M3.3 control when False
    month_collapsed: bool = False       # M3.5 control: cell-only nodes (12 months summed)
    infer_with_edges: bool = True       # M3.4 on the warm split: use interaction edges at inference
    warm_residual: bool = False         # R1: free per-species vector added to the node input of species WITH training edges;
    res_drop: float = 0.5               #     zeroed for anchors and a random `res_drop` share of species each epoch (DropoutNet)
    genus_edges: bool = False           # R2: direct genus <-> partner relations weighted by log1p(training count), leave-own-edges-out
    aggregation: str = "mean"           # "mean" (R-GCN) | "attention" (per-relation GAT-style attention, SimpleHGN-like control)
    pair_stat: str = "none"             # R4: explicit co-presence statistic of the pair into the head: joint | space | time | scalar
    degree_encoding: bool = False       # R5: Graphormer-style centrality encoding -- log(1 + in-degree per relation) added to node inputs
    presence_input: str = "none"        # R6: per-species presence embedding added to the species node input: "field" (SDM species vector,
                                        #     256-D, whose dot with h(c,w) is the presence surface) | "surface" (SVD projection of the full surface)
    presence_fusion: str = "add"        # R6b: "add" (projected presence added to projected text) | "concat" ([text || presence] -> MLP -> d)
    pres_bilinear_rank: int = 0         # R6c: > 0 adds a learned low-rank co-presence metric u_p^T W u_q to the head (generalises R4)
    cooc_neg_frac: float = 0.0          # R7: share of the uniform negatives replaced by in-batch co-occurrence negatives (pollinators that share
                                        #     a cell with a batch plant), so training rewards the within-site discrimination as well
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

    def __init__(self, d, n_rel, bases, dropout, aggregation="mean"):
        super().__init__()
        self.self_w = nn.Linear(d, d)
        self.basis = nn.Parameter(torch.randn(bases, d, d) * (1.0 / d ** 0.5))
        self.coef = nn.Parameter(torch.randn(n_rel, bases) * (1.0 / bases ** 0.5))
        self.norm = nn.LayerNorm(d); self.drop = nn.Dropout(dropout)
        self.aggregation = aggregation
        if aggregation == "attention":
            # GAT-style decomposed attention per relation: e_ij = LeakyReLU(a_dst_r . W_r h_i + a_src_r . W_r h_j) + log w_ij,
            # softmax over the incoming edges of i within the relation (Velickovic et al. 2018; SimpleHGN, Lv et al. 2021).
            self.a_src = nn.Parameter(torch.randn(n_rel, d) * 0.1)
            self.a_dst = nn.Parameter(torch.randn(n_rel, d) * 0.1)

    def forward(self, h, adjs):
        W = torch.einsum("rb,bij->rij", self.coef, self.basis)            # [n_rel, d, d]
        out = self.self_w(h)
        for r, A in enumerate(adjs):
            if A is None:
                continue
            hW = h @ W[r]
            if self.aggregation == "mean":
                out = out + torch.sparse.mm(A, hW)
            else:
                idx = A.indices(); rows, cols = idx[0], idx[1]
                e = F.leaky_relu((hW @ self.a_dst[r])[rows] + (hW @ self.a_src[r])[cols], 0.2) + torch.log(A.values() + 1e-9)
                emax = torch.full((h.shape[0],), -1e30, device=h.device).scatter_reduce(0, rows, e, "amax", include_self=True)
                ex = torch.exp(e - emax[rows])
                den = torch.zeros(h.shape[0], device=h.device).index_add_(0, rows, ex)
                alpha = ex / den[rows]
                out = out + torch.sparse.mm(torch.sparse_coo_tensor(idx, alpha, A.shape), hW)
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
        if cfg.genus_edges:
            assert cfg.use_taxon_nodes, "genus_edges needs taxon nodes"
            self.gen_p_idx = np.array([t2i[("Pg", g)] for g in gen_p], np.int64)
            self.gen_q_idx = np.array([t2i[("Qg", g)] for g in gen_q], np.int64)
            self.rel_names += ["genus_p_partner", "genus_q_partner"]
        self.n_rel = 2 * len(self.rel_names)                                  # each relation and its reverse
        pi = store.idx_plants(edges["plant"]); qi = store.idx_polls(edges["pollinator"])
        self.inter = (pi.astype(np.int64), qi.astype(np.int64))

    def _adjs(self, anchors=None, with_edges=True, anchors_q=None):
        """List of sparse adjacencies in relation order (forward then reverse), interaction edges optionally
        restricted (anchor plants / anchor pollinators removed) or absent."""
        dev = self.dev; N = self.N; off_q = self.off[1]
        adjs = []
        rels = dict(self.rel_static)
        pi, qi = self.inter
        if with_edges:
            keep = np.ones(len(pi), bool) if anchors is None else ~np.isin(pi, anchors)
            if anchors_q is not None:
                keep &= ~np.isin(qi, anchors_q)
            rels["interacts"] = (off_q + qi[keep], pi[keep], np.ones(int(keep.sum())))   # target pollinator <- plant
        else:
            rels["interacts"] = (np.array([], int), np.array([], int), np.array([]))
        if self.cfg.genus_edges:
            empty = (np.array([], int), np.array([], int), np.array([]))
            if with_edges:
                n_p, n_q = self.off[1], self.off[2] - self.off[1]; off_t = self.off[2]
                pk, qk = pi[keep], qi[keep]
                key = self.gen_p_idx[pk] * n_q + qk; u, cnt = np.unique(key, return_counts=True)
                rels["genus_p_partner"] = (off_q + u % n_q, off_t + u // n_q, np.log1p(cnt))      # pollinator <- plant genus
                key = self.gen_q_idx[qk] * n_p + pk; u, cnt = np.unique(key, return_counts=True)
                rels["genus_q_partner"] = (u % n_p, off_t + u // n_p, np.log1p(cnt))              # plant <- pollinator genus
            else:
                rels["genus_p_partner"] = rels["genus_q_partner"] = empty
        degs = []
        for name in self.rel_names:
            r, c, w = rels[name]
            adjs.append(norm_adj(r, c, w, N, dev))
            adjs.append(norm_adj(c, r, w, N, dev))                            # reverse
            if self.cfg.degree_encoding:
                w = np.asarray(w, np.float64)
                degs.append(np.bincount(r, weights=w, minlength=N)); degs.append(np.bincount(c, weights=w, minlength=N))
        if self.cfg.degree_encoding:
            self._deg = torch.from_numpy(np.log1p(np.stack(degs, 1)).astype(np.float32)).to(dev)     # [N, n_rel]
        return adjs

    def _node_inputs(self):
        x = torch.zeros(self.N, self.cfg.d, device=self.dev)
        n_p, n_q = self.text_p.shape[0], self.text_q.shape[0]
        if self.cfg.presence_input != "none" and self.cfg.presence_fusion == "concat":
            x[:n_p] = self.proj_in(torch.cat([self.text_p, self.pres_p], 1)); x[n_p:n_p + n_q] = self.proj_in(torch.cat([self.text_q, self.pres_q], 1))
        else:
            x[:n_p] = self.proj_text(self.text_p); x[n_p:n_p + n_q] = self.proj_text(self.text_q)
        if self.cfg.warm_residual:
            x[:n_p + n_q] = x[:n_p + n_q] + self.res_emb.weight * self._res_mask[:, None]
        if self.cfg.degree_encoding:
            x = x + self.proj_deg(self._deg)
        if self.cfg.presence_input != "none" and self.cfg.presence_fusion == "add":
            x[:n_p] = x[:n_p] + self.proj_pres(self.pres_p); x[n_p:n_p + n_q] = x[n_p:n_p + n_q] + self.proj_pres(self.pres_q)
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
        self.layers = nn.ModuleList([RelLayer(cfg.d, self.n_rel, cfg.bases, cfg.dropout, cfg.aggregation) for _ in range(cfg.layers)]).to(dev)
        self.head = PairHead(cfg.d, 2 * cfg.d, cfg.dropout, head_type=cfg.head_type, bilinear_rank=cfg.bilinear_rank,
                             n_extra=(1 if cfg.pair_stat != "none" else 0) + (1 if cfg.pres_bilinear_rank > 0 else 0)).to(dev)
        mods = [self.proj_text, self.tax_emb, self.proj_cell, self.layers, self.head]
        if cfg.degree_encoding:
            self.proj_deg = nn.Linear(self.n_rel, cfg.d).to(dev); nn.init.normal_(self.proj_deg.weight, std=0.01); mods.append(self.proj_deg)
        if cfg.pair_stat != "none":
            self._load_pair_stat(store)
        if cfg.presence_input != "none":
            if cfg.presence_input == "field":
                A, Bm = np.load(Path(cfg.field_dir) / "plant_field.npy"), np.load(Path(cfg.field_dir) / "poll_field.npy")
            elif cfg.presence_input == "surface":
                A, Bm = np.asarray(store.plant_proj, np.float32), np.asarray(store.poll_proj, np.float32)
            else:
                raise ValueError(cfg.presence_input)
            mu = np.concatenate([A, Bm]).mean(0, keepdims=True); sd = np.concatenate([A, Bm]).std(0, keepdims=True) + 1e-6
            self.pres_p = torch.from_numpy(((A - mu) / sd).astype(np.float32)).to(dev)
            self.pres_q = torch.from_numpy(((Bm - mu) / sd).astype(np.float32)).to(dev)
            if cfg.presence_fusion == "concat":
                self.proj_in = nn.Sequential(nn.Linear(tp.shape[1] + A.shape[1], cfg.d), nn.GELU(), nn.Linear(cfg.d, cfg.d)).to(dev); mods.append(self.proj_in)
            else:
                self.proj_pres = nn.Linear(A.shape[1], cfg.d).to(dev); mods.append(self.proj_pres)
        if cfg.pres_bilinear_rank > 0:
            if cfg.presence_input == "none":
                A, Bm = np.asarray(store.plant_proj, np.float32), np.asarray(store.poll_proj, np.float32)
                mu = np.concatenate([A, Bm]).mean(0, keepdims=True); sd = np.concatenate([A, Bm]).std(0, keepdims=True) + 1e-6
                self.pres_p = torch.from_numpy(((A - mu) / sd).astype(np.float32)).to(dev)
                self.pres_q = torch.from_numpy(((Bm - mu) / sd).astype(np.float32)).to(dev)
            self.bil_p = nn.Linear(self.pres_p.shape[1], cfg.pres_bilinear_rank, bias=False).to(dev)
            self.bil_q = nn.Linear(self.pres_q.shape[1], cfg.pres_bilinear_rank, bias=False).to(dev)
            mods += [self.bil_p, self.bil_q]
        if cfg.cooc_neg_frac > 0:
            Fm = torch.from_numpy(np.asarray(store.F, np.float32)).to(dev); Pm = torch.from_numpy(np.asarray(store.P, np.float32)).to(dev)
            with torch.no_grad():
                self.cooc = ((Fm @ Pm.T) > 0).cpu().numpy()                    # [n_p, n_q] bool: share at least one cell
            del Fm, Pm
        n_sp = len(store.plants) + len(store.polls)
        if cfg.warm_residual:
            self.res_emb = nn.Embedding(n_sp, cfg.d).to(dev); nn.init.zeros_(self.res_emb.weight); mods.append(self.res_emb)
            allowed = np.zeros(n_sp, bool); allowed[np.unique(self.inter[0])] = True; allowed[self.off[1] + np.unique(self.inter[1])] = True
            self.res_allowed = allowed                                        # species with training edges only; cold species stay at zero
            self._res_mask = torch.from_numpy(allowed.astype(np.float32)).to(dev)
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
        train_plants = np.array(sorted(partners)); train_polls = np.unique(qi)
        cnt = np.bincount(qi, minlength=n_q).astype(np.float64)
        logQ_pop = torch.tensor(np.log(np.maximum(cnt / cnt.sum(), 1e-12)), dtype=torch.float32, device=dev)
        logQ_uni = float(np.log(1.0 / len(negpool.pool(n_q)))); n_uni = cfg.n_cand - cfg.in_batch
        pos_weight = torch.tensor(float(cfg.n_cand - 1), device=dev)
        print(f"    rgcn: {self.N:,} nodes ({len(store.plants)} plants, {n_q} polls, {self.n_t} taxa, "
              f"{self.N - self.off[3]} cell-month), {len(pi):,} interaction edges, {self.n_rel} relations", flush=True)

        for ep in range(cfg.epochs):
            for m in mods: m.train()
            anchors = rng.choice(train_plants, int(cfg.anchor_drop * len(train_plants)), replace=False)
            anchors_q = rng.choice(train_polls, int(cfg.anchor_drop * len(train_polls)), replace=False) if cfg.anchor_side == "both" else None
            adjs = self._adjs(anchors=anchors, with_edges=True, anchors_q=anchors_q)
            if cfg.warm_residual:
                m = self.res_allowed & (rng.random(n_sp) >= cfg.res_drop); m[anchors] = False
                if anchors_q is not None: m[off_q + anchors_q] = False
                self._res_mask = torch.from_numpy(m.astype(np.float32)).to(dev)
            perm = rng.permutation(len(pi)); tot = 0.0; nb = 0; t0 = time.time()
            for s in range(0, len(pi), cfg.batch):
                b = perm[s:s + cfg.batch]; bp, bq = pi[b], qi[b]; B = len(bp)
                sub = rng.choice(B, min(cfg.in_batch, B), replace=False)
                n_co = int(cfg.cooc_neg_frac * n_uni)
                if n_co:
                    pool_mask = np.zeros(n_q, bool); pool_mask[negpool.pool(n_q)] = True
                    co = []
                    for pl in rng.choice(bp, n_co):
                        opts = np.flatnonzero(self.cooc[pl] & pool_mask)
                        co.append(rng.choice(opts) if len(opts) else negpool.sample(rng, 1, n_q)[0])
                    cand = np.concatenate([bq[sub], np.array(co, int), negpool.sample(rng, n_uni - n_co, n_q)])
                else:
                    cand = np.concatenate([bq[sub], negpool.sample(rng, n_uni, n_q)])
                Cn = len(cand)
                h = self._encode(adjs)                                            # full graph, [N, d]
                hp = h[torch.from_numpy(bp).long().to(dev)]; hq = h[off_q + torch.from_numpy(cand).long().to(dev)]
                extra = self._extra(torch.from_numpy(bp).long().to(dev), torch.from_numpy(cand).long().to(dev))
                logits = self.head(hp, hq, extra=extra)
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
            if cfg.warm_residual:
                self._res_mask = torch.from_numpy((self.res_allowed if cfg.infer_with_edges else np.zeros(n_sp, bool)).astype(np.float32)).to(dev)
            self.h_all = self._encode(self._adjs(anchors=None, with_edges=cfg.infer_with_edges))
            n_p = len(store.plants)
            self.hq_all = self.h_all[off_q:off_q + n_q]
            self.dq_all = self.deg_q(self.hq_all).squeeze(-1) if cfg.use_degree_heads else None
        return self

    # ---- explicit pair statistic (R4) ---------------------------------------------------------------
    def _load_pair_stat(self, store):
        """Expected co-presence of the pair from the production surfaces, in the four marginalisation forms of
        eval/run_marginalisation_surfaces.py; stored as factor matrices so a B x C block is one matmul."""
        F_ = ROOT / "data/features"; k = self.cfg.pair_stat; dev = self.dev
        if k == "joint":
            A, Bm, scale = np.asarray(store.plant_proj, np.float32), np.asarray(store.poll_proj, np.float32), 1.0
        elif k == "space":
            A, Bm, scale = np.load(F_ / "plant_surf_space.npy"), np.load(F_ / "poll_surf_space.npy"), 1.0 / 52
        elif k == "time":
            A, Bm, scale = np.load(F_ / "plant_surf_time.npy"), np.load(F_ / "poll_surf_time.npy"), 1.0 / 3335
        elif k == "scalar":
            A, Bm, scale = np.load(F_ / "plant_surf_mass.npy")[:, None], np.load(F_ / "poll_surf_mass.npy")[:, None], 1.0 / (3335 * 52)
        else:
            raise ValueError(k)
        self.ps_p = torch.from_numpy(np.ascontiguousarray(A, dtype=np.float32)).to(dev)
        self.ps_q = torch.from_numpy(np.ascontiguousarray(Bm, dtype=np.float32)).to(dev)
        self.ps_scale = scale
        rng = np.random.default_rng(0)
        with torch.no_grad():
            sample = self._pair_stat(torch.from_numpy(rng.choice(A.shape[0], 512, replace=False)).to(dev),
                                     torch.from_numpy(rng.choice(Bm.shape[0], 2048, replace=False)).to(dev), raw=True)
            self.ps_mu, self.ps_sd = float(sample.mean()), float(sample.std().clamp_min(1e-6))

    def _extra(self, pidx, qidx):
        parts = []
        if self.cfg.pair_stat != "none":
            parts.append(self._pair_stat(pidx, qidx))
        if self.cfg.pres_bilinear_rank > 0:
            parts.append((self.bil_p(self.pres_p[pidx]) @ self.bil_q(self.pres_q[qidx]).T).unsqueeze(-1))
        return torch.cat(parts, -1) if parts else None

    def _pair_stat(self, pidx, qidx, raw=False):
        v = torch.log1p(torch.clamp(self.ps_p[pidx] @ self.ps_q[qidx].T * self.ps_scale, min=0))   # [B, C]
        return v if raw else ((v - self.ps_mu) / self.ps_sd).unsqueeze(-1)

    @torch.no_grad()
    def score_plant(self, p):
        hp = self.h_all[p:p + 1]
        extra = self._extra(torch.tensor([p], device=self.dev), torch.arange(self.hq_all.shape[0], device=self.dev))
        z = self.head(hp, self.hq_all, extra=extra).squeeze(0)
        if self.cfg.use_degree_heads:
            z = z + self.deg_p(hp).squeeze() + self.dq_all
        return z.float().cpu().numpy()

    def score_pairs(self, pi, qi):
        out = np.empty(len(pi), np.float64)
        for p in np.unique(pi):
            m = pi == p; out[m] = self.score_plant(int(p))[qi[m]]
        return out
