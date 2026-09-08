"""Joint field v2: one spatio-temporal SINR over both kingdoms with a text-conditioned species head.

Plan §1.3. The v1 joint field (`train_joint_field.py`) covers the 41% of the universe that has
occurrences. Here the species vector is

    u_s = MLP_text(BioCLIP2_s) + r_s

with r_s a free residual for species that have occurrences (zero-initialised, weight-decayed, and
dropped row-wise with probability `--res-drop` during training so the text path has to carry the
species), and r_s = 0 at inference for zero-shot species. This is LE-SINR's text head plus
DropoutNet's residual, trained jointly -- not the post-hoc ridge that failed (R^2 0.047).

Validation has two parts and the second is the gate: 5% of rows held out (pos-NLL, within-kingdom
top-10 as in v1), and **5% of species held out entirely** -- their presence is predicted from text
alone, scored as AUROC of their observed rows against random background (cell, week) rows. The
plan's gate is >= 0.80.

Exports, under <sdm_data>/joint_field/field_v2/:
  model.pt, species.json, metrics.json
  plant_field.npy [11031, d], poll_field.npy [13124, d]   u for every universe taxon
  plant_field_direct.npy, poll_field_direct.npy           True where the species had occurrences
  grid_h.npy [n_cells, 52, d]                              encoder over the grid
  plant_tokens.npz, poll_tokens.npz                        top-k (cell, week, log p) per species, with a
                                                           floor of tokens per month so the season is covered
Scheme: tg_spatiotemporal (plan §1.3).
"""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipelines.config import load_config
from pipelines.ppe.backbone import text_matrix
from pipelines.sdm.climate import gather_cells
from pipelines.sdm.build_deliverable import SINR
from pipelines.sdm.train_joint_field import Featurizer, load_occ

TEXT_DIR = Path("/scratch/cher/antheia-data/text_embeddings")
SDM = Path("/scratch/cher/antheia-data/pollinator_sdm")


class TextHead(nn.Module):
    """u_s = MLP(text_s) + r_s, r_s only for species with occurrences, row-dropped in training."""

    def __init__(self, text, has_res, d, res_drop):
        super().__init__()
        self.register_buffer("text", text)                                   # [n_sp, 768]
        self.register_buffer("has_res", has_res.float()[:, None])            # [n_sp, 1]
        self.mlp = nn.Sequential(nn.Linear(text.shape[1], d), nn.ReLU(inplace=True), nn.Linear(d, d))
        self.res = nn.Parameter(torch.zeros(len(text), d))
        self.res_drop = res_drop

    def weight(self, text_only=False):
        W = self.mlp(self.text)
        if text_only:
            return W
        keep = self.has_res
        if self.training and self.res_drop > 0:
            keep = keep * (torch.rand(len(W), 1, device=W.device) > self.res_drop).float() / (1 - self.res_drop)
        return W + self.res * keep


def text_lookup():
    """name -> BioCLIP-2 text embedding, from every source we have (e98 vocab, universe, occ-aligned)."""
    emb, sp2id = text_matrix(load_config())
    look = {n: emb[i] for n, i in sp2id.items()}
    for f in (TEXT_DIR / "plants_bioclip2.pt", TEXT_DIR / "polls_bioclip2.pt", SDM / "pollinator_species_text_gbifv3.pt"):
        d = torch.load(f, map_location="cpu", weights_only=False)
        E = d["embeddings"].float().numpy()
        for n, v in zip(d["names"], E):
            look.setdefault(str(n), v)
    return look


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--cap", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--res-drop", type=float, default=0.3)
    ap.add_argument("--holdout-rows", type=float, default=0.05)
    ap.add_argument("--holdout-species", type=float, default=0.05)
    ap.add_argument("--topk", type=int, default=256)
    ap.add_argument("--month-floor", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    dev = args.device if torch.cuda.is_available() else "cpu"
    out = Path(args.out) if args.out else cfg["paths"]["sdm_data"] / "joint_field" / ("field_v2_smoke" if args.smoke else "field_v2")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    torch.manual_seed(0)

    # ---- occurrences and text -----------------------------------------------------------------
    lat, lon, week, sid, king, species, n_plant_sp = load_occ(args.cap, rng)
    look = text_lookup()
    has_text = np.array([s in look for s in species])
    print(f"[text] {has_text.sum()}/{len(species)} training species have a BioCLIP-2 text embedding "
          f"({(~has_text[:n_plant_sp]).sum()} plants, {(~has_text[n_plant_sp:]).sum()} pollinators dropped)", flush=True)
    keep_rows = has_text[sid]
    lat, lon, week, sid, king = (a[keep_rows] for a in (lat, lon, week, sid, king))
    # re-index species compactly, plants first
    old_ids = np.flatnonzero(has_text)
    remap = np.full(len(species), -1); remap[old_ids] = np.arange(len(old_ids))
    sid = remap[sid]; species = [species[i] for i in old_ids]; n_plant_sp = int((old_ids < n_plant_sp).sum())
    n_sp = len(species)
    T = torch.tensor(np.stack([look[s] for s in species]), dtype=torch.float32)
    T = T / (T.norm(dim=1, keepdim=True) + 1e-12)

    grid = pd.read_parquet(ROOT / "data/features/grid.parquet").sort_values("col").reset_index(drop=True)
    cc = cfg["paths"]["sdm_data"] / f"clim_cells_{len(grid)}.npy"
    clim_cells = np.load(cc) if cc.exists() else gather_cells(cfg, grid.centroid_lat.to_numpy(), grid.centroid_lon.to_numpy())
    if not cc.exists():
        np.save(cc, clim_cells)
    fz = Featurizer(grid, clim_cells, dev)
    cell, ok = fz.assign(lat, lon)
    lat, lon, week, sid, king, cell = (a[ok] for a in (lat, lon, week, sid, king, cell))
    n = len(sid); n_cells = len(grid)

    # ---- held-out species (entire), then held-out rows -----------------------------------------
    ho_sp = np.zeros(n_sp, bool)
    for k, sl in ((0, np.arange(n_plant_sp)), (1, np.arange(n_plant_sp, n_sp))):
        pick = rng.choice(sl, int(round(args.holdout_species * len(sl))), replace=False)
        ho_sp[pick] = True
    row_ho_sp = ho_sp[sid]
    perm = rng.permutation(np.flatnonzero(~row_ho_sp))
    n_ho = int(len(perm) * args.holdout_rows)
    ho, tr = perm[:n_ho], perm[n_ho:]
    if args.smoke:
        tr = tr[:200_000]
    has_res = torch.tensor(~ho_sp)                       # held-out species get no residual: text only
    print(f"[split] {len(tr):,} train rows, {len(ho):,} held-out rows, {ho_sp.sum()} held-out species "
          f"({int(row_ho_sp.sum()):,} rows) scored zero-shot", flush=True)

    Tt = lambda a, dt=torch.float32: torch.tensor(a, dtype=dt, device=dev)
    lat_t, lon_t, week_t, sid_t, king_t, cell_t = Tt(lat), Tt(lon), Tt(week, torch.long), Tt(sid, torch.long), Tt(king, torch.long), Tt(cell, torch.long)
    tr_t, ho_t = Tt(tr, torch.long), Tt(ho, torch.long)
    by_king = [Tt(tr[king[tr] == k], torch.long) for k in (0, 1)]
    sp_king = torch.zeros(n_sp, dtype=torch.long, device=dev); sp_king[n_plant_sp:] = 1
    king_mask = torch.stack([sp_king == 0, sp_king == 1]).float()
    n_king_sp = king_mask.sum(1)

    enc = SINR(fz.in_dim, 1, h=args.d).to(dev)          # encoder only; the head is TextHead
    head = TextHead(T, has_res, args.d, args.res_drop).to(dev)
    params = list(enc.enc.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params, args.lr, weight_decay=1e-5)

    def background(kidx):
        B = len(kidx); k = king_t[kidx]
        src = torch.where(k == 0, by_king[0][torch.randint(0, len(by_king[0]), (B,), device=dev)],
                          by_king[1][torch.randint(0, len(by_king[1]), (B,), device=dev)])
        c, w = cell_t[src], week_t[src]
        jit = (torch.rand(B, 2, device=dev) - 0.5) * 0.5
        return fz(fz.glat[c] + jit[:, 0], fz.glon[c] + jit[:, 1], c, w), k

    def logits(x, text_only=False):
        return enc.emb(x) @ head.weight(text_only).T

    def evaluate():
        enc.eval(); head.eval()
        nll = top1 = top10 = cnt = 0.0
        with torch.no_grad():
            for i in range(0, len(ho_t), 8192):
                b = ho_t[i:i + 8192]
                z = logits(fz(lat_t[b], lon_t[b], cell_t[b], week_t[b])).masked_fill(king_mask[king_t[b]] == 0, -1e4)
                s = sid_t[b]; ar = torch.arange(len(b), device=dev)
                nll += F.softplus(-z[ar, s]).sum().item()
                rank = (z > z[ar, s][:, None]).sum(1)
                top1 += (rank == 0).sum().item(); top10 += (rank < 10).sum().item(); cnt += len(b)
            # zero-shot: held-out species' observed rows vs equal random background, text-only vectors
            Wt = head.weight(text_only=True)
            zs = {}
            for k in (0, 1):
                rows = np.flatnonzero(row_ho_sp & (king == k))
                rows = rng.choice(rows, min(len(rows), 50_000), replace=False)
                r = Tt(rows, torch.long)
                z_pos = (enc.emb(fz(lat_t[r], lon_t[r], cell_t[r], week_t[r])) * Wt[sid_t[r]]).sum(1)
                c = torch.randint(0, n_cells, (len(r),), device=dev); w = torch.randint(0, 52, (len(r),), device=dev)
                jit = (torch.rand(len(r), 2, device=dev) - 0.5) * 0.5
                z_neg = (enc.emb(fz(fz.glat[c] + jit[:, 0], fz.glon[c] + jit[:, 1], c, w)) * Wt[sid_t[r]]).sum(1)
                y = np.r_[np.ones(len(r)), np.zeros(len(r))]
                zs[k] = roc_auc_score(y, np.r_[z_pos.cpu().numpy(), z_neg.cpu().numpy()])
        enc.train(); head.train()
        return dict(pos_nll=nll / cnt, top1=top1 / cnt, top10=top10 / cnt,
                    zeroshot_auroc_plants=zs[0], zeroshot_auroc_polls=zs[1])

    print(f"[train] {args.epochs} epochs, tg_spatiotemporal, d={args.d}, res_drop={args.res_drop}", flush=True)
    for ep in range(args.epochs):
        t0 = time.time(); tot = 0.0; steps = 0
        order = tr_t[torch.randperm(len(tr_t), device=dev)]
        for i in range(0, len(order), args.batch):
            b = order[i:i + args.batch]
            x_pos = fz(lat_t[b], lon_t[b], cell_t[b], week_t[b]); s, k = sid_t[b], king_t[b]
            x_bg, kb = background(b)
            W = head.weight()
            z = enc.emb(torch.cat([x_pos, x_bg])) @ W.T
            zp, zb = z[:len(b)], z[len(b):]
            ar = torch.arange(len(b), device=dev)
            pos = F.softplus(-zp[ar, s]); neg = (F.softplus(zb) * king_mask[kb]).sum(1)
            loss = ((pos * n_king_sp[k]).mean() + neg.mean()) / n_king_sp.mean()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 5.0); opt.step()
            tot += loss.item(); steps += 1
        if ep % 5 == 0 or ep == args.epochs - 1:
            m = evaluate()
            print(f"    ep{ep:02d} loss {tot/steps:.4f}  held-out pos-NLL {m['pos_nll']:.3f} top10 {m['top10']:.3f}  "
                  f"zero-shot AUROC plants {m['zeroshot_auroc_plants']:.3f} polls {m['zeroshot_auroc_polls']:.3f}  ({time.time()-t0:.0f}s)", flush=True)
        else:
            print(f"    ep{ep:02d} loss {tot/steps:.4f}  ({time.time()-t0:.0f}s)", flush=True)

    # ---- export -------------------------------------------------------------------------------
    enc.eval(); head.eval()
    m = evaluate()
    gate = min(m["zeroshot_auroc_plants"], m["zeroshot_auroc_polls"])
    m["gate_0.80"] = bool(gate >= 0.80)
    print(f"[gate] zero-shot AUROC min over kingdoms {gate:.3f} -> {'PASS' if m['gate_0.80'] else 'FAIL'}", flush=True)

    with torch.no_grad():
        gc = torch.arange(n_cells, device=dev).repeat_interleave(52); gw = torch.arange(52, device=dev).repeat(n_cells)
        Xg = fz(fz.glat[gc], fz.glon[gc], gc, gw)
        Hg = torch.cat([enc.emb(Xg[i:i + 16384]) for i in range(0, len(Xg), 16384)])           # [C*52, d]
        np.save(out / "grid_h.npy", Hg.view(n_cells, 52, -1).cpu().numpy().astype(np.float32))
        W_obs = head.weight().cpu().numpy()                                                    # text + residual (eval mode: no drop)

        u = json.load(open(ROOT / "data/network/modelled_universe.json"))
        s2i = {s: i for i, s in enumerate(species)}
        mp = head.mlp
        for side, key in (("plant", "plants"), ("poll", "pollinators")):
            names = [x["label"] for x in u[key]]
            Fld = np.zeros((len(names), args.d), np.float32); D = np.zeros(len(names), bool); missing = 0
            zs_names, zs_rows = [], []
            for j, nm in enumerate(names):
                i = s2i.get(nm)
                if i is not None and ((side == "plant") == (i < n_plant_sp)) and not ho_sp[i]:
                    Fld[j] = W_obs[i]; D[j] = True
                elif nm in look:
                    zs_names.append(j); zs_rows.append(look[nm])
                else:
                    missing += 1
            if zs_rows:
                Z = torch.tensor(np.stack(zs_rows), dtype=torch.float32, device=dev)
                Z = Z / (Z.norm(dim=1, keepdim=True) + 1e-12)
                Fld[np.array(zs_names)] = mp(Z).cpu().numpy()
            np.save(out / f"{side}_field.npy", Fld); np.save(out / f"{side}_field_direct.npy", D)
            m[f"{side}_universe_observed"] = float(D.mean()); m[f"{side}_universe_missing_text"] = missing
            # sparse tokens: top-k (cell, week, log p) with a per-month floor
            U = torch.tensor(Fld, device=dev)
            tok_c = np.zeros((len(names), args.topk), np.int16); tok_w = np.zeros((len(names), args.topk), np.int8)
            tok_lp = np.zeros((len(names), args.topk), np.float16)
            month_of = torch.clamp((torch.arange(52, device=dev) * 7 + 3) // 30, 0, 11).repeat(n_cells)   # aligned with gc, gw
            for s0 in range(0, len(names), 256):
                P = torch.sigmoid(U[s0:s0 + 256] @ Hg.T)                                           # [b, C*52]
                lp = torch.log(P + 1e-9)
                # floor: best `month_floor` positions in each month, then fill the rest by global top-k
                sel = torch.zeros_like(P, dtype=torch.bool)
                for mth in range(12):
                    idx = torch.nonzero(month_of == mth).squeeze(1)
                    top = P[:, idx].topk(args.month_floor, dim=1).indices
                    sel.scatter_(1, idx[top], True)
                P2 = P.masked_fill(sel, 2.0)                                                       # force floored ones in
                top = P2.topk(args.topk, dim=1).indices                                            # [b, k]
                tok_c[s0:s0 + len(top)] = (top // 52).cpu().numpy(); tok_w[s0:s0 + len(top)] = (top % 52).cpu().numpy()
                tok_lp[s0:s0 + len(top)] = torch.gather(lp, 1, top).cpu().numpy()
            np.savez(out / f"{side}_tokens.npz", cell=tok_c, week=tok_w, logp=tok_lp)
            print(f"  [{side}] universe {len(names)}: observed {D.sum()}, zero-shot {len(zs_names)}, no text {missing}", flush=True)

    torch.save({"enc": enc.state_dict(), "head": head.state_dict(), "in_dim": fz.in_dim, "d": args.d,
                "clim_mu": fz.mu.numpy(), "clim_sd": fz.sd.numpy()}, out / "model.pt")
    json.dump({"species": species, "n_plant_species": n_plant_sp, "heldout_species": np.flatnonzero(ho_sp).tolist(),
               "n_obs": np.bincount(sid[tr], minlength=n_sp).tolist()}, open(out / "species.json", "w"))
    json.dump(m, open(out / "metrics.json", "w"), indent=1)
    print(json.dumps(m, indent=1), flush=True)
    print(f"[wrote] {out}", flush=True)


if __name__ == "__main__":
    main()
