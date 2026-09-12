"""One spatio-temporal field model for both kingdoms, with the negative-sampling scheme as the only knob.

A SINR: shared encoder over (Fourier lat/lon, Climplicit monthly climate of the cell, Fourier week)
and a per-species vector for every plant and pollinator. The species vector is the species'
learned spatio-temporal influence, and because one encoder serves both kingdoms the two sides are
commensurable -- the thing the separate PPE and SDM surfaces never were.

Every scheme sees the same positives: observed (species, location, week) rows, flowering-only
for plants. They differ in where the all-negative background rows come from, i.e. in how much of
the observer-effort structure the model is told to explain away:

  uniform            cell ~ U(grid), week ~ U(52)                    effort is species signal
  tg_spatial         cell of a random observed point, week ~ U(52)    spatial effort cancels
  tg_spatiotemporal  (cell, week) of a random observed point          spatial and seasonal effort cancel
  slds               no background; one random other species is the negative at each observed row

Background rows are drawn within kingdom: plants and insects are recorded by different observer
populations, so each kingdom is its own target group. Occurrences are attached to their nearest
0.5-degree cell for climate; coordinates stay exact for observed rows and are jittered within the
cell for background rows so neither side is identifiable by lying on a centroid.

Outputs per scheme, under <sdm_data>/joint_field/joint_field_<scheme>/:
  model.pt, species.json, metrics.json
  plant_field.npy [n_universe_plants, d], plant_field_direct.npy   zero rows where untrained
  poll_field.npy  [n_universe_polls,  d], poll_field_direct.npy
  grid_h.npy [n_cells, 52, d]   encoder output over the whole grid, for exact encounter integrals
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
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipelines.common import fourier_loc, fourier_week
from pipelines.config import load_config
from pipelines.sdm.climate import gather_cells
from pipelines.sdm.build_deliverable import SINR
from antheia.paths import DATA_ROOT, HF_CACHE

SCHEMES = ("uniform", "tg_spatial", "tg_spatiotemporal", "slds")
PLANT_OCC = Path(str(DATA_ROOT) + "/plant_occ/inat_train_occ.npz")


def week_month(week):
    return np.clip((np.asarray(week) * 7 + 4 - 1) // 30, 0, 11)


def load_occ(cap, rng):
    """Observed rows for both kingdoms: lat, lon, week, species index (plants first), kingdom flag."""
    zp = np.load(PLANT_OCC, allow_pickle=True)
    fl = zp["flowering"]
    p_lat, p_lon, p_week, p_sid = zp["lat"][fl], zp["lon"][fl], ((zp["doy"][fl] - 1) // 7).clip(0, 51), zp["species_id"][fl]
    p_names = zp["names"]
    zq = np.load(load_config()["paths"]["sdm_data"] / "pollinator_occ_gbifv3.npz", allow_pickle=True)
    q_lat, q_lon, q_week, q_sid = zq["lat"], zq["lon"], ((zq["doy"].astype(int) - 1) // 7).clip(0, 51), zq["sidx"].astype(int)
    q_names = zq["names"]

    def cap_rows(sid):
        keep = []
        for s in np.unique(sid):
            idx = np.flatnonzero(sid == s)
            keep.append(idx if len(idx) <= cap else rng.choice(idx, cap, replace=False))
        return np.concatenate(keep)

    kp, kq = cap_rows(p_sid), cap_rows(q_sid)
    # compact species indexing: plants that occur, then pollinators that occur
    p_used = np.unique(p_sid[kp]); q_used = np.unique(q_sid[kq])
    p_map = np.full(len(p_names), -1); p_map[p_used] = np.arange(len(p_used))
    q_map = np.full(len(q_names), -1); q_map[q_used] = np.arange(len(q_used)) + len(p_used)
    lat = np.concatenate([p_lat[kp], q_lat[kq]]).astype(np.float32)
    lon = np.concatenate([p_lon[kp], q_lon[kq]]).astype(np.float32)
    week = np.concatenate([p_week[kp], q_week[kq]]).astype(np.int64)
    sid = np.concatenate([p_map[p_sid[kp]], q_map[q_sid[kq]]]).astype(np.int64)
    king = np.concatenate([np.zeros(len(kp), np.int64), np.ones(len(kq), np.int64)])
    species = [str(p_names[i]) for i in p_used] + [str(q_names[i]) for i in q_used]
    return lat, lon, week, sid, king, species, len(p_used)


class Featurizer:
    """Rows -> [B, 276] on device: Fourier(lat, lon) + standardised cell climate at the row's month + Fourier(week)."""

    def __init__(self, grid, clim_cells, dev):
        self.dev = dev
        self.tree = cKDTree(np.c_[grid.centroid_lat.to_numpy(), grid.centroid_lon.to_numpy()])
        self.glat = torch.tensor(grid.centroid_lat.to_numpy(), dtype=torch.float32, device=dev)
        self.glon = torch.tensor(grid.centroid_lon.to_numpy(), dtype=torch.float32, device=dev)
        c = torch.tensor(clim_cells, dtype=torch.float32)                    # [n_cells, 12, 256]
        self.mu, self.sd = c.mean((0, 1)), c.std((0, 1)) + 1e-6
        self.clim = ((c - self.mu) / self.sd).to(dev)
        self.in_dim = 16 + 256 + 4

    def assign(self, lat, lon, max_dist=0.5):
        d, nn = self.tree.query(np.c_[lat, lon])
        return nn, d <= max_dist

    def __call__(self, lat, lon, cell, week):
        """All arguments are device tensors: lat/lon float [B], cell/week long [B]."""
        month = torch.clamp((week * 7 + 3) // 30, 0, 11)
        x = lat / 90.0; y = lon / 180.0
        loc = []
        for f in (1, 2, 4, 8):
            loc += [torch.sin(np.pi * f * x), torch.cos(np.pi * f * x), torch.sin(np.pi * f * y), torch.cos(np.pi * f * y)]
        w = 2 * np.pi * week.float() / 52.0
        wk = [torch.sin(w), torch.cos(w), torch.sin(2 * w), torch.cos(2 * w)]
        return torch.cat([torch.stack(loc, 1), self.clim[cell, month], torch.stack(wk, 1)], 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--schemes", nargs="*", default=list(SCHEMES))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--cap", type=int, default=1000, help="observed rows kept per species")
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--holdout", type=float, default=0.05)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    dev = args.device if torch.cuda.is_available() else "cpu"
    out_root = Path(args.out) if args.out else cfg["paths"]["sdm_data"] / "joint_field"
    if args.smoke:
        out_root = out_root / "smoke"
    rng = np.random.default_rng(0)

    lat, lon, week, sid, king, species, n_plant_sp = load_occ(args.cap, rng)
    n_sp = len(species)
    print(f"[occ] {len(sid):,} rows: {int((king == 0).sum()):,} plant (flowering) / {int((king == 1).sum()):,} "
          f"pollinator; {n_plant_sp} plant + {n_sp - n_plant_sp} pollinator species", flush=True)

    grid = pd.read_parquet(ROOT / "data/features/grid.parquet").sort_values("col").reset_index(drop=True)
    cc = cfg["paths"]["sdm_data"] / f"clim_cells_{len(grid)}.npy"
    if cc.exists():
        clim_cells = np.load(cc)
    else:
        clim_cells = gather_cells(cfg, grid.centroid_lat.to_numpy(), grid.centroid_lon.to_numpy())
        np.save(cc, clim_cells)
    fz = Featurizer(grid, clim_cells, dev)
    cell, ok = fz.assign(lat, lon)
    print(f"[grid] {len(grid)} cells; {int((~ok).sum()):,} rows farther than 0.5 deg from any cell dropped", flush=True)
    lat, lon, week, sid, king, cell = (a[ok] for a in (lat, lon, week, sid, king, cell))
    n = len(sid)
    perm = rng.permutation(n)
    n_ho = int(n * args.holdout)
    ho, tr = perm[:n_ho], perm[n_ho:]
    if args.smoke:
        tr = tr[:200_000]
    T = lambda a, dt=torch.float32: torch.tensor(a, dtype=dt, device=dev)
    lat_t, lon_t, week_t, sid_t, king_t, cell_t = T(lat), T(lon), T(week, torch.long), T(sid, torch.long), T(king, torch.long), T(cell, torch.long)
    tr_t, ho_t = T(tr, torch.long), T(ho, torch.long)
    by_king = [T(tr[king[tr] == k], torch.long) for k in (0, 1)]              # observed rows per kingdom (train)
    sp_king = torch.zeros(n_sp, dtype=torch.long, device=dev); sp_king[n_plant_sp:] = 1
    king_mask = torch.stack([sp_king == 0, sp_king == 1]).float()            # [2, n_sp]
    n_king_sp = king_mask.sum(1)                                              # positives weighted to balance a kingdom's negatives
    n_cells = len(grid)

    # grid features for export and for the exact encounter integral: [n_cells*52, in_dim]
    gc = torch.arange(n_cells, device=dev).repeat_interleave(52)
    gw = torch.arange(52, device=dev).repeat(n_cells)
    Xg = fz(fz.glat[gc], fz.glon[gc], gc, gw)

    def background(kidx, scheme):
        """All-negative rows for the observed rows `kidx` (same kingdom each), per scheme."""
        B = len(kidx)
        k = king_t[kidx]
        if scheme == "uniform":
            c = torch.randint(0, n_cells, (B,), device=dev)
            w = torch.randint(0, 52, (B,), device=dev)
        else:
            # a random observed row of the same kingdom
            src = torch.where(k == 0, by_king[0][torch.randint(0, len(by_king[0]), (B,), device=dev)],
                              by_king[1][torch.randint(0, len(by_king[1]), (B,), device=dev)])
            c = cell_t[src]
            w = torch.randint(0, 52, (B,), device=dev) if scheme == "tg_spatial" else week_t[src]
        jit = (torch.rand(B, 2, device=dev) - 0.5) * 0.5
        return fz(fz.glat[c] + jit[:, 0], fz.glon[c] + jit[:, 1], c, w), k

    def evaluate(net):
        net.eval()
        nll, top1, top10, cnt = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for i in range(0, len(ho_t), 8192):
                b = ho_t[i:i + 8192]
                z = net.cls(net.emb(fz(lat_t[b], lon_t[b], cell_t[b], week_t[b])))
                z = z.masked_fill(king_mask[king_t[b]] == 0, -1e4)          # rank within kingdom
                s = sid_t[b]
                nll += torch.nn.functional.softplus(-z[torch.arange(len(b), device=dev), s]).sum().item()
                rank = (z > z[torch.arange(len(b), device=dev), s][:, None]).sum(1)
                top1 += (rank == 0).sum().item(); top10 += (rank < 10).sum().item(); cnt += len(b)
        net.train()
        return dict(pos_nll=nll / cnt, top1=top1 / cnt, top10=top10 / cnt)

    n_obs_per_sp = np.bincount(sid[tr], minlength=n_sp)
    for scheme in args.schemes:
        out = out_root / f"joint_field_{scheme}"; out.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(0)
        net = SINR(fz.in_dim, n_sp, h=args.d).to(dev)
        opt = torch.optim.Adam(net.parameters(), args.lr, weight_decay=1e-5)
        print(f"\n[{scheme}] {args.epochs} epochs on {len(tr_t):,} rows", flush=True)
        for ep in range(args.epochs):
            t0 = time.time(); tot = 0.0; steps = 0
            order = tr_t[torch.randperm(len(tr_t), device=dev)]
            for i in range(0, len(order), args.batch):
                b = order[i:i + args.batch]
                x_pos = fz(lat_t[b], lon_t[b], cell_t[b], week_t[b])
                s, k = sid_t[b], king_t[b]
                if scheme == "slds":
                    z = net.cls(net.emb(x_pos))
                    # one random other species of the same kingdom as the negative
                    r = torch.randint(0, n_sp, (len(b),), device=dev)
                    lo = torch.where(k == 0, torch.zeros_like(r), torch.full_like(r, n_plant_sp))
                    hi = torch.where(k == 0, torch.full_like(r, n_plant_sp), torch.full_like(r, n_sp))
                    r = lo + (r % (hi - lo)); r = torch.where(r == s, lo + ((r + 1 - lo) % (hi - lo)), r)
                    ar = torch.arange(len(b), device=dev)
                    loss = torch.nn.functional.softplus(-z[ar, s]).mean() + torch.nn.functional.softplus(z[ar, r]).mean()
                else:
                    x_bg, kb = background(b, scheme)
                    z = net.cls(net.emb(torch.cat([x_pos, x_bg])))
                    zp, zb = z[:len(b)], z[len(b):]
                    ar = torch.arange(len(b), device=dev)
                    pos = torch.nn.functional.softplus(-zp[ar, s])                        # -log p_own
                    neg = (torch.nn.functional.softplus(zb) * king_mask[kb]).sum(1)      # -sum log(1-p) over the kingdom
                    loss = (pos * n_king_sp[k]).mean() + neg.mean()
                    loss = loss / n_king_sp.mean()
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                opt.step(); tot += loss.item(); steps += 1
            if ep % 5 == 0 or ep == args.epochs - 1:
                m = evaluate(net)
                print(f"    ep{ep:02d} loss {tot / steps:.4f}  held-out pos-NLL {m['pos_nll']:.3f}  "
                      f"top1 {m['top1']:.3f} top10 {m['top10']:.3f}  ({time.time() - t0:.0f}s)", flush=True)
            else:
                print(f"    ep{ep:02d} loss {tot / steps:.4f}  ({time.time() - t0:.0f}s)", flush=True)

        net.eval()
        U = net.cls.weight.detach().cpu().numpy().astype(np.float32)          # [n_sp, d]
        m = evaluate(net)
        # how much observer effort leaked into the embedding: |u| against log record count
        norm = np.linalg.norm(U, axis=1)
        m["corr_norm_logcount"] = float(np.corrcoef(norm, np.log1p(n_obs_per_sp))[0, 1])
        m["scheme"] = scheme
        with torch.no_grad():
            Hg = torch.cat([net.emb(Xg[i:i + 16384]) for i in range(0, len(Xg), 16384)]).cpu().numpy()
        np.save(out / "grid_h.npy", Hg.reshape(n_cells, 52, -1).astype(np.float32))
        np.save(out / "head_bias.npy", np.float32(0.0))
        torch.save({"state_dict": net.state_dict(), "in_dim": fz.in_dim, "d": args.d,
                    "clim_mu": fz.mu.numpy(), "clim_sd": fz.sd.numpy()}, out / "model.pt")
        json.dump({"species": species, "n_plant_species": n_plant_sp, "n_obs": n_obs_per_sp.tolist()},
                  open(out / "species.json", "w"))

        # universe alignment
        u = json.load(open(ROOT / "data/network/modelled_universe.json"))
        s2i = {s: i for i, s in enumerate(species)}
        for side, key in (("plant", "plants"), ("poll", "pollinators")):
            names = [x["label"] for x in u[key]]
            F = np.zeros((len(names), U.shape[1]), np.float32); D = np.zeros(len(names), bool)
            for j, nm in enumerate(names):
                i = s2i.get(nm)
                if i is not None and ((side == "plant") == (i < n_plant_sp)):
                    F[j] = U[i]; D[j] = True
            np.save(out / f"{side}_field.npy", F); np.save(out / f"{side}_field_direct.npy", D)
            m[f"{side}_universe_coverage"] = float(D.mean())
        json.dump(m, open(out / "metrics.json", "w"), indent=1)
        print(f"  [{scheme}] held-out pos-NLL {m['pos_nll']:.3f} top10 {m['top10']:.3f}  "
              f"corr(|u|, log n_obs) {m['corr_norm_logcount']:+.3f}  universe coverage plants "
              f"{m['plant_universe_coverage']:.1%} polls {m['poll_universe_coverage']:.1%} -> {out}", flush=True)
        del net, opt; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
