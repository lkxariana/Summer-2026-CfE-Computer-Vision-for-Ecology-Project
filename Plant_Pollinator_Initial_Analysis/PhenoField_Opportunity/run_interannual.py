"""Interannual (temporal-transfer) hold-out: train on years <=2021, test 2022-24.

Tests whether PPE's phenology channel captures YEAR-SPECIFIC flowering shifts a
climatology cannot. Universe = per-(cell,week,year) real climate windows for the
20 used cells (prism_weekly_peryear.npz) x 50 species; label = presence if a
flowering event of that species falls in that (cell,week,year). PPE-L z_dynamic
is encoded on each year's REAL window (so it can, in principle, shift with the
year's climate).

Covariate arms (added to coords + species embedding):
  +week        Fourier(week)          year-INVARIANT clock
  +climplicit  climplicit(cell,month) year-INVARIANT climate climatology (SOTA baseline)
  +gdd         GDD-to-date (per-year) year-AWARE raw climate
  +ppe         PPE-L z_dynamic (per-year window) year-AWARE phenology channel
  +week+ppe

Held-out-year AP. Decisive contrast: do year-AWARE covariates (+ppe,+gdd) beat
year-INVARIANT ones (+week,+climplicit) on 2022-24? If +ppe > clock/climplicit,
PPE resolves interannual shift; if it ties, phenology is ~stationary / PPE adds
no interannual signal.
"""
from __future__ import annotations
import os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score

from common import coord_enc, fourier_week
from extract_ppel_field import load_ppel, climplicit_by_cell, VOCAB, AE_PARQUET

HERE = Path(__file__).resolve().parent
OUTDIR = HERE / 'outputs'; FIGDIR = OUTDIR / 'figures_interannual'
PROJ = Path('/projects/bdbl/cherd/Summer-2026-CfE-Computer-Vision-for-Ecology-Project')
PART2 = PROJ / 'Plant_Pollinator_Initial_Analysis/Part2_PhenoField_Outputs/data'
PERYEAR = PART2 / 'prism_weekly_peryear.npz'
TRAIN_MAX_YEAR = 2021; TEST_YEARS = (2022, 2023, 2024)
NEG_RATIO = 4; EPOCHS = 40; GDD_BASE = 5.0; TMEAN_IDX = 3


class Net(nn.Module):
    def __init__(self, uw, uc, ug, uz, n_species, sp_emb=16, hid=256):
        super().__init__(); self.uw, self.uc, self.ug, self.uz = uw, uc, ug, uz
        self.emb = nn.Embedding(n_species, sp_emb)
        d = sp_emb + 6 + (4 if uw else 0) + (256 if uc else 0) + (1 if ug else 0) + (192 if uz else 0)
        self.net = nn.Sequential(nn.Linear(d, hid), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(hid, hid), nn.GELU(), nn.Dropout(0.1), nn.Linear(hid, 1))
    def forward(self, co, wk, cl, gd, zd, sp):
        p = [self.emb(sp), co]
        if self.uw: p.append(wk)
        if self.uc: p.append(cl)
        if self.ug: p.append(gd)
        if self.uz: p.append(zd)
        return self.net(torch.cat(p, -1)).squeeze(-1)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    FIGDIR.mkdir(parents=True, exist_ok=True); print(f"[device] {device}", flush=True)
    import json
    vocab = json.load(open(VOCAB))
    plants = sorted(pd.read_parquet(PROJ / 'data/plant_flowering_events.parquet').species.unique())
    sp_ids = np.array([vocab[s] for s in plants], np.int64); sp2idx = {s: i for i, s in enumerate(plants)}

    pz = np.load(PERYEAR); pc = pz['cell_idx'].astype(int); pw = pz['week'].astype(int); pyr = pz['year'].astype(int); win = pz['window']
    K = len(pc); S = len(plants)
    print(f"[universe] {K:,} (cell,week,year) windows x {S} species = {K*S:,} bins; years {pyr.min()}-{pyr.max()}", flush=True)

    grid = pd.read_csv(PROJ / 'data/grid_centroids_0.5deg.csv')
    lat_c = grid.centroid_lat.values; lon_c = grid.centroid_lon.values
    ae_df = pd.read_parquet(AE_PARQUET); from scipy.spatial import cKDTree
    _, nn_ae = cKDTree(ae_df[['lat', 'lon']].values).query(grid[['centroid_lat', 'centroid_lon']].values)
    ae_by_cell = ae_df[[f'emb_{i:02d}' for i in range(64)]].values[nn_ae].astype(np.float32)
    clim_by_cell = climplicit_by_cell(grid)   # [ncells,12,256]

    # presence key set (cell,week,year,species)
    ev = pd.read_parquet(PROJ / 'data/plant_flowering_events.parquet')
    klat = np.rint(grid.lat_bin.values*2).astype(int); klon = np.rint(grid.lon_bin.values*2).astype(int)
    k2c = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(klat, klon))}
    ec = np.array([k2c.get((int(a), int(b)), -1) for a, b in zip(np.floor(ev.lat*2).astype(int), np.floor(ev.lon*2).astype(int))])
    ew = np.clip((ev.doy.values-1)//7, 0, 51); ey = ev.year.values; esi = ev.species.map(sp2idx).values
    ok = (ec >= 0) & pd.notna(esi)
    def key4(c, w, y, s): return ((c.astype(np.int64)*52+w)*3000+y)*50+s
    pres = set(key4(ec[ok], ew[ok], ey[ok], esi[ok].astype(np.int64)).tolist())

    # build the bin table (cell,week,year,species) over the universe
    win_row = np.repeat(np.arange(K), S); sp_row = np.tile(np.arange(S), K)
    cell = pc[win_row]; week = pw[win_row]; year = pyr[win_row]
    gk = key4(cell.astype(np.int64), week, year, sp_row.astype(np.int64))
    is_pres = np.fromiter((k in pres for k in gk), bool, len(gk))
    print(f"[labels] presence bins={int(is_pres.sum()):,} ({100*is_pres.mean():.2f}%) over {len(gk):,} bins", flush=True)

    # ---- encode PPE-L z_dynamic for ALL bins (per-year window) ----
    doy = (week*7+4).clip(1, 365); month = np.clip(((doy-1)//30)+1, 1, 12)
    gdd = np.array([float(np.nansum(np.clip(win[win_row[i]][365-int(doy[i]):, TMEAN_IDX].astype(np.float32)-GDD_BASE, 0, None))) for i in range(len(win_row))], np.float32)
    model = load_ppel(device)
    N = len(win_row); zdyn = np.zeros((N, 192), np.float32); t0 = time.time(); bs = 2048
    for i in range(0, N, bs):
        sl = slice(i, i+bs); m = win_row[sl]
        prism = torch.nan_to_num(torch.from_numpy(win[m].astype(np.float32)), nan=0.0).to(device)
        clim = torch.from_numpy(np.stack([clim_by_cell[cell[j], month[j]-1] for j in range(i, min(i+bs, N))]).astype(np.float32)).to(device)
        with torch.no_grad():
            out = model.encode_field_with_head([plants[k] for k in sp_row[sl]],
                torch.tensor(lat_c[cell[sl]], dtype=torch.float32, device=device),
                torch.tensor(lon_c[cell[sl]], dtype=torch.float32, device=device),
                torch.tensor(doy[sl], dtype=torch.long, device=device),
                species_id=torch.tensor(sp_ids[sp_row[sl]], dtype=torch.long, device=device),
                prism_window=prism, alphaearth_emb=torch.from_numpy(ae_by_cell[cell[sl]]).to(device),
                climplicit_emb=clim, return_features=True)
        zdyn[sl] = out['z_dynamic_field'].float().cpu().numpy()
        if (i//bs) % 30 == 0: print(f"  encode {i+len(m):,}/{N:,} ({time.time()-t0:.0f}s)", flush=True)

    # covariates
    co = coord_enc(lat_c[cell], lon_c[cell]); wk = fourier_week(week)
    cl = np.stack([clim_by_cell[cell[i], month[i]-1] for i in range(N)]).astype(np.float32)
    train_mask = year <= TRAIN_MAX_YEAR; test_mask = np.isin(year, TEST_YEARS)
    rng = np.random.RandomState(0)
    def pool(mask):
        p = np.where(is_pres & mask)[0]; ncand = np.where((~is_pres) & mask)[0]
        n = rng.choice(ncand, size=min(NEG_RATIO*len(p), len(ncand)), replace=False)
        idx = np.concatenate([p, n]); return idx, is_pres[idx].astype(np.float32)
    tr, ytr = pool(train_mask); te, yte = pool(test_mask)
    print(f"[split] train bins={len(tr):,} (pos={int(ytr.sum()):,}) | test bins={len(te):,} (pos={int(yte.sum()):,})", flush=True)

    # standardize gdd, clim, zdyn on train
    def st(a, idx, mean, sd): return ((a[idx]-mean)/sd).astype(np.float32)
    gm, gs = gdd[tr].mean(), gdd[tr].std()+1e-6
    cm, cs = cl[tr].mean(0), cl[tr].std(0)+1e-6
    zm, zs = zdyn[tr].mean(0), zdyn[tr].std(0)+1e-6
    T = lambda a: torch.from_numpy(a).to(device)
    def blocks(idx):
        return (T(co[idx]), T(wk[idx]), T((cl[idx]-cm)/cs), T(((gdd[idx]-gm)/gs)[:, None]), T((zdyn[idx]-zm)/zs), T(sp_row[idx]).long())
    tr_b = blocks(tr); te_b = blocks(te); ytr_t = T(ytr)

    ARMS = [('coords', 0,0,0,0), ('+week', 1,0,0,0), ('+climplicit', 0,1,0,0),
            ('+gdd', 0,0,1,0), ('+ppe', 0,0,0,1), ('+week+ppe', 1,0,0,1)]
    rows = []
    for name, uw, uc, ug, uz in ARMS:
        torch.manual_seed(0)
        net = Net(bool(uw), bool(uc), bool(ug), bool(uz), S).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5); lf = nn.BCEWithLogitsLoss()
        n = len(ytr_t); bsz = 8192
        for ep in range(EPOCHS):
            perm = torch.randperm(n, device=device)
            for i in range(0, n, bsz):
                b = perm[i:i+bsz]; opt.zero_grad()
                lf(net(tr_b[0][b], tr_b[1][b], tr_b[2][b], tr_b[3][b], tr_b[4][b], tr_b[5][b]), ytr_t[b]).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            s = torch.sigmoid(net(*te_b)).cpu().numpy()
        auc = roc_auc_score(yte, s); ap = average_precision_score(yte, s)
        yr_aware = 'YES' if (ug or uz) else 'no'
        rows.append(dict(arm=name, year_aware=yr_aware, auc=auc, ap=ap))
        print(f"  {name:12s} [yr-aware={yr_aware:3s}] held-out-year AUC={auc:.3f} AP={ap:.3f}", flush=True)

    res = pd.DataFrame(rows); res.to_csv(OUTDIR / 'interannual_metrics.csv', index=False)
    print("\n===== INTERANNUAL (train<=2021, test 2022-24) =====", flush=True)
    print(res.round(3).to_string(index=False), flush=True)
    base = res[res.arm == '+week'].iloc[0].ap
    print(f"\nAP lift over year-invariant clock (+week={base:.3f}): " +
          "  ".join(f"{r.arm}={r.ap-base:+.3f}" for _, r in res.iterrows() if r.arm not in ('+week', 'coords')), flush=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    cols = ['#7f7f7f' if r.year_aware == 'no' else '#2ca02c' for _, r in res.iterrows()]
    ax.bar(res.arm, res.ap, color=cols); ax.set_ylabel('held-out-year AP'); ax.grid(alpha=.25, axis='y')
    ax.set_title('Interannual transfer (green=year-aware, gray=year-invariant)'); ax.tick_params(axis='x', rotation=25)
    fig.tight_layout(); fig.savefig(FIGDIR / 'interannual.png', dpi=150); plt.close(fig)
    print(f"\n[done] -> interannual_metrics.csv | fig -> {FIGDIR}/interannual.png", flush=True)


if __name__ == '__main__':
    main()
