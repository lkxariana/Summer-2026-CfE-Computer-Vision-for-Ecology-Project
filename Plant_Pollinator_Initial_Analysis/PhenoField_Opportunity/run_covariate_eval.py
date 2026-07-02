"""Covariate ablation on a SINR presence-background model, with spatial-block
hold-out AND a data-efficiency sweep. Tests the real question: does PPE's
phenology channel (z_dynamic) add predictive signal over raw climate (GDD) and
a seasonal clock (week) — and does its lift GROW as occurrence data shrinks
(the foundation-model signature)?

Backbone (always on): coords (SINR spatial encoding) + learned species embedding.
Covariates toggled:
  week   Fourier(week)              -- seasonal clock (occurrence prior)
  gdd    accumulated GDD-to-date    -- raw-climate baseline (from PRISM tmean)
  ppe    e68 z_dynamic (192)        -- PPE's phenology-aware channel

Arms: coords / +week / +gdd / +ppe / +week+gdd / +week+ppe.
Eval: spatial-block hold-out (~5deg blocks, 20%), at train-presence fractions
{1.0, 0.3, 0.1, 0.03}. Metric: held-out AUC / average precision. Plot AP vs
fraction per arm.
"""
from __future__ import annotations
import os, sys
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
from extract_ppel_field import climplicit_by_cell

HERE = Path(__file__).resolve().parent
OUTDIR = HERE / 'outputs'; FIGDIR = OUTDIR / 'figures_covariate'
PROJ = Path('/projects/bdbl/cherd/Summer-2026-CfE-Computer-Vision-for-Ecology-Project')
PART2 = PROJ / 'Plant_Pollinator_Initial_Analysis/Part2_PhenoField_Outputs/data'
FEAT_NPZ = PART2 / 'grid_ppel_features_all.npz'      # PPE-L field features (z_dynamic = last 192)
PRISM_NPZ = PART2 / 'prism_weekly.npz'
NEG_RATIO = 4; EPOCHS = 40; BLOCK_DEG = 5.0; TEST_FRAC = 0.20
FRACTIONS = [1.0, 0.3, 0.1, 0.03]
GDD_BASE = 5.0; TMEAN_IDX = 3


class CovNet(nn.Module):
    def __init__(self, uw, ug, uz, uc, n_species, zdim=192, sp_emb=16, hid=256):
        super().__init__(); self.uw, self.ug, self.uz, self.uc = uw, ug, uz, uc
        self.emb = nn.Embedding(n_species, sp_emb)
        d = sp_emb + 6 + (4 if uw else 0) + (1 if ug else 0) + (zdim if uz else 0) + (256 if uc else 0)
        self.net = nn.Sequential(nn.Linear(d, hid), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(hid, hid), nn.GELU(), nn.Dropout(0.1), nn.Linear(hid, 1))

    def forward(self, co, wk, gd, zd, cl, sp):
        parts = [self.emb(sp), co]
        if self.uw: parts.append(wk)
        if self.ug: parts.append(gd)
        if self.uz: parts.append(zd)
        if self.uc: parts.append(cl)
        return self.net(torch.cat(parts, -1)).squeeze(-1)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--feat', default=str(FEAT_NPZ), help='grid features npz (z_dynamic = last 192 dims)')
    ap.add_argument('--tag', default='e68', help='label for outputs')
    ap.add_argument('--seeds', default='0,1,2,3,4', help='comma-sep spatial-block seeds for CIs')
    args = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    FIGDIR.mkdir(parents=True, exist_ok=True); print(f"[device] {device} feat={args.feat} tag={args.tag}", flush=True)
    z = np.load(args.feat, mmap_mode='r')
    cell = np.asarray(z['cell_idx']); week = np.asarray(z['week']).astype(int); sp_i = np.asarray(z['sp_i']).astype(int)
    lat = np.asarray(z['centroid_lat']); lon = np.asarray(z['centroid_lon']); species = list(z['species']); N = len(cell)
    sp2idx = {s: i for i, s in enumerate(species)}

    # GDD-to-date per (cell,week) from PRISM tmean window (last `doy` days = current-year portion)
    pz = np.load(PRISM_NPZ); pw_win = pz['window']; pw_c = pz['cell_idx'].astype(int); pw_w = pz['week'].astype(int)
    gdd_by_key = {}
    for i in range(len(pw_c)):
        doy = int(pw_w[i]) * 7 + 4
        tmean = pw_win[i, 365 - doy:, TMEAN_IDX].astype(np.float32)
        gdd_by_key[(int(pw_c[i]), int(pw_w[i]))] = float(np.nansum(np.clip(tmean - GDD_BASE, 0, None)))
    gdd = np.array([gdd_by_key.get((int(cell[i]), int(week[i])), 0.0) for i in range(N)], np.float32)
    gdd = (gdd - gdd.mean()) / (gdd.std() + 1e-6)

    # climplicit per (cell,month) -- SOTA year-invariant climate baseline
    month = np.clip(((week*7+4-1)//30)+1, 1, 12)

    # presence mask
    grid = pd.read_csv(PROJ / 'data/grid_centroids_0.5deg.csv')
    clim_by_cell = climplicit_by_cell(grid)   # [n_cells,12,256]
    klat = np.rint(grid.lat_bin.values*2).astype(int); klon = np.rint(grid.lon_bin.values*2).astype(int)
    key2cell = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(klat, klon))}
    ev = pd.read_parquet(PROJ / 'data/plant_flowering_events.parquet')
    ec = np.array([key2cell.get((int(a), int(b)), -1) for a, b in
                   zip(np.floor(ev.lat.values*2).astype(int), np.floor(ev.lon.values*2).astype(int))])
    ew = np.clip((ev.doy.values-1)//7, 0, 51); esi = ev.species.map(sp2idx).values; ok = (ec >= 0) & pd.notna(esi)
    pres_key = set(((ec[ok].astype(np.int64)*52+ew[ok])*50+esi[ok].astype(np.int64)).tolist())
    grid_key = (cell.astype(np.int64)*52+week)*50+sp_i
    is_pres = np.fromiter((k in pres_key for k in grid_key), bool, N)

    blk = np.floor(lat/BLOCK_DEG).astype(int)*1000 + np.floor(lon/BLOCK_DEG).astype(int)

    def pool(mask, rng, pos_frac=1.0):
        p = np.where(is_pres & mask)[0]
        if pos_frac < 1.0:
            p = rng.choice(p, size=max(int(len(p)*pos_frac), 50), replace=False)
        ncand = np.where((~is_pres) & mask)[0]
        n = rng.choice(ncand, size=min(NEG_RATIO*len(p), len(ncand)), replace=False)
        idx = np.concatenate([p, n]); idx.sort(); return idx, is_pres[idx].astype(np.float32)

    def cov(idx):
        return (coord_enc(lat[idx], lon[idx]), fourier_week(week[idx]), gdd[idx][:, None],
                np.asarray(z['features'][idx, 576:], np.float32),
                clim_by_cell[cell[idx], month[idx]-1].astype(np.float32), sp_i[idx])

    ARMS = [('coords', 0, 0, 0, 0), ('+week', 1, 0, 0, 0), ('+gdd', 0, 1, 0, 0),
            ('+climplicit', 0, 0, 0, 1), ('+ppe', 0, 0, 1, 0),
            ('+week+ppe', 1, 0, 1, 0), ('+climplicit+ppe', 0, 0, 1, 1)]
    T = lambda a: torch.from_numpy(a).to(device)
    SEEDS = [int(s) for s in args.seeds.split(',')]
    rows = []
    for seed in SEEDS:
        rng = np.random.RandomState(seed)
        ublk = np.unique(blk).copy(); rng.shuffle(ublk)
        row_test = np.isin(blk, ublk[:int(len(ublk)*TEST_FRAC)])
        te_idx, yte = pool(row_test, rng, 1.0)
        co_te, wk_te, gd_te, zd_te, cl_te, sp_te = cov(te_idx)
        print(f"[seed {seed}] test rows={len(te_idx):,} (pos={int(yte.sum()):,})", flush=True)
        for frac in FRACTIONS:
            tr_idx, ytr = pool(~row_test, rng, frac)
            co_tr, wk_tr, gd_tr, zd_tr, cl_tr, sp_tr = cov(tr_idx)
            zmean = zd_tr.mean(0); zstd = zd_tr.std(0) + 1e-6
            zd_trs = (zd_tr - zmean) / zstd; zd_tes = (zd_te - zmean) / zstd
            cmean = cl_tr.mean(0); cstd = cl_tr.std(0) + 1e-6
            cl_trs = (cl_tr - cmean) / cstd; cl_tes = (cl_te - cmean) / cstd
            for name, uw, ug, uz, uc in ARMS:
                torch.manual_seed(0)
                net = CovNet(bool(uw), bool(ug), bool(uz), bool(uc), len(species)).to(device)
                opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5); lf = nn.BCEWithLogitsLoss()
                ct, wt, gt, zt, clt, st, yt = T(co_tr), T(wk_tr), T(gd_tr), T(zd_trs), T(cl_trs), T(sp_tr).long(), T(ytr)
                n = len(yt); bs = 8192
                for ep in range(EPOCHS):
                    perm = torch.randperm(n, device=device)
                    for i in range(0, n, bs):
                        b = perm[i:i+bs]; opt.zero_grad()
                        lf(net(ct[b], wt[b], gt[b], zt[b], clt[b], st[b]), yt[b]).backward(); opt.step()
                net.eval()
                with torch.no_grad():
                    s = torch.sigmoid(net(T(co_te), T(wk_te), T(gd_te), T(zd_tes), T(cl_tes), T(sp_te).long())).cpu().numpy()
                rows.append(dict(seed=seed, frac=frac, arm=name, auc=roc_auc_score(yte, s), ap=average_precision_score(yte, s)))
        print(f"[seed {seed}] done", flush=True)

    res = pd.DataFrame(rows); res.to_csv(OUTDIR / f'covariate_eval_metrics_{args.tag}.csv', index=False)
    agg = res.groupby(['frac', 'arm'])['ap'].agg(['mean', 'std', 'count']).reset_index()
    nseed = res.seed.nunique()
    print(f"\n===== held-out AP mean±std over {nseed} spatial-block seeds =====", flush=True)
    mean_p = agg.pivot(index='frac', columns='arm', values='mean'); std_p = agg.pivot(index='frac', columns='arm', values='std')
    print(mean_p.round(3).to_string(), flush=True)
    # PPE vs climplicit gap with CI (does PPE beat the SOTA climate baseline, per-seed paired)
    print("\n+ppe − +climplicit (paired per seed): mean ± std", flush=True)
    for frac in FRACTIONS:
        d = res[res.frac == frac].pivot(index='seed', columns='arm', values='ap')
        gap = d['+ppe'] - d['+climplicit']
        print(f"  frac={frac:<5} +ppe−climplicit = {gap.mean():+.3f} ± {gap.std():.3f}  (+ppe−week = {(d['+ppe']-d['+week']).mean():+.3f})", flush=True)

    colors = {'coords': '#7f7f7f', '+week': '#1f77b4', '+gdd': '#ff7f0e', '+climplicit': '#e377c2',
              '+ppe': '#2ca02c', '+week+ppe': '#9467bd', '+climplicit+ppe': '#17becf'}
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for name in colors:
        m = mean_p[name].sort_index(); s = std_p[name].reindex(m.index)
        ax[0].errorbar(m.index, m.values, yerr=s.values, fmt='o-', color=colors[name], label=name, capsize=3)
    ax[0].set_xscale('log'); ax[0].set_xlabel('train presence fraction'); ax[0].set_ylabel('held-out AP')
    ax[0].set_title(f'Data-efficiency (mean±std, {nseed} seeds)'); ax[0].legend(fontsize=8); ax[0].grid(alpha=.25)
    # paired gap of each covariate vs the clock, mean±std over seeds
    for name, c in [('+ppe', '#2ca02c'), ('+climplicit', '#e377c2'), ('+gdd', '#ff7f0e'), ('+climplicit+ppe', '#17becf')]:
        gm, gs = [], []
        for frac in FRACTIONS:
            d = res[res.frac == frac].pivot(index='seed', columns='arm', values='ap'); g = d[name] - d['+week']
            gm.append(g.mean()); gs.append(g.std())
        ax[1].errorbar(FRACTIONS, gm, yerr=gs, fmt='o-', color=c, label=f'{name} − week', capsize=3)
    ax[1].axhline(0, color='k', lw=1, ls='--'); ax[1].set_xscale('log')
    ax[1].set_xlabel('train presence fraction'); ax[1].set_ylabel('AP lift over clock (+week)')
    ax[1].set_title('Lift over clock (mean±std)'); ax[1].legend(fontsize=8); ax[1].grid(alpha=.25)
    fig.suptitle(f'covariate = {args.tag} ({nseed} spatial-block seeds)'); fig.tight_layout()
    fig.savefig(FIGDIR / f'data_efficiency_{args.tag}.png', dpi=150); plt.close(fig)
    print(f"\n[done] metrics -> covariate_eval_metrics_{args.tag}.csv | fig -> {FIGDIR}/data_efficiency_{args.tag}.png", flush=True)


if __name__ == '__main__':
    main()
