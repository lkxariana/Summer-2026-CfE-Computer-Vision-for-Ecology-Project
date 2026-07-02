"""SINR-style presence-background flowering-opportunity model + ablation.

Reframes the readout as a species distribution / SDM problem (Cole et al. 2023
SINR; Maxent presence-background). Positives = observed flowering (cell,week,
species) bins; background = assume-negative random (cell,week,species) bins
across the full year. This is the first setup that gives the model off-season
negatives, so it *can* learn the seasonal envelope the observation-conditioned
probe never saw.

Ablation — three inputs, to locate where any sharpness comes from:
  feat      : PPE e68 field feature (768)                 -> does PPE carry it?
  temporal  : species embedding + Fourier(week)           -> pure seasonal prior
  feat_temp : PPE feature (768) + Fourier(week)           -> does PPE add spatial/
                                                             climate modulation on
                                                             top of the prior?

Diagnostics (vs the deployed e68 curve + Part-1 observed):
  - eff weeks (sharpness), overlap vs observed (timing match)
  - peak-week correlation with observed peak (species-SPECIFIC vs generic season)
  - species peak-week spread

Covariates: full-grid e68 features (grid_field_features_all.npz, 4.94M rows,
3335 cells x 50 species), read via mmap + chunked inference.
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
from common import part1_hist, overlap, eff_weeks, coord_enc, fourier_week

HERE = Path(__file__).resolve().parent
OUTDIR = Path(os.environ.get('SINR_OUT_DIR', HERE / 'outputs'))
FIGDIR = OUTDIR / 'figures'
PROJ = Path('/projects/bdbl/cherd/Summer-2026-CfE-Computer-Vision-for-Ecology-Project')
PART2 = PROJ / 'Plant_Pollinator_Initial_Analysis/Part2_PhenoField_Outputs/data'
FEAT_NPZ = PART2 / 'grid_ppel_features_all.npz'      # PPE-L field features (z_dynamic = last 192)
DEPLOYED = PART2 / 'flowering_curves_used.parquet'
NEG_RATIO = 4          # background per presence
EPOCHS = 40


class Net(nn.Module):
    """SINR-style presence model. Spatial (coords) + learned species embedding are
    always on; timing comes from Fourier(week) [data] and/or z_dynamic [PPE temporal]."""
    def __init__(self, use_week, use_zdyn, n_species, zdim=192, sp_emb=16, hid=256):
        super().__init__()
        self.use_week, self.use_zdyn = use_week, use_zdyn
        self.emb = nn.Embedding(n_species, sp_emb)
        d = sp_emb + 6 + (4 if use_week else 0) + (zdim if use_zdyn else 0)
        self.net = nn.Sequential(nn.Linear(d, hid), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(hid, hid), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(hid, 1))

    def forward(self, coords, tw, zdyn, sp):
        parts = [self.emb(sp), coords]
        if self.use_week:
            parts.append(tw)
        if self.use_zdyn:
            parts.append(zdyn)
        return self.net(torch.cat(parts, -1)).squeeze(-1)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    FIGDIR.mkdir(parents=True, exist_ok=True)
    print(f"[device] {device}", flush=True)

    z = np.load(FEAT_NPZ, mmap_mode='r')
    cell = np.asarray(z['cell_idx']); week = np.asarray(z['week']).astype(int); sp_i = np.asarray(z['sp_i']).astype(int)
    species = list(z['species']); Fdim = z['features'].shape[1]; N = len(cell)
    sp2idx = {s: i for i, s in enumerate(species)}
    print(f"[grid] {N:,} rows, {len(np.unique(cell))} cells, {len(species)} species", flush=True)

    # ---- presence set (cell,week,species) from Part-1 flowering events ----
    grid = pd.read_csv(PROJ / 'data/grid_centroids_0.5deg.csv')
    klat = np.rint(grid.lat_bin.values * 2).astype(int); klon = np.rint(grid.lon_bin.values * 2).astype(int)
    key2cell = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(klat, klon))}
    ev = pd.read_parquet(PROJ / 'data/plant_flowering_events.parquet')
    ec = np.array([key2cell.get((int(a), int(b)), -1) for a, b in
                   zip(np.floor(ev.lat.values * 2).astype(int), np.floor(ev.lon.values * 2).astype(int))])
    ew = np.clip((ev.doy.values - 1) // 7, 0, 51)
    esi = ev.species.map(sp2idx).values
    ok = (ec >= 0) & pd.notna(esi)
    pres_key = set(((ec[ok].astype(np.int64) * 52 + ew[ok]) * 50 + esi[ok].astype(np.int64)).tolist())
    grid_key = (cell.astype(np.int64) * 52 + week) * 50 + sp_i
    is_pres = np.fromiter((k in pres_key for k in grid_key), bool, N)
    pos_idx = np.where(is_pres)[0]
    rng = np.random.RandomState(0)
    neg_idx = rng.choice(np.where(~is_pres)[0], size=min(NEG_RATIO * len(pos_idx), (~is_pres).sum()), replace=False)
    print(f"[labels] presence grid-rows={len(pos_idx):,}  background={len(neg_idx):,}", flush=True)

    tr_idx = np.concatenate([pos_idx, neg_idx]); tr_idx.sort()
    y = is_pres[tr_idx].astype(np.float32)
    lat = np.asarray(z['centroid_lat']); lon = np.asarray(z['centroid_lon'])
    Zdtr = np.asarray(z['features'][tr_idx, 576:], np.float32)     # z_dynamic (PPE temporal), train rows
    zmean = Zdtr.mean(0); zstd = Zdtr.std(0) + 1e-6
    co_tr = coord_enc(lat[tr_idx], lon[tr_idx]); tw_tr = fourier_week(week[tr_idx]); sp_tr = sp_i[tr_idx]

    ARMS = [('data', True, False), ('ppe_temporal', False, True), ('data_ppe', True, True)]
    curves = {}
    for name, uw, uz in ARMS:
        torch.manual_seed(0)
        net = Net(uw, uz, len(species)).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
        lossf = nn.BCEWithLogitsLoss()
        ct = torch.from_numpy(co_tr).to(device); tt = torch.from_numpy(tw_tr).to(device)
        zt = torch.from_numpy((Zdtr - zmean) / zstd).to(device)
        st = torch.from_numpy(sp_tr).long().to(device); yt = torch.from_numpy(y).to(device)
        n = len(yt); bs = 8192
        for ep in range(EPOCHS):
            perm = torch.randperm(n, device=device)
            for i in range(0, n, bs):
                b = perm[i:i + bs]
                opt.zero_grad()
                loss = lossf(net(ct[b], tt[b], zt[b], st[b]), yt[b]); loss.backward(); opt.step()
        # ---- predict over ALL grid rows (chunked from mmap) ----
        net.eval(); score = np.zeros(N, np.float32); t0 = time.time()
        with torch.no_grad():
            for i in range(0, N, 200000):
                sl = slice(i, i + 200000); m = min(200000, N - i)
                zc = ((np.asarray(z['features'][sl, 576:], np.float32) - zmean) / zstd) if uz else np.zeros((m, 192), np.float32)
                out = net(torch.from_numpy(coord_enc(lat[sl], lon[sl])).to(device),
                          torch.from_numpy(fourier_week(week[sl])).to(device),
                          torch.from_numpy(zc).to(device),
                          torch.from_numpy(sp_i[sl]).long().to(device))
                score[sl] = torch.sigmoid(out).cpu().numpy()
        curves[name] = score
        print(f"[{name}] trained + scored grid ({time.time()-t0:.0f}s predict)", flush=True)

    # ---- assemble curves table ----
    df = pd.DataFrame({'cell_idx': cell, 'week': week, 'sp_i': sp_i,
                       'species': [species[i] for i in sp_i]})
    for name in curves:
        df[f'score_{name}'] = curves[name]
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTDIR / 'sinr_curves.parquet', index=False)

    # ---- eval on the 20 used cells (comparable to prior EDA) ----
    used = np.where(grid.used_in_analysis.values == 1)[0]
    du = df[df.cell_idx.isin(used)]
    dep = pd.read_parquet(DEPLOYED, columns=['species', 'week', 'inat_p_flowering'])
    ev2 = pd.read_parquet(PROJ / 'data/plant_flowering_events.parquet', columns=['species', 'doy'])

    def usedmean(d, col):
        g = d.groupby('week')[col].mean(); v = np.full(52, np.nan); v[g.index.values.astype(int)] = g.values; return v

    rows = []
    for sp in species:
        h1 = part1_hist(ev2, sp)
        obs_pk = int(np.argmax(h1)) if h1 is not None else np.nan
        d = du[du.species == sp]
        rec = dict(species=sp, obs_eff=eff_weeks(h1) if h1 is not None else np.nan, obs_peak=obs_pk)
        for name in curves:
            v = usedmean(d, f'score_{name}')
            rec[f'eff_{name}'] = eff_weeks(np.nan_to_num(v))
            rec[f'ov_{name}'] = overlap(np.nan_to_num(v), h1) if h1 is not None else np.nan
            rec[f'peak_{name}'] = int(np.nanargmax(v)) if np.isfinite(v).any() else np.nan
        de = dep[dep.species == sp].groupby('week')['inat_p_flowering'].mean()
        e68 = np.full(52, np.nan); e68[de.index.values.astype(int)] = de.values
        rec['eff_e68'] = eff_weeks(np.nan_to_num(e68))
        rec['ov_e68'] = overlap(np.nan_to_num(e68), h1) if h1 is not None else np.nan
        rows.append(rec)
    eda = pd.DataFrame(rows); eda.to_csv(OUTDIR / 'sinr_eda_metrics.csv', index=False)

    mn = lambda s: float(np.nanmean(s))
    pj = eda.dropna(subset=['obs_eff'])
    def peakcorr(col):
        m = pj[[col, 'obs_peak']].dropna()
        return np.corrcoef(m[col], m.obs_peak)[0, 1] if len(m) > 3 else np.nan
    print("\n===== SINR EDA (used cells; truth obs_eff mean=%.1f) ====="%mn(pj.obs_eff), flush=True)
    print(f"{'arm':12s} {'eff(sharp)':>10s} {'overlap_obs':>12s} {'peak_corr_obs':>14s}", flush=True)
    print(f"{'e68 deployed':12s} {mn(eda.eff_e68):10.1f} {mn(pj.ov_e68):12.3f} {'n/a':>14s}", flush=True)
    for name in curves:
        print(f"{name:12s} {mn(eda['eff_'+name]):10.1f} {mn(pj['ov_'+name]):12.3f} {peakcorr('peak_'+name):14.3f}", flush=True)
    print("\n(lower eff = sharper; higher overlap = better timing; higher peak_corr = species-SPECIFIC not generic season)", flush=True)

    # ---- figures: summary + a few sharp species ----
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for name, c in zip(curves, ['#1f77b4', '#2ca02c', '#9467bd']):
        ax[0].scatter(pj.obs_eff, pj['eff_' + name], alpha=0.6, c=c, label=name)
    ax[0].plot([0, 52], [0, 52], 'k--', lw=1); ax[0].set_xlabel('observed eff'); ax[0].set_ylabel('model eff')
    ax[0].set_title('Sharpness vs truth (on diagonal = matches observed breadth)'); ax[0].legend(); ax[0].grid(alpha=.25)
    for name, c in zip(curves, ['#1f77b4', '#2ca02c', '#9467bd']):
        m = pj[['peak_' + name, 'obs_peak']].dropna()
        ax[1].scatter(m.obs_peak, m['peak_' + name], alpha=0.6, c=c, label=name)
    ax[1].plot([0, 52], [0, 52], 'k--', lw=1); ax[1].set_xlabel('observed peak week'); ax[1].set_ylabel('model peak week')
    ax[1].set_title('Species-specific timing (on diagonal = right peak)'); ax[1].legend(); ax[1].grid(alpha=.25)
    fig.tight_layout(); fig.savefig(FIGDIR / 'sinr_summary.png', dpi=150); plt.close(fig)

    for sp in ['Claytonia virginica', 'Sanguinaria canadensis', 'Maianthemum racemosum', 'Asclepias syriaca']:
        if sp not in species: continue
        d = du[du.species == sp]; h1 = part1_hist(ev2, sp); wk = np.arange(52)
        fig, axx = plt.subplots(figsize=(9, 5))
        for name, c in zip(curves, ['#1f77b4', '#2ca02c', '#9467bd']):
            v = usedmean(d, f'score_{name}'); pk = np.nanmax(v)
            if pk and pk > 0: axx.plot(wk, v / pk, c=c, lw=2, label=f'SINR {name}')
        if h1 is not None: axx.plot(wk, h1 / np.nanmax(h1), 'k', lw=2.4, label='observed (Part 1)')
        axx.set_xlim(0, 52); axx.set_ylim(0, 1.1); axx.grid(alpha=.25); axx.legend(fontsize=8)
        axx.set_title(f'{sp} — SINR arms vs observed (peak-normalized)'); axx.set_xlabel('week')
        fig.tight_layout(); fig.savefig(FIGDIR / f'{sp.replace(" ","_")}.png', dpi=150); plt.close(fig)
    print(f"\n[done] curves -> sinr_curves.parquet | metrics -> sinr_eda_metrics.csv | figs -> {FIGDIR}", flush=True)


if __name__ == '__main__':
    main()
