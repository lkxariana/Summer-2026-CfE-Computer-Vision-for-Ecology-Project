"""Generate the plug-in flowering-opportunity surface for all Pheno3M flowering species.

One multi-species SINR presence-background model (species embedding + coords +
Fourier(week) + PPE-L z_dynamic) is trained on Pheno3M flowering presence, then
predicts opportunity per (cell, week, species) over the covered 0.5-deg grid.
Output: partitioned parquet (one file per species block) with the schema a
downstream temporal-SDM consumes.

Everything is on the CLIMATOLOGICAL grid (multi-year-mean PRISM windows) so train
and predict use the same feature construction. z_dynamic is encoded on the fly by
PPE-L and only the scalar opportunity is stored (the 768-d features would be TBs).
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.spatial import cKDTree
from common import coord_enc, fourier_week
from extract_ppel_field import load_ppel, climplicit_by_cell, AE_PARQUET, VOCAB

PROJ = Path('/projects/bdbl/cherd/Summer-2026-CfE-Computer-Vision-for-Ecology-Project')
PART2 = PROJ / 'Plant_Pollinator_Initial_Analysis/Part2_PhenoField_Outputs/data'
PPEL_CACHE = Path('/projects/bdbl/cherd/data/phenofield_cache/FVA_v4_b2_supcon_cc_k10_bioclip_q_PPE_L')
PRISM_NPZ = PART2 / 'prism_weekly.npz'
FLOWER_IDX = 2
NEG_RATIO = 4
EPOCHS = 30


class SINR(nn.Module):
    def __init__(self, n_species, sp_emb=16, hid=256):
        super().__init__()
        self.emb = nn.Embedding(n_species, sp_emb)
        self.net = nn.Sequential(nn.Linear(sp_emb + 6 + 4 + 192, hid), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(hid, hid), nn.GELU(), nn.Dropout(0.1), nn.Linear(hid, 1))

    def forward(self, co, wk, zd, sp):
        return self.net(torch.cat([self.emb(sp), co, wk, zd], -1)).squeeze(-1)


def load_presence(min_obs):
    """(cell-key, week, vocab_species_id) flowering presence from the PPE-L caches."""
    grid = pd.read_csv(PROJ / 'data/grid_centroids_0.5deg.csv')
    klat = np.rint(grid.lat_bin.values * 2).astype(int); klon = np.rint(grid.lon_bin.values * 2).astype(int)
    key2cell = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(klat, klon))}
    keys, weeks, sps = [], [], []
    for split in ['inat_train', 'inat_test_random', 'inat_test_spatial', 'inat_test_species']:
        z = np.load(PPEL_CACHE / f'{split}.npz', mmap_mode='r')
        fl = np.asarray(z['multi_labels'])[:, FLOWER_IDX] == 1
        lat = np.asarray(z['lat'])[fl]; lon = np.asarray(z['lon'])[fl]
        doy = np.asarray(z['doy'])[fl]; sid = np.asarray(z['species_id'])[fl]
        c = np.array([key2cell.get((int(a), int(b)), -1) for a, b in
                      zip(np.floor(lat * 2).astype(int), np.floor(lon * 2).astype(int))])
        ok = c >= 0
        keys.append(c[ok]); weeks.append(np.clip((doy[ok] - 1) // 7, 0, 51)); sps.append(sid[ok])
    cell = np.concatenate(keys); week = np.concatenate(weeks); sp = np.concatenate(sps)
    counts = pd.Series(sp).value_counts()
    keep = set(counts[counts >= min_obs].index.tolist())
    m = np.array([s in keep for s in sp])
    return cell[m], week[m], sp[m], grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default=str(PART2 / 'opportunity_surface'))
    ap.add_argument('--min-obs', type=int, default=5, help='min flowering obs to include a species')
    ap.add_argument('--max-species', type=int, default=0, help='0 = all; else cap for a validation run')
    ap.add_argument('--batch', type=int, default=4096)
    args = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f'[device] {device} out={out_dir}', flush=True)

    # ---- presence + species set ----
    pc, pw, psp, grid = load_presence(args.min_obs)
    species_ids = np.array(sorted(set(psp.tolist())))
    if args.max_species:
        species_ids = species_ids[:args.max_species]
        keep = set(species_ids.tolist()); m = np.array([s in keep for s in psp])
        pc, pw, psp = pc[m], pw[m], psp[m]
    n_sp = len(species_ids)
    sid2idx = {int(s): i for i, s in enumerate(species_ids)}
    vocab = json.load(open(VOCAB)); id2name = {int(v): k for k, v in vocab.items()}
    print(f'[presence] {len(pc):,} flowering (cell,week,obs) over {n_sp:,} species (min_obs={args.min_obs})', flush=True)

    # ---- grid inputs (windows / AE / climplicit) ----
    npz = np.load(PRISM_NPZ); windows = npz['window']
    win_by_key = {(int(c), int(w)): i for i, (c, w) in enumerate(zip(npz['cell_idx'], npz['week']))}
    cov_cells = np.array(sorted(set(int(c) for c, w in win_by_key)))
    lat_c = grid.centroid_lat.values; lon_c = grid.centroid_lon.values
    ae_df = pd.read_parquet(AE_PARQUET)
    _, nn_ae = cKDTree(ae_df[['lat', 'lon']].values).query(grid[['centroid_lat', 'centroid_lon']].values)
    ae_by_cell = ae_df[[f'emb_{i:02d}' for i in range(64)]].values[nn_ae].astype(np.float32)
    clim_by_cell = climplicit_by_cell(grid)
    model = load_ppel(device)

    def encode_zdyn(cells, weeks, vocab_sids):
        """PPE-L z_dynamic for (cell,week,vocab_species) rows, batched."""
        out = np.zeros((len(cells), 192), np.float32)
        for i in range(0, len(cells), args.batch):
            sl = slice(i, i + args.batch); cc = cells[sl]; ww = weeks[sl]
            doy = (ww * 7 + 4).clip(1, 365); mon = np.clip(((doy - 1) // 30) + 1, 1, 12)
            prism = np.stack([windows[win_by_key[(int(cc[j]), int(ww[j]))]] for j in range(len(cc))]).astype(np.float32)
            clim = np.stack([clim_by_cell[cc[j], mon[j] - 1] for j in range(len(cc))]).astype(np.float32)
            with torch.no_grad():
                o = model.encode_field_with_head(
                    [id2name[int(s)] for s in vocab_sids[sl]],
                    torch.tensor(lat_c[cc], dtype=torch.float32, device=device),
                    torch.tensor(lon_c[cc], dtype=torch.float32, device=device),
                    torch.tensor(doy, dtype=torch.long, device=device),
                    species_id=torch.tensor(vocab_sids[sl], dtype=torch.long, device=device),
                    prism_window=torch.nan_to_num(torch.from_numpy(prism), nan=0.0).to(device),
                    alphaearth_emb=torch.from_numpy(ae_by_cell[cc]).to(device),
                    climplicit_emb=torch.from_numpy(clim).to(device), return_features=True)
            out[sl] = o['z_dynamic_field'].float().cpu().numpy()
        return out

    # ---- training set: presence bins + assume-negative background ----
    pres_bins = np.unique(np.stack([pc, pw, psp], 1), axis=0)
    rng = np.random.RandomState(0)
    bg_cell = rng.choice(cov_cells, NEG_RATIO * len(pres_bins))
    bg_week = rng.randint(0, 52, NEG_RATIO * len(pres_bins))
    bg_sp = species_ids[rng.randint(0, n_sp, NEG_RATIO * len(pres_bins))]
    bg_ok = np.array([(int(c), int(w)) in win_by_key for c, w in zip(bg_cell, bg_week)])
    bg_cell, bg_week, bg_sp = bg_cell[bg_ok], bg_week[bg_ok], bg_sp[bg_ok]
    tr_cell = np.concatenate([pres_bins[:, 0], bg_cell]); tr_week = np.concatenate([pres_bins[:, 1], bg_week])
    tr_sp = np.concatenate([pres_bins[:, 2], bg_sp])
    y = np.concatenate([np.ones(len(pres_bins)), np.zeros(len(bg_cell))]).astype(np.float32)
    print(f'[train] {len(pres_bins):,} presence + {len(bg_cell):,} background bins; encoding z_dynamic...', flush=True)
    t0 = time.time(); tr_z = encode_zdyn(tr_cell, tr_week, tr_sp); print(f'  encoded ({time.time()-t0:.0f}s)', flush=True)
    zmean = tr_z.mean(0); zstd = tr_z.std(0) + 1e-6
    co = coord_enc(lat_c[tr_cell], lon_c[tr_cell]); wk = fourier_week(tr_week)
    spidx = np.array([sid2idx[int(s)] for s in tr_sp])

    net = SINR(n_sp).to(device); opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    lf = nn.BCEWithLogitsLoss(); T = lambda a: torch.from_numpy(a).to(device)
    ct, wt, zt, st, yt = T(co), T(wk), T((tr_z - zmean) / zstd), T(spidx).long(), T(y)
    n = len(yt); bs = 16384
    for ep in range(EPOCHS):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            b = perm[i:i + bs]; opt.zero_grad()
            lf(net(ct[b], wt[b], zt[b], st[b]), yt[b]).backward(); opt.step()
    net.eval(); print('[train] done', flush=True)

    # ---- predict full grid, one parquet per species (resumable) ----
    weeks_all = np.repeat(np.arange(52), len(cov_cells))
    cells_all = np.tile(cov_cells, 52)
    valid = np.array([(int(c), int(w)) in win_by_key for c, w in zip(cells_all, weeks_all)])
    cells_all, weeks_all = cells_all[valid], weeks_all[valid]
    doy_all = (weeks_all * 7 + 4).clip(1, 365)
    co_all = coord_enc(lat_c[cells_all], lon_c[cells_all]); wk_all = fourier_week(weeks_all)
    print(f'[predict] {len(cells_all):,} (cell,week) x {n_sp:,} species = {len(cells_all)*n_sp:,} rows', flush=True)
    t0 = time.time()
    for si, sid in enumerate(species_ids):
        fp = out_dir / f'part_{si:05d}.parquet'
        if fp.exists():
            continue
        vsid = np.full(len(cells_all), int(sid), np.int64)
        zd = encode_zdyn(cells_all, weeks_all, vsid)
        with torch.no_grad():
            score = torch.sigmoid(net(T(co_all), T(wk_all), T((zd - zmean) / zstd),
                                      torch.full((len(cells_all),), sid2idx[int(sid)], dtype=torch.long, device=device))).cpu().numpy()
        s = pd.Series(score); norm = score / np.maximum(s.groupby(cells_all).transform('sum').values, 1e-9)
        pd.DataFrame({'species': id2name[int(sid)], 'species_id': int(sid),
                      'cell_idx': cells_all.astype(np.int32), 'centroid_lat': lat_c[cells_all].astype(np.float32),
                      'centroid_lon': lon_c[cells_all].astype(np.float32), 'week': weeks_all.astype(np.int8),
                      'doy': doy_all.astype(np.int16), 'p_flowering': score.astype(np.float32),
                      'norm': norm.astype(np.float32)}).to_parquet(fp, index=False)
        if si % 25 == 0:
            done = si + 1; rate = done / (time.time() - t0 + 1e-9)
            print(f'  species {done}/{n_sp} ({rate*3600:.0f}/h, eta {(n_sp-done)/max(rate,1e-9)/3600:.1f}h)', flush=True)
    print(f'[done] wrote {n_sp:,} species parquets to {out_dir}', flush=True)


if __name__ == '__main__':
    main()
