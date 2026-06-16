"""Stage A — per-cell x 52-week multi-year-averaged PRISM windows.

For each 0.5-deg grid cell, gather every training observation whose (lat, lon)
falls *inside* the cell (from PhenoField's hf_prism_365.parquet), bin them by
week-of-year, and average their 365-day daily PRISM windows across all years.
The result is one climatological (multi-year-mean) trailing window per
(cell, week) -- exactly the seasonal anchor the PhenoField field encoder reads.

Weeks with zero in-cell observations are skipped and reported (per the spec:
"if we don't have any observations that fit in that grid cell, skip; note it").

Outputs (in <OUT>/data/):
  prism_weekly.npz    cell_idx[K], week[K] (0..51), count[K], window[K,365,7] f16
  coverage.csv        per cell: n_weeks_present, missing weeks (used_in_analysis flagged)

PRISM channel order matches the model: ppt,tmin,tmax,tmean,tdmean,vpdmax,vpdmin.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
OUTDATA = Path(os.environ.get('PPE_OUT_DIR', HERE.parent / 'Part2_PhenoField_Outputs')) / 'data'
DATA = Path('/projects/bdbl/cherd/Summer-2026-CfE-Computer-Vision-for-Ecology-Project/data')
HF_PRISM = Path('/projects/bdbl/cherd/data/phenofield/hf_prism_365.parquet')
PRISM_VARS = ['ppt', 'tmin', 'tmax', 'tmean', 'tdmean', 'vpdmax', 'vpdmin']
N_WEEK, N_DAY, N_VAR = 52, 365, len(PRISM_VARS)


def cell_key(lat, lon):
    """Integer (lat_bin, lon_bin) key at 0.5-deg, matching grid_centroids."""
    return (np.floor(lat * 2).astype(np.int64), np.floor(lon * 2).astype(np.int64))


def main():
    grid = pd.read_csv(DATA / 'grid_centroids_0.5deg.csv')
    n_cells = len(grid)
    # map (lat_bin*2, lon_bin*2) integer key -> cell row index
    klat = np.rint(grid.lat_bin.values * 2).astype(np.int64)
    klon = np.rint(grid.lon_bin.values * 2).astype(np.int64)
    key2cell = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(klat, klon))}
    print(f"grid: {n_cells} cells ({int(grid.used_in_analysis.sum())} used_in_analysis)", flush=True)

    # accumulators: [n_cells, 52, 365, 7] would be huge; use a dict of arrays.
    acc = {}                       # (cell_idx, week) -> float64 [365,7] running sum
    cnt = {}                       # (cell_idx, week) -> int count

    pf = pq.ParquetFile(HF_PRISM)
    cols = ['lat', 'lon', 'obs_date'] + PRISM_VARS
    n_seen = n_inset = 0
    for bi, batch in enumerate(pf.iter_batches(batch_size=50000, columns=cols)):
        lat = batch.column('lat').to_numpy(zero_copy_only=False).astype(np.float64)
        lon = batch.column('lon').to_numpy(zero_copy_only=False).astype(np.float64)
        n_seen += len(lat)
        ka, kb = cell_key(lat, lon)
        # which rows land in one of our cells
        cell_idx = np.array([key2cell.get((int(a), int(b)), -1) for a, b in zip(ka, kb)], dtype=np.int64)
        sel = np.where(cell_idx >= 0)[0]
        if len(sel) == 0:
            continue
        n_inset += len(sel)
        doy = pd.to_datetime(batch.column('obs_date').to_numpy(zero_copy_only=False)).dayofyear.values
        week = np.clip((doy[sel] - 1) // 7, 0, N_WEEK - 1)
        # build [m,365,7] for selected rows (flatten child array; lists are fixed len 365)
        win = np.empty((len(sel), N_DAY, N_VAR), dtype=np.float32)
        for vi, v in enumerate(PRISM_VARS):
            col = batch.column(v)
            flat = col.values.to_numpy(zero_copy_only=False).astype(np.float32)  # [n*365]
            arr = flat.reshape(len(lat), N_DAY)  # [n,365]
            win[:, :, vi] = arr[sel]
        ci = cell_idx[sel]
        for j in range(len(sel)):
            k = (int(ci[j]), int(week[j]))
            if k in acc:
                acc[k] += win[j]; cnt[k] += 1
            else:
                acc[k] = win[j].astype(np.float64); cnt[k] = 1
        if (bi + 1) % 10 == 0:
            print(f"  batch {bi+1}: seen={n_seen:,} in-cell={n_inset:,} pairs={len(acc):,}", flush=True)

    print(f"done streaming: seen={n_seen:,} in-cell-obs={n_inset:,} (cell,week) pairs={len(acc):,}", flush=True)

    # materialize
    keys = sorted(acc.keys())
    K = len(keys)
    cell_arr = np.array([k[0] for k in keys], dtype=np.int32)
    week_arr = np.array([k[1] for k in keys], dtype=np.int8)
    cnt_arr = np.array([cnt[k] for k in keys], dtype=np.int32)
    win_arr = np.empty((K, N_DAY, N_VAR), dtype=np.float16)
    for i, k in enumerate(keys):
        win_arr[i] = (acc[k] / cnt[k]).astype(np.float16)
    OUTDATA.mkdir(parents=True, exist_ok=True)
    np.savez(OUTDATA / 'prism_weekly.npz', cell_idx=cell_arr, week=week_arr,
             count=cnt_arr, window=win_arr, prism_vars=np.array(PRISM_VARS))
    print(f"saved prism_weekly.npz: {K} (cell,week) windows", flush=True)

    # coverage report
    present = {}
    for c, w in zip(cell_arr, week_arr):
        present.setdefault(int(c), set()).add(int(w))
    rows = []
    for i in range(n_cells):
        wks = present.get(i, set())
        miss = sorted(set(range(N_WEEK)) - wks)
        rows.append({
            'cell_idx': i,
            'centroid_lat': grid.centroid_lat.iloc[i], 'centroid_lon': grid.centroid_lon.iloc[i],
            'used_in_analysis': int(grid.used_in_analysis.iloc[i]),
            'n_weeks_present': len(wks),
            'n_obs': int(cnt_arr[cell_arr == i].sum()),
            'missing_weeks': ';'.join(map(str, miss)) if miss else '',
        })
    cov = pd.DataFrame(rows)
    cov.to_csv(OUTDATA / 'coverage.csv', index=False)
    u = cov[cov.used_in_analysis == 1]
    print(f"\ncoverage: cells with >=1 week: {(cov.n_weeks_present>0).sum()}/{n_cells}")
    print(f"used_in_analysis cells: {len(u)} | full 52-wk: {(u.n_weeks_present==52).sum()} "
          f"| median weeks present: {u.n_weeks_present.median():.0f}")
    print(u[['cell_idx', 'centroid_lat', 'centroid_lon', 'n_weeks_present', 'n_obs']].to_string(index=False))


if __name__ == '__main__':
    main()
