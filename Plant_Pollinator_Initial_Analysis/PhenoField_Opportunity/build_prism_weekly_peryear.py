"""Stage A' — per-(cell, week, YEAR) PRISM windows (for the 'average late' test).

Identical to build_prism_weekly.py except the accumulation key carries the
observation year, so we keep one obs-mean window per (cell, week, year) instead
of collapsing across years. The obs-weighted average of these per-year windows
exactly equals build_prism_weekly's multi-year-mean window (same observations,
same weights) — so a downstream early-vs-late comparison isolates only *where*
the cross-year average is taken (climate-space vs probability-space).

Defaults to the 20 used_in_analysis cells (this is a used-arm experiment).

Outputs <OUT>/data/prism_weekly_peryear.npz:
  cell_idx[K], week[K] (0..51), year[K], count[K], window[K,365,7] f16
"""
from __future__ import annotations
import argparse, os, sys
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
    return (np.floor(lat * 2).astype(np.int64), np.floor(lon * 2).astype(np.int64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cells', choices=['used', 'all'], default='used')
    args = ap.parse_args()

    grid = pd.read_csv(DATA / 'grid_centroids_0.5deg.csv')
    klat = np.rint(grid.lat_bin.values * 2).astype(np.int64)
    klon = np.rint(grid.lon_bin.values * 2).astype(np.int64)
    use = (grid.used_in_analysis.values == 1) if args.cells == 'used' else np.ones(len(grid), bool)
    key2cell = {(int(a), int(b)): i for i, (a, b, u) in enumerate(zip(klat, klon, use)) if u}
    print(f"grid: {len(grid)} cells; {len(key2cell)} selected ({args.cells})", flush=True)

    acc, cnt = {}, {}                      # (cell,week,year) -> sum[365,7] / count
    pf = pq.ParquetFile(HF_PRISM)
    cols = ['lat', 'lon', 'obs_date'] + PRISM_VARS
    n_seen = n_inset = 0
    for bi, batch in enumerate(pf.iter_batches(batch_size=50000, columns=cols)):
        lat = batch.column('lat').to_numpy(zero_copy_only=False).astype(np.float64)
        lon = batch.column('lon').to_numpy(zero_copy_only=False).astype(np.float64)
        n_seen += len(lat)
        ka, kb = cell_key(lat, lon)
        cell_idx = np.array([key2cell.get((int(a), int(b)), -1) for a, b in zip(ka, kb)], dtype=np.int64)
        sel = np.where(cell_idx >= 0)[0]
        if len(sel) == 0:
            continue
        n_inset += len(sel)
        dt = pd.to_datetime(batch.column('obs_date').to_numpy(zero_copy_only=False))
        doy = dt.dayofyear.values
        yr = dt.year.values
        week = np.clip((doy[sel] - 1) // 7, 0, N_WEEK - 1)
        win = np.empty((len(sel), N_DAY, N_VAR), dtype=np.float32)
        for vi, v in enumerate(PRISM_VARS):
            flat = batch.column(v).values.to_numpy(zero_copy_only=False).astype(np.float32)
            win[:, :, vi] = flat.reshape(len(lat), N_DAY)[sel]
        ci = cell_idx[sel]; ys = yr[sel]
        for j in range(len(sel)):
            k = (int(ci[j]), int(week[j]), int(ys[j]))
            if k in acc:
                acc[k] += win[j]; cnt[k] += 1
            else:
                acc[k] = win[j].astype(np.float64); cnt[k] = 1
        if (bi + 1) % 10 == 0:
            print(f"  batch {bi+1}: seen={n_seen:,} in-cell={n_inset:,} keys={len(acc):,}", flush=True)

    print(f"done: in-cell-obs={n_inset:,} (cell,week,year) keys={len(acc):,}", flush=True)

    keys = sorted(acc.keys())
    K = len(keys)
    cell_arr = np.array([k[0] for k in keys], dtype=np.int32)
    week_arr = np.array([k[1] for k in keys], dtype=np.int8)
    year_arr = np.array([k[2] for k in keys], dtype=np.int16)
    cnt_arr = np.array([cnt[k] for k in keys], dtype=np.int32)
    win_arr = np.empty((K, N_DAY, N_VAR), dtype=np.float16)
    for i, k in enumerate(keys):
        win_arr[i] = (acc[k] / cnt[k]).astype(np.float16)
    OUTDATA.mkdir(parents=True, exist_ok=True)
    np.savez(OUTDATA / 'prism_weekly_peryear.npz', cell_idx=cell_arr, week=week_arr,
             year=year_arr, count=cnt_arr, window=win_arr, prism_vars=np.array(PRISM_VARS))
    nyr = pd.Series(year_arr).nunique()
    yrs_per = pd.Series(list(zip(cell_arr, week_arr))).groupby(
        pd.Series([(c, w) for c, w in zip(cell_arr, week_arr)])).size()
    print(f"saved prism_weekly_peryear.npz: {K} (cell,week,year) windows; "
          f"{nyr} distinct years; mean years/(cell,week)={yrs_per.mean():.1f}", flush=True)


if __name__ == '__main__':
    main()
