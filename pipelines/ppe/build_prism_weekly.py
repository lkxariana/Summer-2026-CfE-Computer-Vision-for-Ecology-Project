import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipelines.config import load_config, resolve

PRISM_VARS = ["ppt", "tmin", "tmax", "tmean", "tdmean", "vpdmax", "vpdmin"]


def main():
    """Climatological per-(cell, week) PRISM windows: multi-year mean of the 365x7 trailing
    windows of all in-cell observations; zero-observation weeks are skipped and reported."""
    cfg = load_config()
    grid = pd.read_csv(resolve(cfg, "grid_centroids"))
    klat = np.rint(grid.lat_bin.values * 2).astype(np.int64)
    klon = np.rint(grid.lon_bin.values * 2).astype(np.int64)
    key2cell = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(klat, klon))}
    print(f"grid: {len(grid)} cells", flush=True)

    acc, cnt = {}, {}
    pf = pq.ParquetFile(resolve(cfg, "prism_365"))
    n_seen = n_inset = 0
    for bi, batch in enumerate(pf.iter_batches(batch_size=50000, columns=["lat", "lon", "obs_date"] + PRISM_VARS)):
        lat = batch.column("lat").to_numpy(zero_copy_only=False).astype(np.float64)
        lon = batch.column("lon").to_numpy(zero_copy_only=False).astype(np.float64)
        n_seen += len(lat)
        ka, kb = np.floor(lat * 2).astype(np.int64), np.floor(lon * 2).astype(np.int64)
        cell = np.array([key2cell.get((int(a), int(b)), -1) for a, b in zip(ka, kb)])
        sel = np.where(cell >= 0)[0]
        if not len(sel):
            continue
        n_inset += len(sel)
        doy = pd.to_datetime(batch.column("obs_date").to_numpy(zero_copy_only=False)).dayofyear.values
        week = np.clip((doy[sel] - 1) // 7, 0, 51)
        win = np.empty((len(sel), 365, len(PRISM_VARS)), np.float32)
        for vi, v in enumerate(PRISM_VARS):
            win[:, :, vi] = batch.column(v).values.to_numpy(zero_copy_only=False).astype(np.float32).reshape(len(lat), 365)[sel]
        for j, c in enumerate(cell[sel]):
            k = (int(c), int(week[j]))
            if k in acc:
                acc[k] += win[j]; cnt[k] += 1
            else:
                acc[k] = win[j].astype(np.float64); cnt[k] = 1
        if (bi + 1) % 20 == 0:
            print(f"  batch {bi+1}: seen={n_seen:,} in-cell={n_inset:,} pairs={len(acc):,}", flush=True)

    keys = sorted(acc)
    win_arr = np.empty((len(keys), 365, len(PRISM_VARS)), np.float16)
    for i, k in enumerate(keys):
        win_arr[i] = (acc[k] / cnt[k]).astype(np.float16)
    out = cfg["paths"]["out_dir"]
    out.mkdir(parents=True, exist_ok=True)
    cell_arr = np.array([k[0] for k in keys], np.int32)
    week_arr = np.array([k[1] for k in keys], np.int8)
    np.savez(out / "prism_weekly.npz", cell_idx=cell_arr, week=week_arr,
             count=np.array([cnt[k] for k in keys], np.int32), window=win_arr,
             prism_vars=np.array(PRISM_VARS))
    print(f"saved prism_weekly.npz: {len(keys)} (cell,week) windows", flush=True)

    present = pd.DataFrame({"cell_idx": cell_arr, "week": week_arr}).groupby("cell_idx")["week"].nunique()
    cov = pd.DataFrame({"cell_idx": np.arange(len(grid)),
                        "centroid_lat": grid.centroid_lat, "centroid_lon": grid.centroid_lon,
                        "n_weeks_present": present.reindex(np.arange(len(grid)), fill_value=0).values})
    cov.to_csv(out / "coverage.csv", index=False)
    print(f"coverage: {(cov.n_weeks_present > 0).sum()}/{len(grid)} cells with >=1 week")


if __name__ == "__main__":
    main()
