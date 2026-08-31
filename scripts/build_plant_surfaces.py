import argparse
import glob
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from antheia.config import load_config
from antheia.store import FeatureStore


def main():
    ap = argparse.ArgumentParser(description="Cache per-plant PPE opportunity surfaces on the common-bin grid")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    store = FeatureStore(cfg)
    bin_idx = {b: i for i, b in enumerate(store.common_bins)}
    out = cfg["cache_dir"] / f"plant_surfaces_{len(store.plants)}x{len(store.common_bins)}x52.npy"
    out.parent.mkdir(parents=True, exist_ok=True)

    S = np.lib.format.open_memmap(out, mode="w+", dtype=np.float16,
                                  shape=(len(store.plants), len(store.common_bins), 52))
    files = sorted(glob.glob(str(cfg["data_root"] / "ppe-outputs" / "opportunity_surface" / "part_*.parquet")))
    t0, done = time.time(), 0
    for fp in files:
        df = pd.read_parquet(fp, columns=["species", "centroid_lat", "centroid_lon", "week", "norm"])
        sp = df["species"].iloc[0]
        if sp not in store.p2i:
            continue
        lat = (np.floor(df["centroid_lat"] / 0.5) * 0.5).round(1)
        lon = (np.floor(df["centroid_lon"] / 0.5) * 0.5).round(1)
        bins = (lat.astype(str) + "_" + lon.astype(str)).map(bin_idx)
        ok = bins.notna()
        row = np.zeros((len(store.common_bins), 52), dtype=np.float32)
        row[bins[ok].astype(int), df.loc[ok, "week"].clip(0, 51).astype(int)] = df.loc[ok, "norm"].astype(np.float32)
        S[store.p2i[sp]] = row.astype(np.float16)
        done += 1
        if done % 1000 == 0:
            print(f"  {done}/{len(store.plants)} plants ({time.time() - t0:.0f}s)", flush=True)

    S.flush()
    print(f"Saved {out} ({done} plants, {S.nbytes / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
