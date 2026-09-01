import argparse
import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from antheia.config import load_config
from antheia.store import FeatureStore

# Per week: total flowering mass, centroid lat/lon of that mass, spatial spread, occupied-cell fraction.
# Aligned across species by construction (no per-species feature ordering), unlike the original V_delta.
STATS = ["mass", "lat_mean", "lon_mean", "spread", "extent"]


def main():
    ap = argparse.ArgumentParser(description="Encode each plant's PPE surface as a 52-week spatiotemporal trajectory")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    store = FeatureStore(cfg)
    S = store.surfaces                                        # (n_plants, n_bins, 52) float16 memmap
    lat = np.array([float(b.split("_")[0]) for b in store.common_bins], dtype=np.float32)
    lon = np.array([float(b.split("_")[1]) for b in store.common_bins], dtype=np.float32)
    lat_c = (lat - lat.mean()) / lat.std()
    lon_c = (lon - lon.mean()) / lon.std()

    n_pl, n_wk = S.shape[0], S.shape[2]
    out = np.zeros((n_pl, len(STATS), n_wk), dtype=np.float32)
    t0 = time.time()
    for i in range(n_pl):
        s = np.asarray(S[i], dtype=np.float32)                # bins x 52
        mass = s.sum(0)
        w = np.divide(s, np.clip(mass, 1e-12, None), out=np.zeros_like(s), where=mass > 1e-12)
        latm = w.T @ lat_c
        lonm = w.T @ lon_c
        var = (w.T @ (lat_c ** 2) - latm ** 2) + (w.T @ (lon_c ** 2) - lonm ** 2)
        out[i, 0] = np.log1p(mass)
        out[i, 1] = latm
        out[i, 2] = lonm
        out[i, 3] = np.sqrt(np.clip(var, 0, None))
        out[i, 4] = (s > (1.0 / len(store.common_bins))).mean(0)
        if (i + 1) % 1500 == 0:
            print(f"  {i + 1}/{n_pl} ({time.time() - t0:.0f}s)", flush=True)

    flat = out.reshape(n_pl, -1)                              # 5 x 52 = 260 dims
    mu, sd = flat.mean(0, keepdims=True), flat.std(0, keepdims=True) + 1e-6
    flat = ((flat - mu) / sd).astype(np.float16)
    np.save(cfg["cache_dir"] / "pheno_trajectory_plants.npy", flat)
    print(f"saved {flat.shape} -> pheno_trajectory_plants.npy ({time.time() - t0:.0f}s)")
    print("per-stat std across species (sanity, should be non-degenerate):")
    for j, nm in enumerate(STATS):
        print(f"  {nm:<9} {out[:, j, :].std():.4f}")


if __name__ == "__main__":
    main()
