import argparse
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from antheia.config import load_config
from antheia.store import FeatureStore

SDM_DIR = Path("/scratch/ariana.l/Stage 6 Seed Testing")


def main():
    ap = argparse.ArgumentParser(description="Cache SDM per-species x cell x week activity surfaces")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    store = FeatureStore(cfg)
    bin_idx = {b: i for i, b in enumerate(store.common_bins)}

    man = pd.read_csv(SDM_DIR / "species_manifest.csv")
    man = man[man["species"].isin(set(store.polls))]
    sid2sp = dict(zip(man["species_id"], man["species"]))
    species = sorted(man["species"].unique())
    sp2row = {s: i for i, s in enumerate(species)}
    print(f"SDM species in our pollinator universe: {len(species):,}", flush=True)

    out = cfg["cache_dir"] / f"sdm_surfaces_{len(species)}x{len(store.common_bins)}x52.npy"
    S = np.lib.format.open_memmap(out, mode="w+", dtype=np.float16,
                                  shape=(len(species), len(store.common_bins), 52))
    f = pq.ParquetFile(SDM_DIR / "pollinator_activity_curves.parquet")
    t0, seen, dropped = time.time(), 0, 0
    for bi, batch in enumerate(f.iter_batches(batch_size=2_000_000,
                                              columns=["species_id", "centroid_lat", "centroid_lon",
                                                       "week", "activity_norm"])):
        d = batch.to_pandas()
        d["species"] = d["species_id"].map(sid2sp)
        d = d.dropna(subset=["species"])
        if not len(d):
            continue
        # Documented invariant: SDM centroids sit at bin centre; subtract 0.25 for the corner convention.
        lat = (d["centroid_lat"] - 0.25).round(1)
        lon = (d["centroid_lon"] - 0.25).round(1)
        b = (lat.astype(str) + "_" + lon.astype(str)).map(bin_idx)
        ok = b.notna()
        dropped += int((~ok).sum())
        d, b = d[ok], b[ok].astype(int)
        rows = d["species"].map(sp2row).to_numpy()
        wk = d["week"].clip(0, 51).astype(int).to_numpy()
        np.add.at(S, (rows, b.to_numpy(), wk), d["activity_norm"].to_numpy(np.float16))
        seen += len(d)
        if (bi + 1) % 5 == 0:
            print(f"  batch {bi+1}: {seen:,} rows mapped, {dropped:,} out-of-grid ({time.time()-t0:.0f}s)", flush=True)

    S.flush()
    cov = (np.asarray(S).sum(axis=(1, 2)) > 0).sum()
    np.save(cfg["cache_dir"] / "sdm_species.npy", np.array(species))
    print(f"\nSaved {out} ({S.nbytes/1e9:.2f} GB)")
    print(f"rows mapped {seen:,} | dropped out-of-grid {dropped:,} ({dropped/(seen+dropped):.1%})")
    print(f"species with non-empty surface: {cov:,}/{len(species):,}")
    occ = (np.asarray(S).sum(axis=2) > 0).sum(1)
    print(f"occupied cells per species: median {np.median(occ):.0f}, min {occ.min()}, max {occ.max()}")


if __name__ == "__main__":
    main()
