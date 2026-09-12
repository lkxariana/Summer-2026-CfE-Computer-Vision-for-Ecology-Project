"""Dense per-cell x per-week surfaces for every taxon, as memmapped arrays.

The marginal curves in `build_modelled_curves.py` collapse space away. These keep it: a taxon's
predicted probability in every grid cell in every week, which is what a location-conditioned model
needs and what the marginals cannot reconstruct. Continental phenology is nearly synchronous -- a
random plant-pollinator pair already overlaps 0.50 against 0.59 for a true pair -- so whatever
temporal signal exists has to be local.

  plant_surfaces.npy  [n_plants, n_cells, 52] float16   from the opportunity surfaces
  poll_surfaces.npy   [n_polls,  n_cells, 52] float16   from the SDM deliverable

Absolute probability, not the per-cell normalised shape, so a cell outside the range contributes
what it should: nothing. 3.8 GB and 4.6 GB respectively; both are written and read as memmaps and
never held whole in RAM.

Rows follow modelled_universe.json order; columns follow data/features/grid.parquet.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from antheia.paths import DATA_ROOT, HF_CACHE

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--universe", default=ROOT / "data/network/modelled_universe.json")
    ap.add_argument("--features", default=ROOT / "data/features")
    ap.add_argument("--surfaces", nargs="*", default=[str(DATA_ROOT) + "/opportunity_surface_e98",
                                                     str(DATA_ROOT) + "/opportunity_surface_zs"])
    ap.add_argument("--deliverable", default=str(DATA_ROOT) + "/pollinator_sdm/deliverable_universe")
    ap.add_argument("--side", choices=["plants", "pollinators", "both"], default="both")
    args = ap.parse_args()
    feat = Path(args.features)
    u = json.load(open(args.universe))
    grid = pd.read_parquet(feat / "grid.parquet")
    col_of = np.full(int(grid.cell_idx.max()) + 1, -1, dtype=np.int64)
    col_of[grid.cell_idx.to_numpy()] = grid.col.to_numpy()
    nc = len(grid)

    if args.side in ("plants", "both"):
        plants = [p["label"] for p in u["plants"]]
        p2i = {s: i for i, s in enumerate(plants)}
        out = np.lib.format.open_memmap(feat / "plant_surfaces.npy", mode="w+",
                                        dtype=np.float16, shape=(len(plants), nc, 52))
        seen = set()
        parts = [f for d in args.surfaces for f in sorted(Path(d).glob("*.parquet"))]
        for i, f in enumerate(parts):
            d = pd.read_parquet(f, columns=["species", "cell_idx", "week", "p_flowering"])
            sp = d["species"].iloc[0]
            row = p2i.get(sp)
            if row is None or sp in seen:
                continue
            seen.add(sp)
            c = col_of[d["cell_idx"].to_numpy()]
            ok = c >= 0
            out[row, c[ok], d["week"].to_numpy()[ok]] = d["p_flowering"].to_numpy()[ok]
            if i % 1000 == 0:
                print(f"  plant part {i}/{len(parts)}", flush=True)
        out.flush()
        print(f"[plants] {len(seen)}/{len(plants)} filled -> {feat/'plant_surfaces.npy'}", flush=True)

    if args.side in ("pollinators", "both"):
        polls = [p["label"] for p in u["pollinators"]]
        q2i = {s: i for i, s in enumerate(polls)}
        man = pd.read_csv(Path(args.deliverable) / "species_manifest.csv")
        sid2row = {int(r.species_id): q2i[r.species] for r in man.itertuples() if r.species in q2i}
        out = np.lib.format.open_memmap(feat / "poll_surfaces.npy", mode="w+",
                                        dtype=np.float16, shape=(len(polls), nc, 52))
        pf = pq.ParquetFile(Path(args.deliverable) / "pollinator_activity_curves.parquet")
        for g in range(pf.num_row_groups):
            d = pf.read_row_group(g, columns=["species_id", "cell_idx", "week", "p_activity"]).to_pandas()
            r = d["species_id"].map(sid2row)
            keep = r.notna()
            c = col_of[d.loc[keep, "cell_idx"].to_numpy()]
            ok = c >= 0
            out[r[keep].to_numpy(dtype=int)[ok], c[ok],
                d.loc[keep, "week"].to_numpy()[ok]] = d.loc[keep, "p_activity"].to_numpy()[ok]
            if g % 25 == 0:
                print(f"  pollinator row group {g}/{pf.num_row_groups}", flush=True)
        out.flush()
        print(f"[pollinators] -> {feat/'poll_surfaces.npy'}", flush=True)


if __name__ == "__main__":
    main()
