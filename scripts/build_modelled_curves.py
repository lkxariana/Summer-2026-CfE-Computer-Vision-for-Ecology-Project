"""Modelled phenology and activity curves, marginalised over space, for the whole universe.

The observation-derived curves in `build_feature_caches.py` exist only for taxa with records. These
are their modelled counterparts: the per-cell surfaces marginalised over space, giving one 52-week
curve for every taxon in the universe, including the zero-shot half.

Cells are weighted by absolute predicted probability, not by the per-cell normalised curve. The
normalised curve is a within-cell shape and carries no information about whether the taxon is there
at all, so averaging it weights a cell outside the range as heavily as one at the range core. Against
observed histograms for plants that have both, probability weighting recovers the peak week within
two weeks for 60% of species against 41% for shape weighting, and matches the overlap the supervised
phenology head achieves.

  FCm.npy  plants x 52       from the opportunity surfaces (e98 direct + text zero-shot)
  ACm.npy  pollinators x 52  from the SDM deliverable

The per-cell surfaces themselves are deliberately not densified -- 13,124 pollinators x 3,335 cells
x 52 weeks would be several terabytes. Location-conditioned models read them from the parquets on
demand; these marginals are the space-free summary.

Rows follow modelled_universe.json order.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from antheia.paths import DATA_ROOT, HF_CACHE

ROOT = Path(__file__).resolve().parents[1]


def normalise(C, n):
    tot = C.sum(1, keepdims=True)
    out = np.divide(C, tot, out=np.zeros_like(C), where=tot > 0).astype(np.float32)
    print(f"  {int((tot[:, 0] > 0).sum())}/{n} rows populated")
    return out


def pollinator_curves(deliverable, labels):
    q2i = {s: i for i, s in enumerate(labels)}
    man = pd.read_csv(Path(deliverable) / "species_manifest.csv")
    sid2row = {int(r.species_id): q2i[r.species] for r in man.itertuples() if r.species in q2i}
    C = np.zeros((len(labels), 52), dtype=np.float64)
    cnt = np.zeros(len(labels), dtype=np.float64)
    pf = pq.ParquetFile(Path(deliverable) / "pollinator_activity_curves.parquet")
    for g in range(pf.num_row_groups):
        d = pf.read_row_group(g, columns=["species_id", "week", "p_activity"]).to_pandas()
        d["row"] = d["species_id"].map(sid2row)
        d = d.dropna(subset=["row"])
        np.add.at(C, (d["row"].to_numpy(dtype=int), d["week"].to_numpy(dtype=int)),
                  d["p_activity"].to_numpy())
        np.add.at(cnt, d["row"].to_numpy(dtype=int), 1.0)
        if g % 50 == 0:
            print(f"  row group {g}/{pf.num_row_groups}", flush=True)
    return normalise(C, len(labels))


def plant_curves(dirs, labels):
    p2i = {s: i for i, s in enumerate(labels)}
    C = np.zeros((len(labels), 52), dtype=np.float64)
    seen = set()
    parts = [f for d in dirs for f in sorted(Path(d).glob("*.parquet"))]
    for i, f in enumerate(parts):
        d = pd.read_parquet(f, columns=["species", "week", "p_flowering"])
        sp = d["species"].iloc[0]
        row = p2i.get(sp)
        if row is None or sp in seen:      # the zero-shot run repeated 263 taxa
            continue
        seen.add(sp)
        np.add.at(C, (np.full(len(d), row), d["week"].to_numpy(dtype=int)),
                  d["p_flowering"].to_numpy())
        if i % 1000 == 0:
            print(f"  part {i}/{len(parts)}", flush=True)
    return normalise(C, len(labels))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--universe", default=ROOT / "data/network/modelled_universe.json")
    ap.add_argument("--deliverable", default=str(DATA_ROOT) + "/pollinator_sdm/deliverable_universe")
    ap.add_argument("--surfaces", nargs="*", default=[str(DATA_ROOT) + "/opportunity_surface_e98",
                                                     str(DATA_ROOT) + "/opportunity_surface_zs"])
    ap.add_argument("--side", choices=["plants", "pollinators", "both"], default="both")
    ap.add_argument("--out", default=ROOT / "data/features")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    u = json.load(open(args.universe))

    if args.side in ("pollinators", "both"):
        print("[ACm] marginalising the SDM deliverable")
        np.save(out / "ACm.npy", pollinator_curves(args.deliverable, [p["label"] for p in u["pollinators"]]))
    if args.side in ("plants", "both"):
        print("[FCm] marginalising the opportunity surfaces")
        np.save(out / "FCm.npy", plant_curves(args.surfaces, [p["label"] for p in u["plants"]]))
    print(f"[wrote] {out}")


if __name__ == "__main__":
    main()
