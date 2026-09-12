"""Observation-derived feature caches, indexed to the frozen modelled universe.

Four arrays keyed on the 0.5-degree grid the phenology and activity surfaces already use, so that
observed and modelled features are directly comparable cell for cell:

  F.npy   plants x cells       bool     occupancy from iNaturalist flowering observations
  P.npy   pollinators x cells  bool     occupancy from the GBIF occurrence extract
  FCo.npy plants x 52          float32  observed flowering curve, row-normalised
  ACo.npy pollinators x 52     float32  observed activity curve, row-normalised
  N.npy   plants x pollinators uint16   co-occurrence, F @ P.T over shared cells

Plant presence is where a plant was observed *flowering*, taken from the same e98 phenology cache
the opportunity surface was fit on, so the two are on identical footing. Pollinator presence is any
occurrence in the GBIF extract.

These are the explicit end of the feature axis: every value is a count of records. Only taxa with
observations of their own can have one, so the zero-shot half of the universe gets an all-zero row --
the honest representation, since the modelled surfaces built separately are what cover them.
`coverage.parquet` records which taxa those are.

Row order is `modelled_universe.json` and must not be reindexed.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipelines.config import load_config, resolve
from pipelines.ppe.generate_opportunity import load_grid, load_vocab, load_presence
from antheia.paths import DATA_ROOT, HF_CACHE

ROOT = Path(__file__).resolve().parents[1]


def to_cell(lat, lon, cell_of):
    """0.5-degree bin -> compact column index; -1 where the cell is outside the modelled grid."""
    key = (np.floor(np.asarray(lat) * 2) / 2 + 0.25).round(2) * 1000 + \
          (np.floor(np.asarray(lon) * 2) / 2 + 0.25).round(2)
    return np.array([cell_of.get(k, -1) for k in key], dtype=np.int64)


def occupancy(cells, idx, n_rows, n_cells):
    F = np.zeros((n_rows, n_cells), dtype=bool)
    ok = (cells >= 0) & (idx >= 0)
    F[idx[ok], cells[ok]] = True
    return F


def curves(weeks, idx, cells, n_rows):
    """Weekly histogram over the same records occupancy uses -- observations outside the modelled
    grid are excluded from both, so a taxon's curve and its cells always agree on support."""
    C = np.zeros((n_rows, 52), dtype=np.float64)
    ok = (idx >= 0) & (cells >= 0)
    np.add.at(C, (idx[ok], weeks[ok]), 1.0)
    tot = C.sum(1, keepdims=True)
    return np.divide(C, tot, out=np.zeros_like(C), where=tot > 0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--universe", default=ROOT / "data/network/modelled_universe.json")
    ap.add_argument("--surface", default=str(DATA_ROOT) + "/opportunity_surface_e98/part_00000.parquet")
    ap.add_argument("--occ", default=str(DATA_ROOT) + "/pollinator_sdm/pollinator_occ_gbifv3.npz")
    ap.add_argument("--out", default=ROOT / "data/features")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cfg = load_config()

    u = json.load(open(args.universe))
    plants = [p["label"] for p in u["plants"]]
    polls = [p["label"] for p in u["pollinators"]]
    p2i = {s: i for i, s in enumerate(plants)}
    q2i = {s: i for i, s in enumerate(polls)}

    # cell_idx indexes the full 6,136-cell CONUS grid; the surfaces cover 3,335 of them. Columns
    # are that compact subset, in cell_idx order, and grid.parquet carries the mapping.
    grid = (pd.read_parquet(args.surface, columns=["cell_idx", "centroid_lat", "centroid_lon"])
            .drop_duplicates("cell_idx").sort_values("cell_idx").reset_index(drop=True))
    grid["col"] = np.arange(len(grid))
    cell_of = {round(la, 2) * 1000 + round(lo, 2): int(c)
               for c, la, lo in zip(grid.col, grid.centroid_lat, grid.centroid_lon)}
    nc = len(grid)
    print(f"[grid] {nc} cells")

    # cell_idx is a row index into the full CONUS grid, the same space load_presence returns,
    # so the surface's cells map straight onto a compact column.
    col_of = np.full(len(load_grid(cfg)), -1, dtype=np.int64)
    col_of[grid.cell_idx.to_numpy()] = grid.col.to_numpy()
    vocab = load_vocab(cfg)
    id2name = {int(v): k for k, v in vocab.items()}
    pcell, pweek, psp = load_presence(cfg, load_grid(cfg), min_obs=1)
    pi = pd.Series([p2i.get(id2name.get(int(s), ""), -1) for s in psp]).to_numpy()
    pcol = col_of[pcell]
    F = occupancy(pcol, pi, len(plants), nc)
    FCo = curves(pweek, pi, pcol, len(plants))
    print(f"[F]   {F.shape} from {len(pcell):,} flowering observations over "
          f"{len(set(psp.tolist())):,} species; {(pi >= 0).sum():,} matched a universe plant; "
          f"{int((F.sum(1) == 0).sum())} empty rows")

    z = np.load(args.occ, allow_pickle=True)
    names = np.array([str(s) for s in z["names"]])
    qi_by_sidx = np.array([q2i.get(s, -1) for s in names])
    qi = qi_by_sidx[z["sidx"].astype(int)]
    qcol = to_cell(z["lat"], z["lon"], cell_of)
    P = occupancy(qcol, qi, len(polls), nc)
    ACo = curves(((z["doy"].astype(int) - 1) // 7).clip(0, 51), qi, qcol, len(polls))
    print(f"[P]   {P.shape} from {len(qi):,} occurrences; "
          f"{(qi >= 0).sum():,} matched a universe pollinator; {int((P.sum(1) == 0).sum())} empty rows")

    N = (F.astype(np.uint16) @ P.astype(np.uint16).T)
    print(f"[N]   {N.shape} max={N.max()} nonzero={np.count_nonzero(N):,}")

    for name, arr in [("F", F), ("P", P), ("FCo", FCo), ("ACo", ACo), ("N", N),
                      ("Frs", F.sum(1).astype(np.int32)), ("Prs", P.sum(1).astype(np.int32))]:
        np.save(out / f"{name}.npy", arr)
    grid.to_parquet(out / "grid.parquet", index=False)

    pd.DataFrame({"side": ["plant"] * len(plants) + ["pollinator"] * len(polls),
                  "label": plants + polls,
                  "n_cells": np.concatenate([F.sum(1), P.sum(1)]),
                  "n_weeks": np.concatenate([(FCo > 0).sum(1), (ACo > 0).sum(1)]),
                  "feature_source": [p["feature_source"] for p in u["plants"]] +
                                    [p["feature_source"] for p in u["pollinators"]]}
                 ).to_parquet(out / "coverage.parquet", index=False)

    tax = pd.concat([
        pd.DataFrame({"side": "plant", "label": plants,
                      "rank": [p["rank"] for p in u["plants"]],
                      "genus": [p["label"].split()[0] for p in u["plants"]]}),
        pd.DataFrame({"side": "pollinator", "label": polls,
                      "rank": [p["rank"] for p in u["pollinators"]],
                      "genus": [p["label"].split()[0] for p in u["pollinators"]]})])
    npl = pd.read_parquet(ROOT / "data/network/nodes_plants.parquet")[["label", "family"]]
    npo = pd.read_parquet(ROOT / "data/network/nodes_pollinators.parquet")[["label", "family", "order"]]
    tax = tax.merge(pd.concat([npl.assign(side="plant"), npo.assign(side="pollinator")]),
                    on=["side", "label"], how="left")
    tax.to_parquet(out / "taxonomy.parquet", index=False)
    print(f"[wrote] {out}")


if __name__ == "__main__":
    main()
