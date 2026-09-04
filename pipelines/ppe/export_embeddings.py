import argparse
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipelines.config import load_config
from pipelines.ppe.backbone import (alphaearth_by_cell, encode_static, encode_zdyn_emb, load_e98,
                                    load_grid, load_prism_weekly, null_species_rows, text_matrix)


def main():
    """Two e98 embedding caches. species_static.npz: z_static [n_vocab,576] — the per-species
    phenology-model embedding (deterministic in the text embedding; constant over space/time).
    grid_zdyn_null.npz: z_dynamic [K,192] per (cell, week) under the trained null-species
    conditioning — a species-free climate-phenology state for the whole grid. The dense
    per-species z_dynamic grid (~TB) is deliberately not cached; re-encode on demand."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--batch", type=int, default=4096)
    args = ap.parse_args()
    cfg = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out_dir) if args.out_dir else cfg["paths"]["out_dir"] / "ppe_embeddings"
    out.mkdir(parents=True, exist_ok=True)
    model = load_e98(cfg, device)

    emb, sp2id = text_matrix(cfg)
    names = np.empty(len(emb), object)
    for n, i in sp2id.items():
        names[i] = n
    z_static = encode_static(model, emb, device)
    np.savez(out / "species_static.npz", z_static=z_static, species=names.astype(str))
    print(f"species_static.npz {z_static.shape}", flush=True)

    windows, win_by_key = load_prism_weekly(cfg)
    grid = load_grid(cfg)
    ae_by_cell = alphaearth_by_cell(cfg, grid)
    keys = sorted(win_by_key)
    cells = np.array([k[0] for k in keys], np.int32)
    weeks = np.array([k[1] for k in keys], np.int8)
    prism = windows[[win_by_key[k] for k in keys]]
    zd = encode_zdyn_emb(model, null_species_rows(model, len(keys)), prism, ae_by_cell[cells],
                         device, args.batch)
    np.savez(out / "grid_zdyn_null.npz", cell_idx=cells, week=weeks, z_dynamic=zd,
             centroid_lat=grid.centroid_lat.values[cells], centroid_lon=grid.centroid_lon.values[cells])
    print(f"grid_zdyn_null.npz {zd.shape}", flush=True)


if __name__ == "__main__":
    main()
