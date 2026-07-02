"""Stage B (PPE-L) — image-free field features + supervised head for the grid.

Swaps the e68 backbone for the released PhenoField model PPE-L
(FVA_v4_b2_supcon_cc_k10_bioclip_q_PPE_L). Unlike e68, PPE-L (a) has a
supervised 4-class phenology head (class_logits: 0 no_phenology / 1 budding /
2 FLOWERING / 3 fruiting) so flowering can be read directly, and (b) consumes
day-of-year + a climplicit location embedding.

For each used cell x 52 weeks x 50 species, calls
`PhenoField.encode_field_with_head(species, lat, lon, doy, species_id,
prism_window, alphaearth_emb, climplicit_emb, return_features=True)` and saves:
  features   [N,768]  field feature (z_static_field 576 + z_dynamic_field 192)
  class_logits [N,4]  supervised phenophase head (pre-sigmoid)
plus cell_idx, week, sp_i, species, centroid_lat/lon.

Inputs reused from the e68 pipeline: prism_weekly.npz (per cell,week window),
alphaearth_2017 (nearest cell). New: climplicit monthly lookup (nearest cell x
month), day-of-year (week -> doy).

Outputs <OUT>/data/grid_ppel_features_<cells>.npz.
"""
from __future__ import annotations
import argparse, gc, json, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
OUTDATA = Path(os.environ.get('PPE_OUT_DIR', HERE.parent / 'Part2_PhenoField_Outputs')) / 'data'
PROJDATA = Path('/projects/bdbl/cherd/Summer-2026-CfE-Computer-Vision-for-Ecology-Project/data')
PHENO = Path('/projects/bdbl/cherd/PhenoField')
sys.path.insert(0, str(PHENO))
AE_PARQUET = Path('/projects/bdbl/cherd/data/conus_eval_grid/alphaearth_2017.parquet')
VOCAB = Path('/projects/bdbl/cherd/data/phenofield_multi_365/species_vocab.json')
SPECIES_DIR = '/projects/bdbl/cherd/data/phenofield_multi_365'
PPEL_CKPT = str(PHENO / 'checkpoints/FVA_v4_b2_supcon_cc_k10_bioclip_q_PPE_L/last.ckpt')
CLIMPLICIT_LOOKUP = '/projects/bdbl/cherd/data/phenofield/embeddings/climplicit/climplicit_monthly_lookup.pt'


def load_ppel(device):
    from utils.eval_helpers import load_model
    from data.hf_dataset import load_species_vocab
    cfg = OmegaConf.create(torch.load(PPEL_CKPT, map_location='cpu', weights_only=False)['hyper_parameters']['config'])
    model = load_model(PPEL_CKPT, cfg).to(device).eval()
    model.build_species_cache(load_species_vocab(SPECIES_DIR))
    print("PPE-L loaded + species cache built", flush=True)
    return model


def climplicit_by_cell(grid):
    """Nearest monthly-lookup point per centroid -> [n_cells, 12, 256]."""
    o = torch.load(CLIMPLICIT_LOOKUP, map_location='cpu', weights_only=False)
    tree = cKDTree(np.c_[o['lat'].numpy(), o['lon'].numpy()])
    _, nn = tree.query(grid[['centroid_lat', 'centroid_lon']].values)
    emb = o['embeddings'][nn].float().numpy()          # [n_cells,12,256]
    del o; gc.collect()
    return emb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cells', choices=['used', 'all'], default='used')
    ap.add_argument('--batch_size', type=int, default=1024)
    args = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device={device} cells={args.cells}", flush=True)

    npz = np.load(OUTDATA / 'prism_weekly.npz')
    win_by_key = {(int(c), int(w)): i for i, (c, w) in enumerate(zip(npz['cell_idx'], npz['week']))}
    windows = npz['window']
    grid = pd.read_csv(PROJDATA / 'grid_centroids_0.5deg.csv')
    cell_sel = np.where(grid.used_in_analysis.values == 1)[0] if args.cells == 'used' else np.arange(len(grid))

    ae_df = pd.read_parquet(AE_PARQUET)
    emb_cols = [f'emb_{i:02d}' for i in range(64)]
    _, nn = cKDTree(ae_df[['lat', 'lon']].values).query(grid[['centroid_lat', 'centroid_lon']].values)
    ae_by_cell = ae_df[emb_cols].values[nn].astype(np.float32)
    clim_by_cell = climplicit_by_cell(grid)           # [n_cells,12,256]

    vocab = json.load(open(VOCAB))
    plants = sorted(pd.read_parquet(PROJDATA / 'plant_flowering_events.parquet').species.unique())
    sp_ids = np.array([vocab[s] for s in plants], dtype=np.int64)

    rows = []
    for c in cell_sel:
        for w in range(52):
            if (int(c), w) in win_by_key:
                for si in range(len(plants)):
                    rows.append((int(c), grid.centroid_lat.iloc[c], grid.centroid_lon.iloc[c], w, si))
    meta = pd.DataFrame(rows, columns=['cell_idx', 'centroid_lat', 'centroid_lon', 'week', 'sp_i'])
    N = len(meta)
    doy_all = (meta.week.values * 7 + 4).clip(1, 365)
    month_all = np.clip(((doy_all - 1) // 30) + 1, 1, 12)
    print(f"assembled {N:,} (cell,week,species) rows over {len(cell_sel)} cells", flush=True)

    model = load_ppel(device)
    feats = np.zeros((N, 768), np.float32)
    logits = np.zeros((N, 4), np.float32)
    t0 = time.time()
    for i in range(0, N, args.batch_size):
        sub = meta.iloc[i:i + args.batch_size]; m = len(sub)
        ci = sub.cell_idx.values; wk = sub.week.values; si = sub.sp_i.values
        prism = np.stack([windows[win_by_key[(int(ci[j]), int(wk[j]))]] for j in range(m)]).astype(np.float32)
        ae = ae_by_cell[ci]
        clim = np.stack([clim_by_cell[ci[j], month_all[i + j] - 1] for j in range(m)]).astype(np.float32)
        species = [plants[k] for k in si]
        sid = torch.tensor(sp_ids[si], dtype=torch.long, device=device)
        lat = torch.tensor(sub.centroid_lat.values, dtype=torch.float32, device=device)
        lon = torch.tensor(sub.centroid_lon.values, dtype=torch.float32, device=device)
        doy = torch.tensor(doy_all[i:i + m], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model.encode_field_with_head(
                species, lat, lon, doy, species_id=sid,
                prism_window=torch.nan_to_num(torch.from_numpy(prism), nan=0.0).to(device),
                alphaearth_emb=torch.from_numpy(ae).to(device),
                climplicit_emb=torch.from_numpy(clim).to(device),
                return_features=True)
        feats[i:i + m] = out['features'].float().cpu().numpy()
        logits[i:i + m] = out['class_logits'].float().cpu().numpy()
        if (i // args.batch_size) % 10 == 0:
            print(f"  {i+m:,}/{N:,} ({time.time()-t0:.0f}s)", flush=True)

    OUTDATA.mkdir(parents=True, exist_ok=True)
    np.savez(OUTDATA / f'grid_ppel_features_{args.cells}.npz',
             features=feats, class_logits=logits,
             cell_idx=meta.cell_idx.values.astype(np.int32), week=meta.week.values.astype(np.int8),
             sp_i=meta.sp_i.values.astype(np.int16),
             centroid_lat=meta.centroid_lat.values, centroid_lon=meta.centroid_lon.values,
             species=np.array(plants))
    print(f"saved grid_ppel_features_{args.cells}.npz ({N:,} rows)", flush=True)


if __name__ == '__main__':
    main()
