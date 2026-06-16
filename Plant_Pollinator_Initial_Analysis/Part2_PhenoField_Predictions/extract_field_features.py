"""Stage B — e68 field features for each (cell, week, species).

Image-free: runs the e68 field encoder (CrossModalVAE.forward_field) on the
per-(cell,week) multi-year-mean PRISM window (Stage A) + 2017 AlphaEarth +
species id, and saves the 768-d field feature (z_static[576] + z_dynamic[192]).
These features feed the iNat flowering probe (run_inat_probe.py).

Outputs <OUT>/data/grid_field_features_<cells>.npz.
Defaults to the 20 used_in_analysis cells; pass --cells all for the full grid.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
OUTDIR = Path(os.environ.get('PPE_OUT_DIR', HERE.parent / 'Part2_PhenoField_Outputs'))
OUTDATA = OUTDIR / 'data'
PROJDATA = Path('/projects/bdbl/cherd/Summer-2026-CfE-Computer-Vision-for-Ecology-Project/data')
PHENO = Path('/projects/bdbl/cherd/PhenoField')
sys.path.insert(0, str(PHENO))
AE_PARQUET = Path('/projects/bdbl/cherd/data/conus_eval_grid/alphaearth_2017.parquet')
VOCAB = Path('/projects/bdbl/cherd/data/phenofield_multi_365/species_vocab.json')
E68_CKPT = '/projects/bdbl/bwei/PhenoField/logs/e68_large_adaln_both_infonce/version_0/checkpoints/last.ckpt'
BWEI_ROOT = '/projects/bdbl/bwei/PhenoField'


def load_e68(device):
    """Build CrossModalVAE from the checkpoint's own hyper_parameters using
    bwei's repo (utils.load_model builds the legacy archive wrapper, which has a
    null location_encoder and crashes for this field-only model)."""
    if BWEI_ROOT not in sys.path:
        sys.path.insert(0, BWEI_ROOT)
    try:
        from models.cross_modal_vae import CrossModalVAE
    finally:
        if sys.path and sys.path[0] == BWEI_ROOT:
            sys.path.pop(0)
    ckpt = torch.load(E68_CKPT, map_location='cpu', weights_only=False)
    cfg = OmegaConf.create(ckpt['hyper_parameters'])
    model = CrossModalVAE(OmegaConf.to_container(cfg.model, resolve=True)).to(device).eval()
    state = {k[len('model.'):]: v for k, v in ckpt['state_dict'].items() if k.startswith('model.')}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"CrossModalVAE loaded (missing={len(missing)} unexpected={len(unexpected)})", flush=True)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cells', choices=['used', 'all'], default='used')
    ap.add_argument('--batch_size', type=int, default=1024)
    args = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device={device} cells={args.cells} out={OUTDATA}", flush=True)

    npz = np.load(OUTDATA / 'prism_weekly.npz')
    win_by_key = {(int(c), int(w)): i for i, (c, w) in enumerate(zip(npz['cell_idx'], npz['week']))}
    windows = npz['window']  # [K,365,7] f16
    grid = pd.read_csv(PROJDATA / 'grid_centroids_0.5deg.csv')
    cell_sel = np.where(grid.used_in_analysis.values == 1)[0] if args.cells == 'used' else np.arange(len(grid))

    # AlphaEarth: nearest 2017 grid cell to each centroid
    ae_df = pd.read_parquet(AE_PARQUET)
    emb_cols = [f'emb_{i:02d}' for i in range(64)]
    _, nn = cKDTree(ae_df[['lat', 'lon']].values).query(grid[['centroid_lat', 'centroid_lon']].values)
    ae_by_cell = ae_df[emb_cols].values[nn].astype(np.float32)

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
    print(f"assembled {N:,} (cell,week,species) rows over {len(cell_sel)} cells", flush=True)

    model = load_e68(device)
    feats = np.zeros((N, 768), np.float32)
    t0 = time.time()
    for i in range(0, N, args.batch_size):
        sub = meta.iloc[i:i + args.batch_size]
        m = len(sub)
        prism = np.empty((m, 365, 7), np.float32)
        ae = np.empty((m, 64), np.float32)
        for j, (_, r) in enumerate(sub.iterrows()):
            prism[j] = windows[win_by_key[(int(r.cell_idx), int(r.week))]].astype(np.float32)
            ae[j] = ae_by_cell[int(r.cell_idx)]
        sid = torch.tensor(sp_ids[sub.sp_i.values], dtype=torch.long, device=device)
        prism_t = torch.nan_to_num(torch.from_numpy(prism), nan=0.0).to(device)  # zero NaN days
        with torch.no_grad():
            z_s, z_d = model.forward_field(species_id=sid, prism_window=prism_t,
                                           alphaearth_emb=torch.from_numpy(ae).to(device))
            fb = torch.cat([z_s, z_d], dim=-1)   # [m, 576+192=768], order matches the iNat cache
        feats[i:i + m] = fb.detach().float().cpu().numpy()
        if (i // args.batch_size) % 20 == 0:
            print(f"  feat {i+m:,}/{N:,} ({time.time()-t0:.0f}s)", flush=True)

    OUTDATA.mkdir(parents=True, exist_ok=True)
    np.savez(OUTDATA / f'grid_field_features_{args.cells}.npz',
             features=feats.astype(np.float32),
             cell_idx=meta.cell_idx.values.astype(np.int32),
             week=meta.week.values.astype(np.int8),
             sp_i=meta.sp_i.values.astype(np.int16),
             centroid_lat=meta.centroid_lat.values, centroid_lon=meta.centroid_lon.values,
             species=np.array(plants))
    print(f"saved grid_field_features_{args.cells}.npz ({N:,} rows)", flush=True)


if __name__ == '__main__':
    main()
