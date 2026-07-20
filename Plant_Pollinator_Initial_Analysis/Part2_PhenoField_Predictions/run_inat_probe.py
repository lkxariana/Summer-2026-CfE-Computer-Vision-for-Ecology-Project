"""iNat 4-class field probe — train + apply to the grid.

No saved iNat field probe exists, so we train one on the cached e68 iNat field
features (features_field = z_static576 + z_dynamic192, identical definition to
the grid features) with the iNat 4-class multi-label scheme:
    0 = no_phenology, 1 = budding, 2 = FLOWERING, 3 = fruiting.

Train on inat_test_random, validate on inat_test_spatial (no e68 inat_train
cache exists on disk; the test splits are large -- ~177k rows each -- and serve
only to fit a small frozen-feature head). Then apply to the grid features saved
by run_field_predictions.py.

Outputs predictions_inat_<cells>.parquet (p_flowering = class 2).
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = Path(__file__).resolve().parent
OUTDATA = Path(os.environ.get('PPE_OUT_DIR', HERE.parent / 'Part2_PhenoField_Outputs')) / 'data'
PHENO = Path('/projects/bdbl/cherd/PhenoField')
sys.path.insert(0, str(PHENO))
CACHE = Path('/projects/bdbl/cherd/data/phenofield_cache/e68_large_adaln_both_infonce')
FLOWER_IDX = 2
CLS = ['no_phenology', 'budding', 'flowering', 'fruiting']


def load_split(name):
    z = np.load(CACHE / f'{name}.npz', mmap_mode='r')
    X = np.asarray(z['features_field'], dtype=np.float32)
    y = np.asarray(z['multi_labels'], dtype=np.float32)  # [N,4] multi-hot
    fin = np.isfinite(X).all(1)
    return X[fin], y[fin]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cells', choices=['used', 'all'], default='used')
    ap.add_argument('--epochs', type=int, default=30)
    args = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from utils.probes import train_multi_label_probe, predict_multi_label_probe

    Xtr, ytr = load_split('inat_test_random')
    Xva, yva = load_split('inat_test_spatial')
    print(f"train {Xtr.shape} val {Xva.shape} | flowering prevalence "
          f"tr={ytr[:,FLOWER_IDX].mean():.3f} va={yva[:,FLOWER_IDX].mean():.3f}", flush=True)

    # standardize on train stats (mirrors run_phenology_probe)
    mean = Xtr.mean(0); std = Xtr.std(0) + 1e-6
    Xtr_s = (Xtr - mean) / std
    Xva_s = (Xva - mean) / std

    probe = train_multi_label_probe(
        Xtr_s, ytr, n_classes=4, epochs=args.epochs, lr=3e-4, wd=1e-4,
        probe_arch='mlp', mlp_hidden_dim=128, mlp_n_layers=2,
        batch_size=16384, device=device)

    _, _, pva = predict_multi_label_probe(probe, Xva_s, n_classes=4, device=device)
    print("\nval metrics (per class AP / AUC):")
    for c in range(4):
        if yva[:, c].sum() > 0:
            ap_ = average_precision_score(yva[:, c], pva[:, c])
            auc = roc_auc_score(yva[:, c], pva[:, c])
            print(f"  {CLS[c]:14s} AP={ap_:.3f} AUC={auc:.3f}")

    # ---- apply to grid features ----
    g = np.load(OUTDATA / f'grid_field_features_{args.cells}.npz', allow_pickle=True)
    Xg = (np.asarray(g['features'], np.float32) - mean) / std
    _, _, pg = predict_multi_label_probe(probe, Xg, n_classes=4, device=device)

    species = list(g['species'])
    out = pd.DataFrame({
        'cell_idx': g['cell_idx'], 'centroid_lat': g['centroid_lat'],
        'centroid_lon': g['centroid_lon'], 'week': g['week'].astype(int),
        'species': [species[i] for i in g['sp_i']],
    })
    out['doy'] = (out.week * 7 + 4).clip(1, 365)
    out['p_flowering'] = pg[:, FLOWER_IDX]
    for c in range(4):
        out[f'inat_p_{CLS[c]}'] = pg[:, c]
    OUTDATA.mkdir(parents=True, exist_ok=True)
    out_path = OUTDATA / f'predictions_inat_{args.cells}.parquet'
    out.to_parquet(out_path, index=False)
    print(f"\nsaved {out_path} ({len(out):,} rows)")

    # sanity: peak flowering week for a few species in the first cell
    c0 = out.cell_idx.iloc[0]
    s = out[out.cell_idx == c0]
    print(f"\nSanity (cell {c0}) peak flowering week:")
    for sp in s.species.unique()[:8]:
        ss = s[s.species == sp]
        pk = ss.loc[ss.p_flowering.idxmax()]
        print(f"  {sp:32s} peak wk={int(pk.week):2d} p={pk.p_flowering:.3f}")


if __name__ == '__main__':
    main()
