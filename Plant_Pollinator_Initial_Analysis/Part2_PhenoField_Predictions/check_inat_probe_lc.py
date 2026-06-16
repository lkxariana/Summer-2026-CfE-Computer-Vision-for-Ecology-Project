"""Sanity: does the iNat field probe's flowering AP saturate well below the
177k 'test_random' training set? Train on subsamples + on the union of the two
other cached splits; evaluate flowering AP on a held-out split each time."""
import sys, numpy as np
sys.path.insert(0, '/projects/bdbl/cherd/PhenoField')
from utils.probes import train_multi_label_probe, predict_multi_label_probe
from sklearn.metrics import average_precision_score, roc_auc_score

C = '/projects/bdbl/cherd/data/phenofield_cache/e68_large_adaln_both_infonce/'
FL = 2


def load(name):
    z = np.load(C + name + '.npz', mmap_mode='r')
    X = np.asarray(z['features_field'], np.float32)
    y = np.asarray(z['multi_labels'], np.float32)
    f = np.isfinite(X).all(1)
    return X[f], y[f]


def fit_eval(Xtr, ytr, Xva, yva, tag):
    mean = Xtr.mean(0); std = Xtr.std(0) + 1e-6
    p = train_multi_label_probe((Xtr - mean) / std, ytr, n_classes=4, epochs=30,
                                lr=3e-4, wd=1e-4, probe_arch='mlp', mlp_hidden_dim=128,
                                mlp_n_layers=2, batch_size=16384, device='cpu')
    _, _, pv = predict_multi_label_probe(p, (Xva - mean) / std, n_classes=4, device='cpu')
    ap = average_precision_score(yva[:, FL], pv[:, FL])
    auc = roc_auc_score(yva[:, FL], pv[:, FL])
    print(f"  {tag:42s} flowering AP={ap:.4f} AUC={auc:.4f}", flush=True)


def main():
    Xr, yr = load('inat_test_random')
    Xs, ys = load('inat_test_spatial')   # held-out eval throughout
    Xp, yp = load('inat_test_species')
    print(f"sizes: random={len(Xr)} spatial={len(Xs)} species={len(Xp)} | "
          f"flowering prevalence(spatial)={ys[:,FL].mean():.3f}", flush=True)
    rng = np.random.RandomState(0); idx = rng.permutation(len(Xr))
    for n in [5000, 20000, 60000, len(Xr)]:
        sub = idx[:n]
        fit_eval(Xr[sub], yr[sub], Xs, ys, f"train=random[{n}]")
    Xu = np.concatenate([Xr, Xp]); yu = np.concatenate([yr, yp])
    fit_eval(Xu, yu, Xs, ys, f"train=random+species[{len(Xu)}] (full cached)")


if __name__ == '__main__':
    main()
