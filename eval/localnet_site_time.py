"""Site-time overlap at inference (R-site): does the field's phenology *at the site's cell* sharpen a model's
within-site predictions?

Within a surveyed site space is fixed, so the only spatio-temporal information left is when each species is
active at that cell. For every in-grid network at cell c, the term t(p, q) = log1p(sum_w P[p, c, w] Q[q, c, w])
is computed from the production surfaces, z-scored within the network, and added to the model's z-scored
scores: S' = z(S) + beta * z(t). beta is cross-fitted two-fold over networks (chosen on one half by mean
AUPR, applied to the other), so every network is scored with a beta it did not select. Reads the score blocks
saved by eval/run_localnets.py (blocks.npz); no refit.

  python eval/localnet_site_time.py runs/<name>/<hash>/localnet/s42 [--betas 0 0.25 0.5 1 2]
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from antheia.store import UniverseStore


def z(x):
    x = np.asarray(x, np.float64); sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else np.zeros_like(x)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("bundle")
    ap.add_argument("--betas", nargs="*", type=float, default=[0.0, 0.25, 0.5, 1.0, 2.0, 4.0])
    ap.add_argument("--folds", type=int, default=2)
    args = ap.parse_args()
    b = Path(args.bundle)
    blocks = np.load(b / "blocks.npz"); meta = json.load(open(b / "blocks.json"))
    store = UniverseStore(curves="modelled")
    grid = pd.read_parquet(ROOT / "data/features/grid.parquet")
    tree = cKDTree(np.c_[grid.centroid_lat, grid.centroid_lon])
    Psurf = np.load(ROOT / "data/features/plant_surfaces.npy", mmap_mode="r")
    Qsurf = np.load(ROOT / "data/features/poll_surfaces.npy", mmap_mode="r")

    nets = []
    for i, m in enumerate(meta):
        S, Y = blocks[f"S{i}"].astype(np.float64), blocks[f"Y{i}"]
        T = None
        if m["cell_lat"] is not None:
            d, c = tree.query([m["cell_lat"], m["cell_lon"]])
            if d <= 0.5:
                pi, qi = store.idx_plants(m["plants"]), store.idx_polls(m["polls"])
                P = np.asarray(Psurf[np.sort(pi), c, :], np.float64)[np.argsort(np.argsort(pi))]      # memmap wants sorted indices
                Q = np.asarray(Qsurf[np.sort(qi), c, :], np.float64)[np.argsort(np.argsort(qi))]
                T = np.log1p(P @ Q.T)
        nets.append(dict(network=m["network"], S=S, Y=Y, T=T))
    print(f"[site-time] {len(nets)} networks, {sum(n['T'] is not None for n in nets)} in grid")

    def score(n, beta):
        s = z(n["S"].ravel())
        if n["T"] is not None and beta != 0:
            s = s + beta * z(n["T"].ravel())
        return s

    def evaluate(idx, beta):
        out = []
        for i in idx:
            n = nets[i]; y = n["Y"].ravel(); s = score(n, beta)
            L = int(y.sum()); top = np.argsort(-s)[:L]; hit = y[top].sum() / L
            out.append((average_precision_score(y, s), roc_auc_score(y, s), hit))
        return np.array(out)

    rng = np.random.default_rng(42); order = rng.permutation(len(nets)); folds = np.array_split(order, args.folds)
    rows = []; chosen = []
    for k, test_idx in enumerate(folds):
        train_idx = np.concatenate([f for j, f in enumerate(folds) if j != k])
        curve = {beta: evaluate(train_idx, beta)[:, 0].mean() for beta in args.betas}
        beta = max(curve, key=curve.get); chosen.append(beta)
        ev = evaluate(test_idx, beta); ev0 = evaluate(test_idx, 0.0)
        for i, r, r0 in zip(test_idx, ev, ev0):
            rows.append(dict(network=nets[i]["network"], fold=k, beta=beta, aupr=r[0], auroc=r[1], prec_at_L=r[2],
                             aupr_base=r0[0], auroc_base=r0[1], prec_at_L_base=r0[2], in_grid=nets[i]["T"] is not None))
        print(f"  fold {k}: beta selected on {len(train_idx)} nets = {beta} (curve {', '.join(f'{b_}:{v:.3f}' for b_, v in curve.items())})")
    df = pd.DataFrame(rows)
    d = df.aupr - df.aupr_base
    boots = np.array([d.to_numpy()[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)])
    lo, hi = np.percentile(boots, [2.5, 97.5]); p = 2 * min((boots > 0).mean(), (boots < 0).mean())
    print(f"  mean AUPR {df.aupr_base.mean():.3f} -> {df.aupr.mean():.3f} (delta {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}], p={p:.3f}); "
          f"in-grid only {df[df.in_grid].aupr_base.mean():.3f} -> {df[df.in_grid].aupr.mean():.3f}; "
          f"AUROC {df.auroc_base.mean():.3f} -> {df.auroc.mean():.3f}; precision@L {df.prec_at_L_base.mean():.3f} -> {df.prec_at_L.mean():.3f}")
    df.to_csv(b / "site_time.csv", index=False)
    json.dump(dict(betas=chosen, mean_aupr_base=float(df.aupr_base.mean()), mean_aupr=float(df.aupr.mean()), delta=float(d.mean()),
                   delta_lo=float(lo), delta_hi=float(hi), p=float(p)), open(b / "site_time.json", "w"), indent=1)


if __name__ == "__main__":
    main()
