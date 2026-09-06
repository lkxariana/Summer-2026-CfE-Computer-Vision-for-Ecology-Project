"""A shared low-rank basis for the [cells x weeks] surface space, and both sides projected onto it.

Per-cell co-activity is an inner product between two 173,420-dimensional surfaces. Computing it
exactly costs a GPU matvec per plant, which is affordable once but not inside a training loop, and a
network given the raw surface has to discover the space from a random initialisation -- the learned
encoder tried that and lost to no encoder at all.

Projecting both sides onto one orthonormal basis B fixes both problems. The inner product is
preserved, <P_p, Q_q> ~ <P_p B, Q_q B>, so the projection is a drop-in for the exact feature at a
fraction of the cost, and it hands a model a few hundred meaningful dimensions instead of asking it
to learn a spatial basis from 132k interactions.

The basis is fit on a sample of both sides together -- it has to be shared, or the inner product is
not preserved -- by randomized SVD on GPU.

  surface_basis.npy       [n_cells*52, rank] float32   the shared basis
  plant_surf_proj.npy     [n_plants, rank]  float32
  poll_surf_proj.npy      [n_polls,  rank]  float32
"""
import argparse
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def project(surf, B, dev, chunk=512):
    out = np.empty((len(surf), B.shape[1]), np.float32)
    for s in range(0, len(surf), chunk):
        x = torch.from_numpy(np.array(surf[s:s + chunk]).reshape(len(surf[s:s + chunk]), -1)).to(dev).float()
        out[s:s + chunk] = (x @ B).cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", default=ROOT / "data/features")
    ap.add_argument("--rank", type=int, default=256)
    ap.add_argument("--sample", type=int, default=3000, help="taxa per side used to fit the basis")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    feat = Path(args.features)
    dev = args.device if torch.cuda.is_available() else "cpu"

    P = np.load(feat / "plant_surfaces.npy", mmap_mode="r")
    Q = np.load(feat / "poll_surfaces.npy", mmap_mode="r")
    rng = np.random.default_rng(0)
    pi = rng.choice(len(P), min(args.sample, len(P)), replace=False)
    qi = rng.choice(len(Q), min(args.sample, len(Q)), replace=False)
    X = np.vstack([np.array(P[np.sort(pi)]).reshape(len(pi), -1),
                   np.array(Q[np.sort(qi)]).reshape(len(qi), -1)]).astype(np.float32)
    print(f"[fit] {X.shape} sample matrix -> rank {args.rank}", flush=True)
    Xg = torch.from_numpy(X).to(dev)
    _, S, V = torch.svd_lowrank(Xg, q=args.rank, niter=4)
    B = V.contiguous()
    var = (S ** 2)
    print(f"[fit] variance captured by rank {args.rank}: {float(var.sum() / (Xg ** 2).sum()):.4f}", flush=True)
    del Xg
    torch.cuda.empty_cache()

    np.save(feat / "surface_basis.npy", B.cpu().numpy())
    np.save(feat / "plant_surf_proj.npy", project(P, B, dev))
    np.save(feat / "poll_surf_proj.npy", project(Q, B, dev))
    print(f"[wrote] {feat}/surface_basis.npy, plant_surf_proj.npy, poll_surf_proj.npy", flush=True)

    # how well does the projected inner product stand in for the exact one?
    Pp = np.load(feat / "plant_surf_proj.npy"); Qp = np.load(feat / "poll_surf_proj.npy")
    a = rng.integers(0, len(P), 2000); b = rng.integers(0, len(Q), 2000)
    ex = np.array([float(np.dot(np.array(P[i]).ravel().astype(np.float32),
                                np.array(Q[j]).ravel().astype(np.float32))) for i, j in zip(a, b)])
    ap_ = (Pp[a] * Qp[b]).sum(1)
    print(f"[check] exact vs projected co-activity on 2,000 random pairs: "
          f"pearson {np.corrcoef(ex, ap_)[0, 1]:.4f}  "
          f"spearman {np.corrcoef(np.argsort(np.argsort(ex)), np.argsort(np.argsort(ap_)))[0, 1]:.4f}",
          flush=True)


if __name__ == "__main__":
    main()
