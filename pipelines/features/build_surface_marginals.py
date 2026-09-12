"""Space and time marginals, and total mass, of every taxon's (cell x week) surface.

The marginalisation test needs, for the same surface P[c, w], the three things the field's overlap
statistics collapse to before comparing two species:

  {side}_surf_space.npy  [taxa, n_cells]  sum_w P[c, w]     what range overlap sees
  {side}_surf_time.npy   [taxa, 52]       sum_c P[c, w]     what phenology overlap sees (unnormalised;
                                                            FCm/ACm are the same curves scaled to sum 1)
  {side}_surf_mass.npy   [taxa]           sum_{c,w} P       what a prevalence product sees

One streaming pass over the memmapped surfaces; nothing is held whole in RAM.
"""
import argparse
from pathlib import Path
import numpy as np

from antheia.paths import REPO_ROOT as ROOT


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", default=ROOT / "data/features")
    ap.add_argument("--chunk", type=int, default=256)
    args = ap.parse_args()
    feat = Path(args.features)
    for side in ("plant", "poll"):
        S = np.load(feat / f"{side}_surfaces.npy", mmap_mode="r")
        n, C, W = S.shape
        space = np.empty((n, C), np.float32); time_ = np.empty((n, W), np.float32); mass = np.empty(n, np.float32)
        for s in range(0, n, args.chunk):
            x = np.asarray(S[s:s + args.chunk], dtype=np.float32)
            space[s:s + len(x)] = x.sum(2); time_[s:s + len(x)] = x.sum(1); mass[s:s + len(x)] = x.sum((1, 2))
            if (s // args.chunk) % 10 == 0:
                print(f"  {side} {s}/{n}", flush=True)
        np.save(feat / f"{side}_surf_space.npy", space); np.save(feat / f"{side}_surf_time.npy", time_)
        np.save(feat / f"{side}_surf_mass.npy", mass)
        print(f"[{side}] {n} taxa: space {space.shape}, time {time_.shape}, mass median {np.median(mass):.1f}", flush=True)


if __name__ == "__main__":
    main()
