import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BinEncoder(nn.Module):
    """Encodes one species' 52-week activity AT ONE LOCATION into a vector.

    This is the location-conditioned species embedding: the same species gets a different
    vector in different grid cells, because its phenology differs there. Replaces the
    hand-computed per-bin overlap scalar with a learned match.
    """

    def __init__(self, n_weeks=52, hid=64, out=32):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(n_weeks, hid), nn.ReLU(), nn.Linear(hid, out))

    def forward(self, curves):
        return F.normalize(self.mlp(curves), dim=-1)


class LocalMatch(nn.Module):
    """Bilateral local temporal score: encode both sides per shared bin, match, aggregate."""

    def __init__(self, n_weeks=52, out=32):
        super().__init__()
        self.ep = BinEncoder(n_weeks, out=out)
        self.eq = BinEncoder(n_weeks, out=out)
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, p_curves, q_curves, mask):
        """p_curves/q_curves: (..., B, 52) at B sampled shared bins; mask: (..., B) 1 where valid."""
        zp, zq = self.ep(p_curves), self.eq(q_curves)
        per_bin = (zp * zq).sum(-1) * mask
        denom = mask.sum(-1).clamp(min=1.0)
        return self.scale * (per_bin.sum(-1) / denom)


def sample_shared_bins(F_bin, P_bin, pi, qi, n_bins, rng):
    """For each (plant, pollinator) pair, sample up to n_bins co-occupied cells.

    Returns (bin_idx, mask) of shape (len(pi), n_bins). Pairs with no shared cell get mask 0.
    """
    out = np.zeros((len(pi), n_bins), dtype=np.int64)
    mask = np.zeros((len(pi), n_bins), dtype=np.float32)
    for i, (p, q) in enumerate(zip(pi, qi)):
        shared = np.flatnonzero(F_bin[p] & P_bin[q])
        if len(shared) == 0:
            continue
        take = shared if len(shared) <= n_bins else rng.choice(shared, n_bins, replace=False)
        out[i, :len(take)] = take
        mask[i, :len(take)] = 1.0
    return out, mask
