import numpy as np


def norm_dist(a):
    a = np.where(np.isfinite(a), a, 0.0)
    s = a.sum()
    return a / s if s > 0 else a


def overlap(p, q):
    """Coefficient of overlapping on two normalized weekly curves (Schoener's D)."""
    return float(np.minimum(norm_dist(p), norm_dist(q)).sum())


def coord_enc(lat, lon):
    """SINR-style spatial encoding: sinusoidal + raw normalized lat/lon (6 dims)."""
    x = np.asarray(lat) / 90.0
    y = np.asarray(lon) / 180.0
    return np.stack([np.sin(np.pi * x), np.cos(np.pi * x),
                     np.sin(np.pi * y), np.cos(np.pi * y), x, y], -1).astype(np.float32)


def fourier_loc(lat, lon, freqs=(1, 2, 4, 8)):
    x = np.asarray(lat) / 90.0
    y = np.asarray(lon) / 180.0
    out = []
    for f in freqs:
        out += [np.sin(np.pi * f * x), np.cos(np.pi * f * x), np.sin(np.pi * f * y), np.cos(np.pi * f * y)]
    return np.stack(out, -1).astype(np.float32)


def fourier_week(week):
    w = 2 * np.pi * np.asarray(week) / 52.0
    return np.stack([np.sin(w), np.cos(w), np.sin(2 * w), np.cos(2 * w)], -1).astype(np.float32)
