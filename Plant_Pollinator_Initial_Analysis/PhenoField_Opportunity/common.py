"""Shared helpers for the PhenoField opportunity evaluations."""
import numpy as np


def part1_hist(events, name):
    """Normalized week-of-year histogram of observed flowering for `name`, or None if sparse."""
    sub = events[events.species == name]
    if len(sub) < 30:
        return None
    wk = ((sub.doy.values - 1) // 7).clip(0, 51)
    h = np.bincount(wk, minlength=52).astype(float)
    return h / h.sum() if h.sum() > 0 else None


def norm_dist(a):
    a = np.where(np.isfinite(a), a, 0.0)
    s = a.sum()
    return a / s if s > 0 else a


def overlap(p, q):
    """Coefficient of overlapping on two normalized weekly curves (Schoener's D)."""
    return float(np.minimum(norm_dist(p), norm_dist(q)).sum())


def eff_weeks(a):
    """Effective number of weeks (inverse Simpson) of a weekly curve; ~52 = flat, low = peaked."""
    p = norm_dist(a)
    s = (p ** 2).sum()
    return float(1.0 / s) if s > 0 else np.nan


def coord_enc(lat, lon):
    """SINR-style spatial encoding: sinusoidal + raw normalized lat/lon (6 dims)."""
    latn = np.asarray(lat) / 90.0
    lonn = np.asarray(lon) / 180.0
    return np.stack([np.sin(np.pi * latn), np.cos(np.pi * latn),
                     np.sin(np.pi * lonn), np.cos(np.pi * lonn), latn, lonn], -1).astype(np.float32)


def fourier_week(week):
    a = 2 * np.pi * np.asarray(week) / 52.0
    return np.stack([np.sin(a), np.cos(a), np.sin(2 * a), np.cos(2 * a)], -1).astype(np.float32)
