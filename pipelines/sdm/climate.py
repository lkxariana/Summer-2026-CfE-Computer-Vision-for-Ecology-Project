import numpy as np
import torch
from scipy.spatial import cKDTree
from pipelines.config import resolve


def load_lookup(cfg):
    """The climplicit monthly lookup is a 33 GB tensor table; loading needs ~40 GB RAM."""
    o = torch.load(resolve(cfg, "climplicit"), map_location="cpu", weights_only=False)
    return cKDTree(np.c_[o["lat"].numpy(), o["lon"].numpy()]), o["embeddings"]


def gather_obs(cfg, lat, lon, doy, chunk=200_000):
    """(ann [N,256], mon [N,256]): annual-mean and observation-month embeddings, cached per
    occurrence-set size (clim_gathered.npz is the legacy 3.3M-point cache)."""
    sd = cfg["paths"]["sdm_data"]
    cache = sd / f"clim_gathered_{len(lat)}.npz"
    for p in (cache, sd / "clim_gathered.npz"):
        if p.exists():
            d = np.load(p)
            if len(d["ann"]) == len(lat):
                return d["ann"], d["mon"]
    tree, emb = load_lookup(cfg)
    month = np.clip((np.asarray(doy) - 1) // 30, 0, 11)
    N = len(lat)
    ann = np.empty((N, 256), np.float32)
    mon = np.empty((N, 256), np.float32)
    for i in range(0, N, chunk):
        j = min(i + chunk, N)
        _, nn = tree.query(np.c_[lat[i:j], lon[i:j]], workers=-1)
        e = emb[torch.from_numpy(nn).long()]
        ann[i:j] = e.mean(1).numpy()
        mon[i:j] = e[torch.arange(j - i), torch.from_numpy(month[i:j]).long()].numpy()
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, ann=ann, mon=mon)
    return ann, mon


def gather_cells(cfg, lat, lon):
    """Per-cell monthly embeddings [n_cells, 12, 256]."""
    tree, emb = load_lookup(cfg)
    _, nn = tree.query(np.c_[lat, lon], workers=-1)
    return emb[torch.from_numpy(nn).long()].numpy()
