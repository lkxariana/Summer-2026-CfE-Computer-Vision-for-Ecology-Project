"""Standalone inference: predict a pollinator's 52-week ACTIVITY curve at any location.

Loads the production per-species-head model shipped in this directory and returns, for a given
pollinator (by iNat taxon_id) and location(s), the weekly activity probability and the normalized
weekly curve (`activity_norm`, sums to 1) -- the drop-in analog of the plant `inat_norm` for the
plant-pollinator overlap coefficient.

Example:
    from predict_pollinator_activity import PollinatorActivityModel
    m = PollinatorActivityModel()                       # loads model_head.pt etc. from this dir
    curve = m.curve(taxon_id=120215, lat=42.0, lon=-88.0)   # -> dict: week, p_activity, activity_norm
    # batch over cells: m.curves(taxon_id, lats, lons) -> [n_cells, 52]
"""
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
OCC = Path('/projects/bdbl/cherd/data/pollinator_sdm/pollinator_occ.npz')
CLIM = '/projects/bdbl/cherd/data/phenofield/embeddings/climplicit/climplicit_monthly_lookup.pt'
FREQS = (1, 2, 4, 8)


def _floc(lat, lon):
    x = np.asarray(lat, float)/90.0; y = np.asarray(lon, float)/180.0; out = []
    for f in FREQS:
        out += [np.sin(np.pi*f*x), np.cos(np.pi*f*x), np.sin(np.pi*f*y), np.cos(np.pi*f*y)]
    return np.stack(out, -1).astype(np.float32)


def _fweek(week):
    w = 2*np.pi*(np.asarray(week)/52.0)
    return np.stack([np.sin(w), np.cos(w), np.sin(2*w), np.cos(2*w)], -1).astype(np.float32)


class _ResLayer(nn.Module):
    def __init__(s, h):
        super().__init__(); s.w1 = nn.Linear(h, h); s.w2 = nn.Linear(h, h); s.a = nn.ReLU(inplace=True); s.d = nn.Dropout(0.5)
    def forward(s, x): return x + s.a(s.w2(s.d(s.a(s.w1(x)))))


class _SINR(nn.Module):
    def __init__(s, in_dim, nsp, h=256, depth=4):
        super().__init__(); s.enc = nn.Sequential(nn.Linear(in_dim, h), nn.ReLU(inplace=True), *[_ResLayer(h) for _ in range(depth)]); s.cls = nn.Linear(h, nsp, bias=False)
    def emb(s, x): return s.enc(x)


class PollinatorActivityModel:
    def __init__(self, model_dir=HERE, device='cpu'):
        self.dev = device
        md = Path(model_dir)
        z = np.load(OCC); self.taxon_ids = z['taxon_ids']
        self.tid2sidx = {int(t): i for i, t in enumerate(self.taxon_ids)}
        nsp = len(self.taxon_ids)
        st = np.load(md/'model_standardization.npz'); self.am = st['clim_annual_mean']; self.asd = st['clim_annual_std']
        in_dim = 16 + 256 + 4
        self.net = _SINR(in_dim, nsp).to(device)
        self.net.load_state_dict(torch.load(md/'model_head.pt', map_location=device)); self.net.eval()
        o = torch.load(CLIM, map_location='cpu', weights_only=False)
        self._tree = cKDTree(np.c_[o['lat'].numpy(), o['lon'].numpy()]); self._emb = o['embeddings']

    def _feats(self, lats, lons):
        _, nn_ = self._tree.query(np.c_[lats, lons])
        clim = self._emb[torch.from_numpy(nn_).long()].numpy()          # [n,12,256]
        wk = np.arange(52); wmonth = np.clip((wk*7+4-1)//30, 0, 11)
        floc = _floc(lats, lons); fwk = _fweek(wk); n = len(lats)
        cr = np.repeat(np.arange(n), 52); wr = np.tile(wk, n)
        return np.concatenate([floc[cr], ((clim[cr, wmonth[wr]]-self.am)/self.asd).astype(np.float32), fwk[wr]], 1).astype(np.float32), n

    def curves(self, taxon_id, lats, lons):
        """Return [n_locations, 52] normalized weekly activity curves."""
        s = self.tid2sidx[int(taxon_id)]
        lats = np.atleast_1d(lats).astype(float); lons = np.atleast_1d(lons).astype(float)
        X, n = self._feats(lats, lons)
        with torch.no_grad():
            p = torch.sigmoid(self.net.emb(torch.from_numpy(X).to(self.dev)) @ self.net.cls.weight[s]).cpu().numpy()
        p = p.reshape(n, 52); ssum = p.sum(1, keepdims=True)
        return np.where(ssum > 0, p/ssum, 1/52)

    def curve(self, taxon_id, lat, lon):
        norm = self.curves(taxon_id, [lat], [lon])[0]
        return {'week': np.arange(52), 'activity_norm': norm}
