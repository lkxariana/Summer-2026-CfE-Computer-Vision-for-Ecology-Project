import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipelines.common import coord_enc, fourier_week
from pipelines.config import load_config, resolve
from pipelines.ppe.backbone import Z_DYN, alphaearth_by_cell, encode_zdyn, load_e98, load_grid, load_prism_weekly, load_vocab

FLOWER_IDX = 2
NEG_RATIO = 4
SPLITS = ["inat_train", "inat_test_random", "inat_test_spatial", "inat_test_species"]


class SINR(nn.Module):
    def __init__(self, n_species, sp_emb=16, hid=256):
        super().__init__()
        self.emb = nn.Embedding(n_species, sp_emb)
        self.net = nn.Sequential(nn.Linear(sp_emb + 6 + 4 + Z_DYN, hid), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(hid, hid), nn.GELU(), nn.Dropout(0.1), nn.Linear(hid, 1))

    def forward(self, co, wk, zd, sp):
        return self.net(torch.cat([self.emb(sp), co, wk, zd], -1)).squeeze(-1)


def load_presence(cfg, grid, min_obs):
    """(cell, week, vocab_species_id) flowering presence rows from the e98 caches."""
    klat = np.rint(grid.lat_bin.values * 2).astype(int)
    klon = np.rint(grid.lon_bin.values * 2).astype(int)
    key2cell = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(klat, klon))}
    cache = resolve(cfg, "ppe_cache")
    keys, weeks, sps = [], [], []
    for split in SPLITS:
        z = np.load(cache / f"{split}.npz", mmap_mode="r")
        fl = np.asarray(z["multi_labels"])[:, FLOWER_IDX] == 1
        lat = np.asarray(z["lat"])[fl]; lon = np.asarray(z["lon"])[fl]
        doy = np.asarray(z["doy"])[fl]; sid = np.asarray(z["species_id"])[fl]
        c = np.array([key2cell.get((int(a), int(b)), -1) for a, b in
                      zip(np.floor(lat * 2).astype(int), np.floor(lon * 2).astype(int))])
        ok = c >= 0
        keys.append(c[ok]); weeks.append(np.clip((doy[ok] - 1) // 7, 0, 51)); sps.append(sid[ok])
    cell = np.concatenate(keys); week = np.concatenate(weeks); sp = np.concatenate(sps)
    counts = pd.Series(sp).value_counts()
    keep = set(counts[counts >= min_obs].index.tolist())
    m = np.array([s in keep for s in sp])
    return cell[m], week[m], sp[m]


def species_filter(path, vocab):
    spec = json.load(open(path))
    names = spec["plants"] if isinstance(spec, dict) else spec
    names = [n["label"] if isinstance(n, dict) else n for n in names]
    return {vocab[n] for n in names if n in vocab}


def main():
    """Flowering-opportunity surface: multi-species SINR (species embedding + coords +
    Fourier(week) + e98 z_dynamic) on cache flowering presence vs uniform background, predicted
    per (cell, week, species) over the climatological grid; one parquet per species, resumable."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--min-obs", type=int, default=5)
    ap.add_argument("--max-species", type=int, default=0, help="0 = all; else cap for a validation run")
    ap.add_argument("--species-file", default=None, help="JSON species list or modelled_universe.json")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=4096)
    args = ap.parse_args()
    cfg = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir) if args.out_dir else cfg["paths"]["out_dir"] / "opportunity_surface"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[device] {device} out={out_dir}", flush=True)

    grid = load_grid(cfg)
    vocab = load_vocab(cfg)
    id2name = {int(v): k for k, v in vocab.items()}
    pc, pw, psp = load_presence(cfg, grid, args.min_obs)
    species_ids = np.array(sorted(set(psp.tolist())))
    if args.species_file:
        allowed = species_filter(args.species_file, vocab)
        species_ids = np.array([s for s in species_ids if int(s) in allowed])
    if args.max_species:
        species_ids = species_ids[:args.max_species]
    keep = set(species_ids.tolist())
    m = np.array([s in keep for s in psp])
    pc, pw, psp = pc[m], pw[m], psp[m]
    n_sp = len(species_ids)
    sid2idx = {int(s): i for i, s in enumerate(species_ids)}
    print(f"[presence] {len(pc):,} flowering (cell,week,obs) over {n_sp:,} species (min_obs={args.min_obs})", flush=True)

    windows, win_by_key = load_prism_weekly(cfg)
    cov_cells = np.array(sorted({c for c, _ in win_by_key}))
    lat_c = grid.centroid_lat.values; lon_c = grid.centroid_lon.values
    ae_by_cell = alphaearth_by_cell(cfg, grid)
    model = load_e98(cfg, device)

    def zdyn(cells, weeks, vocab_sids):
        prism = windows[[win_by_key[(int(c), int(w))] for c, w in zip(cells, weeks)]]
        return encode_zdyn(model, vocab_sids, prism, ae_by_cell[cells], device, args.batch)

    pres_bins = np.unique(np.stack([pc, pw, psp], 1), axis=0)
    rng = np.random.default_rng(0)
    bg_cell = rng.choice(cov_cells, NEG_RATIO * len(pres_bins))
    bg_week = rng.integers(0, 52, NEG_RATIO * len(pres_bins))
    bg_sp = species_ids[rng.integers(0, n_sp, NEG_RATIO * len(pres_bins))]
    bg_ok = np.array([(int(c), int(w)) in win_by_key for c, w in zip(bg_cell, bg_week)])
    bg_cell, bg_week, bg_sp = bg_cell[bg_ok], bg_week[bg_ok], bg_sp[bg_ok]
    tr_cell = np.concatenate([pres_bins[:, 0], bg_cell])
    tr_week = np.concatenate([pres_bins[:, 1], bg_week])
    tr_sp = np.concatenate([pres_bins[:, 2], bg_sp])
    y = np.concatenate([np.ones(len(pres_bins)), np.zeros(len(bg_cell))]).astype(np.float32)
    print(f"[train] {len(pres_bins):,} presence + {len(bg_cell):,} background bins; encoding z_dynamic...", flush=True)
    t0 = time.time()
    tr_z = zdyn(tr_cell, tr_week, tr_sp)
    print(f"  encoded ({time.time()-t0:.0f}s)", flush=True)
    zmean = tr_z.mean(0); zstd = tr_z.std(0) + 1e-6
    co = coord_enc(lat_c[tr_cell], lon_c[tr_cell]); wk = fourier_week(tr_week)
    spidx = np.array([sid2idx[int(s)] for s in tr_sp])

    net = SINR(n_sp).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    lf = nn.BCEWithLogitsLoss()
    T = lambda a: torch.from_numpy(a).to(device)
    ct, wt, zt, st, yt = T(co), T(wk), T((tr_z - zmean) / zstd), T(spidx).long(), T(y)
    n, bs = len(yt), 16384
    for ep in range(args.epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            lf(net(ct[b], wt[b], zt[b], st[b]), yt[b]).backward()
            opt.step()
    net.eval()
    print("[train] done", flush=True)

    weeks_all = np.repeat(np.arange(52), len(cov_cells))
    cells_all = np.tile(cov_cells, 52)
    valid = np.array([(int(c), int(w)) in win_by_key for c, w in zip(cells_all, weeks_all)])
    cells_all, weeks_all = cells_all[valid], weeks_all[valid]
    doy_all = (weeks_all * 7 + 4).clip(1, 365)
    co_all = coord_enc(lat_c[cells_all], lon_c[cells_all]); wk_all = fourier_week(weeks_all)
    print(f"[predict] {len(cells_all):,} (cell,week) x {n_sp:,} species", flush=True)
    t0 = time.time()
    for si, sid in enumerate(species_ids):
        fp = out_dir / f"part_{si:05d}.parquet"
        if fp.exists():
            continue
        zd = zdyn(cells_all, weeks_all, np.full(len(cells_all), int(sid), np.int64))
        with torch.no_grad():
            score = torch.sigmoid(net(T(co_all), T(wk_all), T((zd - zmean) / zstd),
                                      torch.full((len(cells_all),), sid2idx[int(sid)], dtype=torch.long, device=device))).cpu().numpy()
        norm = score / np.maximum(pd.Series(score).groupby(cells_all).transform("sum").values, 1e-9)
        pd.DataFrame({"species": id2name[int(sid)], "species_id": int(sid),
                      "cell_idx": cells_all.astype(np.int32), "centroid_lat": lat_c[cells_all].astype(np.float32),
                      "centroid_lon": lon_c[cells_all].astype(np.float32), "week": weeks_all.astype(np.int8),
                      "doy": doy_all.astype(np.int16), "p_flowering": score.astype(np.float32),
                      "norm": norm.astype(np.float32)}).to_parquet(fp, index=False)
        if si % 25 == 0:
            rate = (si + 1) / (time.time() - t0 + 1e-9)
            print(f"  species {si+1}/{n_sp} ({rate*3600:.0f}/h, eta {(n_sp-si-1)/max(rate,1e-9)/3600:.1f}h)", flush=True)
    print(f"[done] wrote {n_sp:,} species parquets to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
