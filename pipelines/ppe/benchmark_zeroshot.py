import argparse
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipelines.common import coord_enc, fourier_week, overlap
from pipelines.config import load_config
from pipelines.ppe.backbone import (alphaearth_by_cell, encode_zdyn_emb, load_e98, load_grid,
                                    load_prism_weekly, load_vocab, text_matrix)
from pipelines.ppe.generate_opportunity import NEG_RATIO, SINR, load_presence


def main():
    """Fallback-policy benchmark for uncovered plants: hold out covered species, predict their
    week-of-year curve zero-shot (text head trained without them) vs genus aggregation (mean
    curve of congeners), scored by overlap with the species' own observed flowering histogram.
    Baselines: global clock (mean histogram of training species) and the id-head-trained curve
    (--surface-dir, upper bound). Curves pooled over each species' own presence cells."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", type=int, default=50)
    ap.add_argument("--min-obs", type=int, default=50, help="held-out species need a stable observed histogram")
    ap.add_argument("--train-min-obs", type=int, default=5)
    ap.add_argument("--max-train-species", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--surface-dir", default=None, help="id-head opportunity surface for the upper-bound arm")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}", flush=True)

    grid = load_grid(cfg)
    vocab = load_vocab(cfg)
    id2name = {int(v): k for k, v in vocab.items()}
    pc, pw, psp = load_presence(cfg, grid, args.train_min_obs)
    counts = pd.Series(psp).value_counts()
    genus = {int(s): id2name[int(s)].split()[0] for s in counts.index}
    rng = np.random.default_rng(args.seed)
    eligible = np.array(sorted(s for s in counts[counts >= args.min_obs].index
                               if sum(genus[int(t)] == genus[int(s)] for t in counts.index) > 1))
    held = set(rng.choice(eligible, args.holdout, replace=False).tolist())
    train_ids = np.array(sorted(set(counts.index) - held))
    if args.max_train_species:
        train_ids = train_ids[:args.max_train_species]
    keep = set(train_ids.tolist())
    m = np.array([s in keep for s in psp])
    tc, tw, tsp = pc[m], pw[m], psp[m]
    print(f"[split] train {len(train_ids)} species / held-out {len(held)}", flush=True)

    windows, win_by_key = load_prism_weekly(cfg)
    cov_cells = np.array(sorted({c for c, _ in win_by_key}))
    lat_c = grid.centroid_lat.values; lon_c = grid.centroid_lon.values
    ae_by_cell = alphaearth_by_cell(cfg, grid)
    model = load_e98(cfg, device)
    emb_all, _ = text_matrix(cfg)

    def zdyn(cells, weeks, sids):
        prism = windows[[win_by_key[(int(c), int(w))] for c, w in zip(cells, weeks)]]
        return encode_zdyn_emb(model, emb_all[sids], prism, ae_by_cell[cells], device, args.batch)

    pres_bins = np.unique(np.stack([tc, tw, tsp], 1), axis=0)
    bg_cell = rng.choice(cov_cells, NEG_RATIO * len(pres_bins))
    bg_week = rng.integers(0, 52, NEG_RATIO * len(pres_bins))
    bg_sp = train_ids[rng.integers(0, len(train_ids), NEG_RATIO * len(pres_bins))]
    ok = np.array([(int(c), int(w)) in win_by_key for c, w in zip(bg_cell, bg_week)])
    trc = np.concatenate([pres_bins[:, 0], bg_cell[ok]])
    trw = np.concatenate([pres_bins[:, 1], bg_week[ok]])
    trs = np.concatenate([pres_bins[:, 2], bg_sp[ok]])
    y = np.concatenate([np.ones(len(pres_bins)), np.zeros(ok.sum())]).astype(np.float32)
    print(f"[train] {len(pres_bins):,} presence + {ok.sum():,} background; encoding...", flush=True)
    t0 = time.time()
    tz = zdyn(trc, trw, trs)
    print(f"  encoded ({time.time()-t0:.0f}s)", flush=True)
    zmean, zstd = tz.mean(0), tz.std(0) + 1e-6
    sid2idx = {int(s): i for i, s in enumerate(train_ids)}
    net = SINR(len(train_ids), txt=emb_all[train_ids]).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    lf = nn.BCEWithLogitsLoss()
    T = lambda a: torch.from_numpy(a).to(device)
    ct, wt = T(coord_enc(lat_c[trc], lon_c[trc])), T(fourier_week(trw))
    zt, yt = T((tz - zmean) / zstd), T(y)
    st = T(np.array([sid2idx[int(s)] for s in trs])).long()
    for ep in range(args.epochs):
        perm = torch.randperm(len(yt), device=device)
        for i in range(0, len(yt), 16384):
            b = perm[i:i + 16384]
            opt.zero_grad()
            lf(net(ct[b], wt[b], zt[b], st[b]), yt[b]).backward()
            opt.step()
    net.eval()
    print("[train] done", flush=True)

    def obs_hist(s):
        h = np.bincount(pw[psp == s], minlength=52).astype(float)
        return h / h.sum()

    clock = np.stack([obs_hist(s) for s in train_ids]).mean(0)
    clock /= clock.sum()
    surface = {}
    if args.surface_dir:
        for f in Path(args.surface_dir).glob("part_*.parquet"):
            surface[int(pd.read_parquet(f, columns=["species_id"]).species_id.iloc[0])] = f
    rows = []
    for s in sorted(held):
        truth = obs_hist(s)
        cells = np.unique(pc[psp == s])
        cells = cells[np.isin(cells, cov_cells)]
        gc = np.repeat(cells, 52); gw = np.tile(np.arange(52), len(cells))
        v = np.array([(int(c), int(w)) in win_by_key for c, w in zip(gc, gw)])
        gc, gw = gc[v], gw[v]
        zd = zdyn(gc, gw, np.full(len(gc), int(s)))
        with torch.no_grad():
            p = torch.sigmoid(net.forward_emb(T(coord_enc(lat_c[gc], lon_c[gc])), T(fourier_week(gw)),
                                              T((zd - zmean) / zstd), T(emb_all[np.full(len(gc), int(s))]))).cpu().numpy()
        zs = np.bincount(gw, weights=p, minlength=52) / np.maximum(np.bincount(gw, minlength=52), 1)
        zs /= zs.sum()
        cong = [t for t in train_ids if genus[int(t)] == genus[int(s)]]
        ga_score = np.nan
        if cong:
            ga = np.stack([obs_hist(t) for t in cong]).mean(0)
            ga_score = overlap(ga / ga.sum(), truth)
        row = {"species": id2name[int(s)], "n_obs": int(counts[s]), "n_congeners": len(cong),
               "zeroshot": overlap(zs, truth), "genus_obs": ga_score, "clock": overlap(clock, truth)}
        if int(s) in surface:
            df = pd.read_parquet(surface[int(s)])
            df = df[df.cell_idx.isin(cells)]
            ih = df.groupby("week").p_flowering.mean().reindex(range(52), fill_value=0).values
            row["id_head"] = overlap(ih / ih.sum(), truth)
        rows.append(row)
        print(f"  {row}", flush=True)
    res = pd.DataFrame(rows)
    out = Path(args.out) if args.out else cfg["paths"]["out_dir"] / "zeroshot_benchmark.csv"
    res.to_csv(out, index=False)
    print(res.drop(columns=["species"]).mean(numeric_only=True).round(3).to_string(), flush=True)
    print(f"[done] {out}", flush=True)


if __name__ == "__main__":
    main()
