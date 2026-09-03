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
from pipelines.common import fourier_loc, fourier_week
from pipelines.config import load_config, resolve
from pipelines.sdm.climate import gather_cells, gather_obs

CAP = 1000
POS_WEIGHT = 2048.0
THR = 0.30
CAPCELLS = 1500
CORE_ORDERS = {"Hymenoptera", "Lepidoptera", "Diptera", "Coleoptera", "Hemiptera", "Apodiformes"}


class ResLayer(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.w1 = nn.Linear(h, h); self.w2 = nn.Linear(h, h)
        self.a = nn.ReLU(inplace=True); self.d = nn.Dropout(0.5)

    def forward(self, x):
        return x + self.a(self.w2(self.d(self.a(self.w1(x)))))


class SINR(nn.Module):
    def __init__(self, in_dim, nsp, h=256, depth=4):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(in_dim, h), nn.ReLU(inplace=True), *[ResLayer(h) for _ in range(depth)])
        self.cls = nn.Linear(h, nsp, bias=False)

    def emb(self, x):
        return self.enc(x)


class LESINR(nn.Module):
    def __init__(self, pos_in, txt_in=768, d=256, depth=4):
        super().__init__()
        self.spec = nn.Sequential(nn.Linear(txt_in, d), nn.ReLU(inplace=True), nn.Linear(d, d))
        self.pos = nn.Sequential(nn.Linear(pos_in, d), nn.ReLU(inplace=True), *[ResLayer(d) for _ in range(depth)])

    def species_emb(self, t):
        return self.spec(t)

    def pos_emb(self, x):
        return self.pos(x)


def an_full_loss(logits, sid):
    p = torch.sigmoid(logits)
    loss = -torch.log(1 - p + 1e-5)
    idx = torch.arange(len(sid), device=sid.device)
    loss[idx, sid] = POS_WEIGHT * -torch.log(p[idx, sid] + 1e-5)
    return loss.mean()


def fit(net, logits_fn, X, sid, epochs, tag, dev):
    opt = torch.optim.Adam(net.parameters(), 5e-4, weight_decay=1e-5)
    for ep in range(epochs):
        t0 = time.time()
        perm = torch.randperm(len(sid), device=dev)
        for i in range(0, len(sid), 2048):
            b = perm[i:i + 2048]
            opt.zero_grad()
            an_full_loss(logits_fn(X[b]), sid[b]).backward()
            opt.step()
        print(f"    {tag} ep{ep} {time.time()-t0:.0f}s", flush=True)
    return net.eval()


def main():
    """Pollinator activity-curve deliverable, trained on all occurrences: per-species-head SINR
    where occurrences exist, zero-shot LE-SINR (BioCLIP-2 text) for GloBI-linked zero-obs
    species. Curves on the plant grid, range-masked (annual mean >= 30% of peak cell, <= 1,500
    cells), activity_norm summing to 1 per (species, cell)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--occ", default=None, help="occurrence npz (default <sdm_data>/pollinator_occ.npz)")
    ap.add_argument("--occ-text", default=None, help="text pt aligned to the occ npz sidx")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--universe", default=None, help="modelled_universe.json; restricts species to its pollinators[]")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    dev = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    epochs = 1 if args.smoke else args.epochs
    out = Path(args.out_dir) if args.out_dir else cfg["paths"]["sdm_data"] / "deliverable_v2"
    if args.smoke:
        out = out / "smoke"
    out.mkdir(parents=True, exist_ok=True)
    print(f"[device] {dev} epochs={epochs} out={out}", flush=True)

    sd = cfg["paths"]["sdm_data"]
    z = np.load(args.occ or sd / "pollinator_occ.npz")
    lat, lon, doy = z["lat"], z["lon"], z["doy"].astype(int)
    sid = z["sidx"].astype(int)
    occ_names = z["names"] if "names" in z.files else None
    taxon_ids = z["taxon_ids"] if occ_names is None else None
    week = ((doy - 1) // 7).clip(0, 51)
    nsp = int(sid.max()) + 1
    counts = np.bincount(sid, minlength=nsp)
    rng = np.random.default_rng(0)

    ann, mon = gather_obs(cfg, lat, lon, doy)
    am, asd = ann.mean(0), ann.std(0) + 1e-6
    X = np.concatenate([fourier_loc(lat, lon), (mon - am) / asd, fourier_week(week)], 1).astype(np.float32)
    tr_idx = np.concatenate([si if len(si) <= CAP else rng.choice(si, CAP, replace=False)
                             for si in (np.where(sid == s)[0] for s in np.unique(sid))])
    Xt = torch.from_numpy(np.ascontiguousarray(X[tr_idx])).float().to(dev)
    st = torch.from_numpy(sid[tr_idx]).long().to(dev)
    print(f"[train] head + LE-SINR on {len(tr_idx):,} obs", flush=True)
    torch.manual_seed(0)
    head = SINR(X.shape[1], nsp).to(dev)
    fit(head, lambda x: head.cls(head.emb(x)), Xt, st, epochs, "head", dev)
    torch.save(head.state_dict(), out / "model_head.pt")
    np.savez(out / "model_standardization.npz", clim_annual_mean=am, clim_annual_std=asd)
    txt_occ = torch.load(args.occ_text or sd / "pollinator_species_text.pt",
                         map_location="cpu")["embeddings"].float().to(dev)
    torch.manual_seed(0)
    L = LESINR(X.shape[1]).to(dev)
    fit(L, lambda x: L.pos_emb(x) @ L.species_emb(txt_occ).T, Xt, st, epochs, "lesinr", dev)
    torch.save(L.state_dict(), out / "model_lesinr.pt")

    fc = pd.read_parquet(resolve(cfg, "flowering_curves"),
                         columns=["cell_idx", "centroid_lat", "centroid_lon"]).drop_duplicates("cell_idx").sort_values("cell_idx")
    gc_idx = fc["cell_idx"].to_numpy(); glat = fc["centroid_lat"].to_numpy(); glon = fc["centroid_lon"].to_numpy()
    nc = len(gc_idx)
    clim_cell = gather_cells(cfg, glat, glon)
    wk = np.arange(52)
    wmonth = np.clip((wk * 7 + 4 - 1) // 30, 0, 11)
    cr = np.repeat(np.arange(nc), 52); wr = np.tile(wk, nc)
    Xg = np.concatenate([fourier_loc(glat, glon)[cr], ((clim_cell[cr, wmonth[wr]] - am) / asd).astype(np.float32),
                         fourier_week(wk)[wr]], 1).astype(np.float32)
    Xg = torch.from_numpy(Xg).float().to(dev)
    with torch.no_grad():
        Gg = head.emb(Xg).detach(); Wh = head.cls.weight.detach(); Gl = L.pos_emb(Xg).detach()
    print(f"[grid] {nc} cells", flush=True)

    plants = set(pd.read_parquet(resolve(cfg, "flowering_curves"), columns=["species"])["species"].unique())
    g = pd.read_csv(resolve(cfg, "globi"), usecols=["sourceTaxonName", "sourceTaxonOrderName", "targetTaxonName"])
    linked = set(g[g.sourceTaxonOrderName.isin(CORE_ORDERS) & g.targetTaxonName.isin(plants)]["sourceTaxonName"].dropna().unique())
    if args.universe:
        u = json.load(open(args.universe))
        linked &= {p["label"] if isinstance(p, dict) else p for p in u["pollinators"]}
    A = []
    if occ_names is not None:
        for s in range(nsp):
            name = str(occ_names[s])
            if name in linked:
                A.append((s, -1, name, int(counts[s])))
    else:
        lock = pd.read_csv(resolve(cfg, "locked_species"))
        tid2globi = dict(zip(lock["train_taxon_id"].astype(int), lock["globi_name"]))
        id2name = {int(m["taxon_id"]): (m.get("latin_name") or "") for m in json.load(open(resolve(cfg, "geo_prior_meta")))}
        for s in range(nsp):
            tid = int(taxon_ids[s])
            gname = tid2globi.get(tid, id2name.get(tid, ""))
            if gname in linked:
                A.append((s, tid, id2name.get(tid, gname) or gname, int(counts[s])))
    zt = torch.load(sd / "zeroshot_text.pt", map_location="cpu")
    txt_B = zt["embeddings"].float().to(dev); namesB = zt["names"]
    if args.universe:
        keep = [i for i, n in enumerate(namesB) if n in linked]
        txt_B = txt_B[keep]; namesB = [namesB[i] for i in keep]
    if args.smoke:
        A = A[:20]; txt_B = txt_B[:20]; namesB = namesB[:20]
    with torch.no_grad():
        Fb = L.species_emb(txt_B).detach()
    print(f"[species] Group A (head)={len(A)}  Group B (lesinr_zeroshot)={len(namesB)}", flush=True)

    cols = {k: [] for k in ["species_id", "cell_idx", "centroid_lat", "centroid_lon", "week", "doy", "p_activity", "activity_norm"]}
    wk_doy = (np.arange(52) * 7 + 4).astype(np.int16)

    def emit(P, sp_ids):
        annualP = P.mean(1)
        for j in range(P.shape[2]):
            apc = annualP[:, j]; mx = apc.max()
            keep = (np.argsort(-apc)[:1] if mx <= 0 else
                    np.where(apc >= THR * mx)[0][np.argsort(-apc[apc >= THR * mx])][:CAPCELLS])
            K = len(keep)
            blk = P[keep, :, j]
            ssum = blk.sum(1, keepdims=True)
            norm = np.where(ssum > 0, blk / ssum, 1 / 52)
            cols["species_id"].append(np.full(K * 52, sp_ids[j], np.int32))
            cols["cell_idx"].append(np.repeat(gc_idx[keep], 52).astype(np.int32))
            cols["centroid_lat"].append(np.repeat(glat[keep], 52).astype(np.float32))
            cols["centroid_lon"].append(np.repeat(glon[keep], 52).astype(np.float32))
            cols["week"].append(np.tile(np.arange(52, dtype=np.int16), K))
            cols["doy"].append(np.tile(wk_doy, K))
            cols["p_activity"].append(blk.reshape(-1).astype(np.float32))
            cols["activity_norm"].append(norm.reshape(-1).astype(np.float32))

    sidsA = np.array([s for s, _, _, _ in A]); idA = np.arange(len(A))
    for k0 in range(0, len(sidsA), 200):
        ch = sidsA[k0:k0 + 200]
        with torch.no_grad():
            P = torch.sigmoid(Gg @ Wh[torch.from_numpy(ch).to(dev)].T).cpu().numpy().reshape(nc, 52, len(ch))
        emit(P, idA[k0:k0 + 200])
        print(f"  A curves {k0+len(ch)}/{len(sidsA)}", flush=True)
    idB = np.arange(len(A), len(A) + len(namesB))
    for k0 in range(0, len(namesB), 200):
        with torch.no_grad():
            P = torch.sigmoid(Gl @ Fb[k0:k0 + 200].T).cpu().numpy().reshape(nc, 52, min(200, len(namesB) - k0))
        emit(P, idB[k0:k0 + 200])
        print(f"  B curves {k0+P.shape[2]}/{len(namesB)}", flush=True)

    df = pd.DataFrame({k: np.concatenate(v) for k, v in cols.items()})
    df.to_parquet(out / "pollinator_activity_curves.parquet", index=False, compression="zstd")
    print(f"[write] {out/'pollinator_activity_curves.parquet'} rows={len(df):,} "
          f"species={df.species_id.nunique()} cells={df.cell_idx.nunique()}", flush=True)

    conf = lambda n: "high" if n >= 100 else ("med" if n >= 20 else "low")
    man = ([(i, name, tid, nob, "head", conf(nob)) for i, (_, tid, name, nob) in zip(idA, A)] +
           [(i, name, -1, 0, "lesinr_zeroshot", "zeroshot") for i, name in zip(idB, namesB)])
    pd.DataFrame(man, columns=["species_id", "species", "taxon_id", "n_obs", "model_source", "confidence"]).to_csv(
        out / "species_manifest.csv", index=False)
    print(f"[manifest] {len(man)} species (A={len(A)} head, B={len(namesB)} zeroshot)", flush=True)


if __name__ == "__main__":
    main()
