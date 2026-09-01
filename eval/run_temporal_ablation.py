import argparse
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from antheia.config import load_config
from antheia.metrics import bootstrap_mean, pair_metrics
from antheia.pairs import load_split, sample_negatives
from antheia.store import FeatureStore
from antheia.twohead import make_scorers, mask_delta, plant_context, train_twohead
from antheia.twotower import Data

# The 2x2 that actually isolates temporal information: curves in the towers x overlap scalars in wide.
GRID = [
    ("curves+delta", True, "both"),
    ("curves_only", True, "none"),
    ("delta_only", False, "both"),
    ("no_temporal", False, "none"),
]


def main():
    ap = argparse.ArgumentParser(description="True temporal ablation: 52-week curves AND overlap scalars")
    ap.add_argument("--config", default=None)
    ap.add_argument("--emb", default="bioclip", choices=["none", "bioclip", "bioclip2", "bioclip2img"])
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--seeds", default="42,0,1")
    ap.add_argument("--tag", default="temporal_ablation_v1")
    args = ap.parse_args()

    cfg = load_config(args.config)
    store = FeatureStore(cfg)
    edges = pd.read_parquet(cfg["edges"])
    prov = pd.read_csv(cfg["edges"].parent / "edge_provenance_v1.csv")
    edges = edges.merge(prov, on=["plant", "pollinator"], how="left")
    edges["tier1"] = edges["n_inat"].notna() & (edges["n_inat"] < edges["n"])
    known_pos = set(zip(edges["plant"], edges["pollinator"]))
    split = load_split(cfg["edges"].parent / "split_v1.json")
    seeds = [int(x) for x in args.seeds.split(",")]
    seed, nboot = seeds[0], cfg["eval"]["bootstrap_n"]
    ratio, mode = cfg["negatives"]["ratio"], cfg["negatives"]["mode"]

    emb = None
    if args.emb != "none":
        pre = {"bioclip": "bioclip_text", "bioclip2": "bioclip2_text", "bioclip2img": "bioclip2_img"}[args.emb]
        emb = (np.load(cfg["cache_dir"] / f"{pre}_plants.npy"),
               np.load(cfg["cache_dir"] / f"{pre}_polls.npy"))

    t0 = time.time()
    data = Data(store, edges, split, cfg, emb=emb, seed=seed)
    ctx = plant_context(store)
    # Tower inputs are [curve(52) | log range(1) | emb...]; dropping the first 52 removes phenology.
    p_full, q_full = data.p_dense, data.q_dense
    p_nocurve, q_nocurve = p_full[:, 52:].copy(), q_full[:, 52:].copy()
    print(f"data ready ({time.time() - t0:.0f}s); tower dims with curves {p_full.shape[1]}/{q_full.shape[1]}, "
          f"without {p_nocurve.shape[1]}/{q_nocurve.shape[1]}", flush=True)

    def partners(plants, df):
        sub = df[df["plant"].isin(plants)]
        return {sp: set(store.idx_polls(g["pollinator"])) for sp, g in sub.groupby("plant")}

    val_part, test_part = partners(split["val"], edges), partners(split["test"], edges)
    t1_part = partners(split["test"], edges[edges["tier1"]])
    val_wide = {sp: data.wide_plant(store.p2i[sp]) for sp in val_part}
    test_wide = {sp: data.wide_plant(store.p2i[sp]) for sp in test_part}
    print(f"wide caches ready ({time.time() - t0:.0f}s)", flush=True)

    pos_te = edges[edges["plant"].isin(split["test"])]
    neg_te = sample_negatives(pos_te, sorted(split["test"]), store, ratio, mode, seed + 1, known_pos)
    te = pd.concat([pos_te[["plant", "pollinator"]].assign(label=1),
                    neg_te.assign(label=0)], ignore_index=True)
    tpi, tqi = store.idx_plants(te["plant"]), store.idx_polls(te["pollinator"])
    twide, y_te = data._wide(tpi, tqi), te["label"].values

    pos_va = edges[edges["plant"].isin(split["val"])]
    neg_va = sample_negatives(pos_va, sorted(split["val"]), store, ratio, mode, seed + 7, known_pos)
    va = pd.concat([pos_va[["plant", "pollinator"]].assign(label=1),
                    neg_va.assign(label=0)], ignore_index=True)
    vpi, vqi = store.idx_plants(va["plant"]), store.idx_polls(va["pollinator"])
    vwide, y_va = data._wide(vpi, vqi), va["label"].values

    import torch

    def pair_scores(m, pi, qi, wide, dmode):
        m.eval()
        r, c = [], []
        with torch.no_grad():
            for s in range(0, len(pi), 8192):
                sl = slice(s, s + 8192)
                pv = m.pt(torch.from_numpy(data.p_dense[pi[sl]]).to(args.device),
                          torch.from_numpy(data.p_gi[pi[sl]]).to(args.device),
                          torch.from_numpy(data.p_fi[pi[sl]]).to(args.device))
                qv = m.qt(torch.from_numpy(data.q_dense[qi[sl]]).to(args.device),
                          torch.from_numpy(data.q_gi[qi[sl]]).to(args.device),
                          torch.from_numpy(data.q_fi[qi[sl]]).to(args.device))
                w = mask_delta(torch.from_numpy(wide[sl]).to(args.device), dmode)
                cc = torch.from_numpy(ctx[pi[sl]]).to(args.device)
                r.append(m.score_r(pv, qv, w).cpu().numpy())
                c.append(m.score_c(pv, qv, w, cc).cpu().numpy())
        return np.concatenate(r), np.concatenate(c)

    def rank_eval(scorer, part, cache):
        r10, r50, hit = [], [], []
        for sp, tp in part.items():
            s = scorer(store.p2i[sp], cache[sp])
            top = np.argpartition(-s, 50)[:50]
            top = top[np.argsort(-s[top])]
            r10.append(len(tp & set(top[:10].tolist())) / len(tp))
            r50.append(len(tp & set(top.tolist())) / len(tp))
            hit.append(float(len(tp & set(top[:10].tolist())) > 0))
        return np.array(r10), np.mean(r50), np.mean(hit)

    rows = []
    per_plant = {}
    plant_order = list(test_part)
    for name, use_curves, dmode in GRID:
        data.p_dense = p_full if use_curves else p_nocurve
        data.q_dense = q_full if use_curves else q_nocurve
        tv = time.time()

        def val_eval(m, dm=dmode):
            rs, _ = make_scorers(m, data, ctx, args.device, dm)
            r = rank_eval(rs, val_part, val_wide)[0].mean()
            _, pc = pair_scores(m, vpi, vqi, vwide, dm)
            return r, f"val R@10 {r:.4f} pooledPR {pair_metrics(y_va, pc)['pr_auc']:.4f}"

        seed_r10, seed_stats = {h: [] for h in ("retrieval", "compatibility")}, []
        for sd in seeds:
            m, _ = train_twohead(data, ctx, args.device, seed=sd, delta_mode=dmode,
                                 val_eval=val_eval, log=lambda s: None)
            rs, cs = make_scorers(m, data, ctx, args.device, dmode)
            pr_r, pr_c = pair_scores(m, tpi, tqi, twide, dmode)
            seed_stats.append((rs, cs, pr_r, pr_c))
            for h, sc in (("retrieval", rs), ("compatibility", cs)):
                seed_r10[h].append(np.array([
                    len(test_part[sp] & set(np.argpartition(-sc(store.p2i[sp], test_wide[sp]), 10)[:10].tolist()))
                    / len(test_part[sp]) for sp in plant_order]))
            print(f"  [{name}] seed {sd}: R@10 {seed_r10['retrieval'][-1].mean():.4f}", flush=True)
        for h in seed_r10:
            per_plant[(name, h)] = np.mean(seed_r10[h], axis=0)
        rs, cs, pr_r, pr_c = seed_stats[0]
        for head, scorer, pooled in (("retrieval", rs, pr_r), ("compatibility", cs, pr_c)):
            _, r50, hit = rank_eval(scorer, test_part, test_wide)
            r10 = per_plant[(name, head)]
            mm, lo, hi, _ = bootstrap_mean(r10, nboot, seed)
            t1 = []
            for sp, tp in t1_part.items():
                if not tp:
                    continue
                s = scorer(store.p2i[sp], test_wide[sp]).copy()
                others = list(test_part[sp] - tp)
                if others:
                    s[others] = -np.inf
                t1.append(len(tp & set(np.argpartition(-s, 10)[:10].tolist())) / len(tp))
            rows.append({"variant": name, "curves": use_curves, "delta": dmode, "head": head,
                         "recall@10": mm, "lo": lo, "hi": hi, "recall@50": r50, "hit@10": hit,
                         "pooled_PR": pair_metrics(y_te, pooled)["pr_auc"], "t1_recall@10": np.mean(t1)})
            print(f"  {name:<14} {head:<14} R@10 {mm:.4f} [{lo:.4f},{hi:.4f}] "
                  f"pooledPR {rows[-1]['pooled_PR']:.4f} T1 {np.mean(t1):.4f}", flush=True)
        print(f"  ({name} in {time.time() - tv:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(cfg["edges"].parent / f"{args.tag}.csv", index=False)
    print("\n" + df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Paired bootstrap over plants (seed-averaged per-plant recall).
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(plant_order), size=(10000, len(plant_order)))
    print("\npaired bootstrap over test plants (seed-averaged):")
    for h in ("retrieval", "compatibility"):
        for a, b in (("curves_only", "no_temporal"), ("curves+delta", "curves_only"),
                     ("curves+delta", "no_temporal"), ("delta_only", "no_temporal")):
            d = per_plant[(a, h)] - per_plant[(b, h)]
            bs = d[idx].mean(1)
            pv = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
            print(f"  {h:<14} {a:<13} - {b:<13} = {d.mean():+.4f} "
                  f"[{np.percentile(bs, 2.5):+.4f},{np.percentile(bs, 97.5):+.4f}] p={max(pv, 1e-4):.4f}")
    np.savez(cfg["edges"].parent / f"{args.tag}_perplant.npz",
             plants=np.array(plant_order), **{f"{k[0]}|{k[1]}": v for k, v in per_plant.items()})


if __name__ == "__main__":
    main()
