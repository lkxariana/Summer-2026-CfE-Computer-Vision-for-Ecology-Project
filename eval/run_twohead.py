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
from antheia.twohead import make_scorers, plant_context, train_twohead
from antheia.twotower import Data

VARIANTS = [
    ("joint",        dict(lam_rank=1.0, lam_comp=1.0, delta_mode="both")),
    ("rank_only",    dict(lam_rank=1.0, lam_comp=0.0, delta_mode="both")),
    ("comp_only",    dict(lam_rank=0.0, lam_comp=1.0, delta_mode="both")),
    ("joint_noDelta", dict(lam_rank=1.0, lam_comp=1.0, delta_mode="none")),
    ("joint_globalDelta", dict(lam_rank=1.0, lam_comp=1.0, delta_mode="global")),
    ("joint_localDelta", dict(lam_rank=1.0, lam_comp=1.0, delta_mode="local")),
]


def main():
    ap = argparse.ArgumentParser(description="Two-head model: retrieval + compatibility on shared towers")
    ap.add_argument("--config", default=None)
    ap.add_argument("--emb", default="bioclip", choices=["none", "bioclip", "bioclip2"])
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = load_config(args.config)
    store = FeatureStore(cfg)
    edges = pd.read_parquet(cfg["edges"])
    prov = pd.read_csv(cfg["edges"].parent / "edge_provenance_v1.csv")
    edges = edges.merge(prov, on=["plant", "pollinator"], how="left")
    edges["tier1"] = edges["n_inat"].notna() & (edges["n_inat"] < edges["n"])
    known_pos = set(zip(edges["plant"], edges["pollinator"]))
    split = load_split(cfg["edges"].parent / "split_v1.json")
    seed, nboot = args.seed, cfg["eval"]["bootstrap_n"]
    ratio, mode = cfg["negatives"]["ratio"], cfg["negatives"]["mode"]

    emb = None
    if args.emb != "none":
        pre = {"bioclip": "bioclip_text", "bioclip2": "bioclip2_text"}[args.emb]
        emb = (np.load(cfg["cache_dir"] / f"{pre}_plants.npy"),
               np.load(cfg["cache_dir"] / f"{pre}_polls.npy"))

    t0 = time.time()
    data = Data(store, edges, split, cfg, emb=emb, seed=seed)
    ctx = plant_context(store)
    print(f"data ready ({time.time() - t0:.0f}s)", flush=True)

    def partners(plants, df):
        sub = df[df["plant"].isin(plants)]
        return {sp: set(store.idx_polls(g["pollinator"])) for sp, g in sub.groupby("plant")}

    val_part = partners(split["val"], edges)
    test_part = partners(split["test"], edges)
    t1_part = partners(split["test"], edges[edges["tier1"]])
    val_wide = {sp: data.wide_plant(store.p2i[sp]) for sp in val_part}
    test_wide = {sp: data.wide_plant(store.p2i[sp]) for sp in test_part}
    print(f"wide caches ready ({time.time() - t0:.0f}s)", flush=True)

    def pooled_set(pos_plants, rng_seed):
        pos = edges[edges["plant"].isin(pos_plants)]
        neg = sample_negatives(pos, sorted(pos_plants), store, ratio, mode, rng_seed, known_pos)
        pr = pd.concat([pos[["plant", "pollinator"]].assign(label=1),
                        neg.assign(label=0)], ignore_index=True)
        pi = store.idx_plants(pr["plant"])
        qi = store.idx_polls(pr["pollinator"])
        return pr["label"].values, pi, qi, data._wide(pi, qi)

    y_val, vpi, vqi, vwide = pooled_set(split["val"], seed + 7)
    y_te, tpi, tqi, twide = pooled_set(split["test"], seed + 1)
    print(f"pooled sets ready: val {len(y_val):,} test {len(y_te):,} ({time.time() - t0:.0f}s)", flush=True)

    import torch

    def pair_scores(m, pi, qi, wide, dmode):
        from antheia.twohead import mask_delta
        m.eval()
        outs_r, outs_c = [], []
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
                c = torch.from_numpy(ctx[pi[sl]]).to(args.device)
                outs_r.append(m.score_r(pv, qv, w).cpu().numpy())
                outs_c.append(m.score_c(pv, qv, w, c).cpu().numpy())
        return np.concatenate(outs_r), np.concatenate(outs_c)

    def rank_metrics_for(scorer, part, wide_cache, ks=(10, 50)):
        out = {f"r{k}": [] for k in ks}
        out["hit10"] = []
        for sp, tp in part.items():
            s = scorer(store.p2i[sp], wide_cache[sp])
            top = np.argpartition(-s, max(ks))[:max(ks)]
            top = top[np.argsort(-s[top])]
            for k in ks:
                out[f"r{k}"].append(len(tp & set(top[:k].tolist())) / len(tp))
            out["hit10"].append(float(len(tp & set(top[:10].tolist())) > 0))
        return {k: np.array(v) for k, v in out.items()}

    def t1_recall(scorer, sp):
        """recall@10 against curated partners only, with non-curated partners masked out of the ranking."""
        s_vec = scorer(store.p2i[sp], test_wide[sp]).copy()
        others = list(test_part[sp] - t1_part[sp])
        if others:
            s_vec[others] = -np.inf
        top = set(np.argpartition(-s_vec, 10)[:10].tolist())
        return len(t1_part[sp] & top) / len(t1_part[sp])

    rows = []
    for name, kw in VARIANTS:
        dmode = kw["delta_mode"]
        tv = time.time()

        def val_eval(m, dm=dmode):
            rs, cs = make_scorers(m, data, ctx, args.device, dm)
            r = rank_metrics_for(rs, val_part, val_wide)["r10"].mean()
            _, pc = pair_scores(m, vpi, vqi, vwide, dm)
            pr = pair_metrics(y_val, pc)["pr_auc"]
            return r, f"val R@10 {r:.4f} pooledPR {pr:.4f}"

        m, _ = train_twohead(data, ctx, args.device, seed=seed, val_eval=val_eval,
                             log=lambda s: print(f"  [{name}] {s}", flush=True), **kw)
        rs, cs = make_scorers(m, data, ctx, args.device, dmode)
        pr_r, pr_c = pair_scores(m, tpi, tqi, twide, dmode)

        for head, scorer, pooled in (("retrieval", rs, pr_r), ("compatibility", cs, pr_c)):
            rm = rank_metrics_for(scorer, test_part, test_wide)
            r10, lo, hi, _ = bootstrap_mean(rm["r10"], nboot, seed)
            t1 = np.array([t1_recall(scorer, sp) for sp in test_part
                           if t1_part.get(sp)])
            rows.append({"variant": name, "head": head, "recall@10": r10, "lo": lo, "hi": hi,
                         "recall@50": rm["r50"].mean(), "hit@10": rm["hit10"].mean(),
                         "pooled_PR": pair_metrics(y_te, pooled)["pr_auc"],
                         "t1_recall@10": t1.mean()})
            print(f"  {name:<18} {head:<14} R@10 {r10:.4f} [{lo:.4f},{hi:.4f}] "
                  f"pooledPR {rows[-1]['pooled_PR']:.4f} T1 {t1.mean():.4f}", flush=True)
        print(f"  ({name} done in {time.time() - tv:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(cfg["edges"].parent / "twohead_v1.csv", index=False)
    print("\n" + df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
