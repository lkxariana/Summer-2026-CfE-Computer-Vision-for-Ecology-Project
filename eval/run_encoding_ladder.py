import argparse
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from antheia.config import load_config
from antheia.metrics import bootstrap_mean, pooled_metrics
from antheia.pairs import load_split, sample_negatives
from antheia.store import FeatureStore
from antheia.twohead import make_scorers, plant_context, train_twohead
from antheia.twotower import Data
from run_overlap_summaries import overlap_summaries

# One protocol, one wide base ([log N, tax_genus, tax_family]), one training loop.
# Arms differ ONLY in how phenology enters. This is the paper's central table.
ARMS = [
    ("none",              "base",      False, False),
    ("delta_scalar",      "delta",     False, False),
    ("summaries_7",       "summaries", False, False),
    ("curves",            "base",      True,  False),
    ("curves+delta",      "delta",     True,  False),
    ("curves+trajectory", "base",      True,  True),
]


class LadderData(Data):
    """Data with wide = [log N, tax_genus, tax_family] (+ delta | + 7 overlap summaries)."""

    def __init__(self, store, edges, split, cfg, wide_mode, **kw):
        self.wide_mode = wide_mode
        super().__init__(store, edges, split, cfg, **kw)
        self.wide_dim = 3 + {"base": 0, "delta": 1, "summaries": 7}[wide_mode]
        flat = np.repeat(self.pos_pi, self.pools.shape[1])
        self.pos_wide = self.wide_pairs(self.pos_pi, self.pos_qi)
        self.pool_wide = self.wide_pairs(flat, self.pools.ravel()).reshape(
            len(self.pos_pi), self.pools.shape[1], self.wide_dim)

    def wide_pairs(self, pi, qi):
        st = self.store
        blocks = [np.log1p(st.N_full[pi, qi].astype(np.float32))[:, None],
                  self.aff.pairs(pi, qi).astype(np.float32)[:, [1, 3]]]
        if self.wide_mode == "delta":
            blocks.append(np.minimum(st.FC[pi], st.AC[qi]).sum(1).astype(np.float32)[:, None])
        elif self.wide_mode == "summaries":
            blocks.append(overlap_summaries(st.FC[pi], st.AC[qi]))
        return np.hstack(blocks).astype(np.float32)

    def wide_for_plant(self, p):
        st = self.store
        blocks = [np.log1p(st.N_full[p].astype(np.float32))[:, None],
                  self.aff.plant(p).astype(np.float32)[:, [1, 3]]]
        if self.wide_mode == "delta":
            blocks.append(np.minimum(st.FC[p][None, :], st.AC).sum(1).astype(np.float32)[:, None])
        elif self.wide_mode == "summaries":
            blocks.append(overlap_summaries(np.repeat(st.FC[p][None, :], len(st.AC), 0), st.AC))
        return np.hstack(blocks).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description="Unified phenology-encoding ladder: one protocol, all arms")
    ap.add_argument("--config", default=None)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--seeds", default="42,0,1")
    ap.add_argument("--tag", default="encoding_ladder_v1")
    args = ap.parse_args()
    import torch
    from antheia.twohead import mask_delta

    cfg = load_config(args.config)
    store = FeatureStore(cfg)
    edges = pd.read_parquet(cfg["edges"])
    prov = pd.read_csv(cfg["edges"].parent / "edge_provenance_v1.csv")
    edges = edges.merge(prov, on=["plant", "pollinator"], how="left")
    edges["tier1"] = edges["n_inat"].notna() & (edges["n_inat"] < edges["n"])
    known = set(zip(edges["plant"], edges["pollinator"]))
    split = load_split(cfg["edges"].parent / "split_v1.json")
    seeds = [int(x) for x in args.seeds.split(",")]
    nboot = cfg["eval"]["bootstrap_n"]
    ratio, mode = cfg["negatives"]["ratio"], cfg["negatives"]["mode"]
    emb = (np.load(cfg["cache_dir"] / "bioclip_text_plants.npy"),
           np.load(cfg["cache_dir"] / "bioclip_text_polls.npy"))
    traj = np.load(cfg["cache_dir"] / "pheno_trajectory_plants.npy").astype(np.float32)
    ctx = plant_context(store)

    t0 = time.time()
    datas, wide_cache = {}, {}
    for wm in ("base", "delta", "summaries"):
        d = LadderData(store, edges, split, cfg, wm, emb=emb, seed=seeds[0])
        datas[wm] = d
        print(f"data[{wm}] wide_dim={d.wide_dim} ({time.time() - t0:.0f}s)", flush=True)

    def partners(pl, df):
        sub = df[df["plant"].isin(pl)]
        return {sp: set(store.idx_polls(g["pollinator"])) for sp, g in sub.groupby("plant")}

    val_part, test_part = partners(split["val"], edges), partners(split["test"], edges)
    t1_part = partners(split["test"], edges[edges["tier1"]])
    for wm, d in datas.items():
        wide_cache[wm] = {"val": {sp: d.wide_for_plant(store.p2i[sp]) for sp in val_part},
                          "test": {sp: d.wide_for_plant(store.p2i[sp]) for sp in test_part}}
        print(f"wide cache[{wm}] ready ({time.time() - t0:.0f}s)", flush=True)

    def pooled(pl, sd):
        pos = edges[edges["plant"].isin(pl)]
        neg = sample_negatives(pos, sorted(pl), store, ratio, mode, sd, known)
        pr = pd.concat([pos[["plant", "pollinator"]].assign(label=1),
                        neg.assign(label=0)], ignore_index=True)
        return (store.idx_plants(pr["plant"]), store.idx_polls(pr["pollinator"]), pr["label"].values)
    vpi, vqi, y_va = pooled(split["val"], seeds[0] + 7)
    tpi, tqi, y_te = pooled(split["test"], seeds[0] + 1)

    p_base = {wm: datas[wm].p_dense.copy() for wm in datas}
    q_base = {wm: datas[wm].q_dense.copy() for wm in datas}
    plant_order = list(test_part)
    rows, per_plant = [], {}

    def pair_scores(m, d, pi, qi):
        m.eval()
        r, c = [], []
        with torch.no_grad():
            for s in range(0, len(pi), 8192):
                sl = slice(s, s + 8192)
                pv = m.pt(torch.from_numpy(d.p_dense[pi[sl]]).to(args.device),
                          torch.from_numpy(d.p_gi[pi[sl]]).to(args.device),
                          torch.from_numpy(d.p_fi[pi[sl]]).to(args.device))
                qv = m.qt(torch.from_numpy(d.q_dense[qi[sl]]).to(args.device),
                          torch.from_numpy(d.q_gi[qi[sl]]).to(args.device),
                          torch.from_numpy(d.q_fi[qi[sl]]).to(args.device))
                w = torch.from_numpy(d.wide_pairs(pi[sl], qi[sl])).to(args.device)
                cc = torch.from_numpy(ctx[pi[sl]]).to(args.device)
                r.append(m.score_r(pv, qv, w).cpu().numpy())
                c.append(m.score_c(pv, qv, w, cc).cpu().numpy())
        return np.concatenate(r), np.concatenate(c)

    def rank_r10(scorer, part, cache):
        return np.array([len(part[sp] & set(np.argpartition(-scorer(store.p2i[sp], cache[sp]), 10)[:10].tolist()))
                         / len(part[sp]) for sp in (plant_order if part is test_part else part)])

    for name, wm, use_curves, use_traj in ARMS:
        d = datas[wm]
        pd_ = p_base[wm] if use_curves else p_base[wm][:, 52:]
        if use_traj:
            pd_ = np.hstack([pd_, traj])
        d.p_dense = pd_
        d.q_dense = q_base[wm] if use_curves else q_base[wm][:, 52:]
        tv = time.time()
        seed_r10 = {"retrieval": [], "compatibility": []}
        last = None
        for sd in seeds:
            def val_eval(m, dd=d, wmm=wm):
                rs, _ = make_scorers(m, dd, ctx, args.device, "both")
                r = np.mean([len(val_part[sp] & set(np.argpartition(
                    -rs(store.p2i[sp], wide_cache[wmm]["val"][sp]), 10)[:10].tolist())) / len(val_part[sp])
                    for sp in val_part])
                _, pc = pair_scores(m, dd, vpi, vqi)
                return r, f"val R@10 {r:.4f} pooledPR {pooled_metrics(y_va, pc)['pr_auc']:.4f}"
            m, _ = train_twohead(d, ctx, args.device, seed=sd, delta_mode="both",
                                 val_eval=val_eval, log=lambda s: None)
            rs, cs = make_scorers(m, d, ctx, args.device, "both")
            for h, sc in (("retrieval", rs), ("compatibility", cs)):
                seed_r10[h].append(np.array([
                    len(test_part[sp] & set(np.argpartition(-sc(store.p2i[sp], wide_cache[wm]["test"][sp]), 10)[:10].tolist()))
                    / len(test_part[sp]) for sp in plant_order]))
            last = (m, rs, cs)
            print(f"  [{name}] seed {sd} R@10 {seed_r10['retrieval'][-1].mean():.4f}", flush=True)
        m, rs, cs = last
        pr_r, pr_c = pair_scores(m, d, tpi, tqi)
        for h, sc, pooled_s in (("retrieval", rs, pr_r), ("compatibility", cs, pr_c)):
            per_plant[(name, h)] = np.mean(seed_r10[h], axis=0)
            mm, lo, hi, _ = bootstrap_mean(per_plant[(name, h)], nboot, seeds[0])
            t1 = []
            for sp, tp in t1_part.items():
                if not tp:
                    continue
                v = sc(store.p2i[sp], wide_cache[wm]["test"][sp]).copy()
                others = list(test_part[sp] - tp)
                if others:
                    v[others] = -np.inf
                t1.append(len(tp & set(np.argpartition(-v, 10)[:10].tolist())) / len(tp))
            rows.append({"arm": name, "wide": wm, "curves": use_curves, "trajectory": use_traj,
                         "head": h, "recall@10": mm, "lo": lo, "hi": hi,
                         "pooled_PR": pooled_metrics(y_te, pooled_s)["pr_auc"], "t1_recall@10": np.mean(t1)})
            print(f"  {name:<18} {h:<14} R@10 {mm:.4f} [{lo:.4f},{hi:.4f}] "
                  f"pooledPR {rows[-1]['pooled_PR']:.4f} T1 {np.mean(t1):.4f}", flush=True)
        print(f"  ({name} in {time.time() - tv:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(cfg["edges"].parent / f"{args.tag}.csv", index=False)
    print("\n" + df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    rng = np.random.default_rng(seeds[0])
    idx = rng.integers(0, len(plant_order), size=(10000, len(plant_order)))
    print("\npaired bootstrap vs 'none' (seed-averaged per-plant recall@10):")
    for h in ("retrieval", "compatibility"):
        for a in [x[0] for x in ARMS if x[0] != "none"]:
            dd = per_plant[(a, h)] - per_plant[("none", h)]
            bs = dd[idx].mean(1)
            pv = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
            print(f"  {h:<14} {a:<18} = {dd.mean():+.4f} "
                  f"[{np.percentile(bs, 2.5):+.4f},{np.percentile(bs, 97.5):+.4f}] p={max(pv, 1e-4):.4f}")
    np.savez(cfg["edges"].parent / f"{args.tag}_perplant.npz",
             plants=np.array(plant_order), **{f"{k[0]}|{k[1]}": v for k, v in per_plant.items()})


if __name__ == "__main__":
    main()
