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
from antheia.twohead import plant_context
from antheia.twotower import Data

SUMMARY_NAMES = ["sum_min", "cosine", "bhattacharyya", "pearson", "joint_weeks", "peak_offset", "js_sim"]


def overlap_summaries(Fc, Ac):
    """Seven scalar summaries of the overlap between normalized 52-week curves, row-wise."""
    f = Fc / np.clip(Fc.sum(1, keepdims=True), 1e-12, None)
    a = Ac / np.clip(Ac.sum(1, keepdims=True), 1e-12, None)
    out = [np.minimum(f, a).sum(1)]                                            # coefficient of overlapping
    out.append((f * a).sum(1) / np.clip(np.linalg.norm(f, axis=1) * np.linalg.norm(a, axis=1), 1e-12, None))
    out.append(np.sqrt(np.clip(f * a, 0, None)).sum(1))                        # Bhattacharyya
    fz, az = f - f.mean(1, keepdims=True), a - a.mean(1, keepdims=True)
    out.append((fz * az).sum(1) / np.clip(np.linalg.norm(fz, axis=1) * np.linalg.norm(az, axis=1), 1e-12, None))
    thr = 1.0 / f.shape[1]
    out.append(((f > thr) & (a > thr)).sum(1) / f.shape[1])                    # weeks both active
    d = np.abs(f.argmax(1) - a.argmax(1))
    out.append(np.minimum(d, f.shape[1] - d) / (f.shape[1] / 2))               # circular peak offset
    m = 0.5 * (f + a)
    kl = lambda x, y: (x * np.log(np.clip(x, 1e-12, None) / np.clip(y, 1e-12, None))).sum(1)
    out.append(1.0 - np.clip(0.5 * kl(f, m) + 0.5 * kl(a, m), 0, np.log(2)) / np.log(2))
    return np.stack(out, 1).astype(np.float32)


class SummaryData(Data):
    """Data with a configurable wide block: [log N, tax_genus, tax_family] (+ optional overlap summaries)."""

    def __init__(self, store, edges, split, cfg, use_summaries, **kw):
        self.use_summaries = use_summaries
        super().__init__(store, edges, split, cfg, **kw)
        self.wide_dim = 3 + (len(SUMMARY_NAMES) if use_summaries else 0)
        flat_pi = np.repeat(self.pos_pi, self.pools.shape[1])
        self.pos_wide = self._wide2(self.pos_pi, self.pos_qi)
        self.pool_wide = self._wide2(flat_pi, self.pools.ravel()).reshape(
            len(self.pos_pi), self.pools.shape[1], self.wide_dim)

    def _wide2(self, pi, qi):
        st = self.store
        n = np.log1p(st.N_full[pi, qi].astype(np.float32))[:, None]
        t = self.aff.pairs(pi, qi).astype(np.float32)[:, [1, 3]]
        blocks = [n, t]
        if self.use_summaries:
            blocks.append(overlap_summaries(st.FC[pi], st.AC[qi]))
        return np.hstack(blocks).astype(np.float32)

    def wide_plant2(self, p):
        st = self.store
        n = np.log1p(st.N_full[p].astype(np.float32))[:, None]
        t = self.aff.plant(p).astype(np.float32)[:, [1, 3]]
        blocks = [n, t]
        if self.use_summaries:
            blocks.append(overlap_summaries(np.repeat(st.FC[p][None, :], len(st.AC), 0), st.AC))
        return np.hstack(blocks).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description="Do ANY scalar overlap summaries recover what the raw curves carry?")
    ap.add_argument("--config", default=None)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--seeds", default="42,0,1")
    args = ap.parse_args()
    import torch
    import torch.nn.functional as Fn
    from antheia.twohead import CTX_DIM, TwoHead
    from antheia.twotower import Tower

    cfg = load_config(args.config)
    store = FeatureStore(cfg)
    edges = pd.read_parquet(cfg["edges"])
    known = set(zip(edges["plant"], edges["pollinator"]))
    split = load_split(cfg["edges"].parent / "split_v1.json")
    seeds = [int(x) for x in args.seeds.split(",")]
    nboot = cfg["eval"]["bootstrap_n"]
    ratio, mode = cfg["negatives"]["ratio"], cfg["negatives"]["mode"]
    emb = (np.load(cfg["cache_dir"] / "bioclip_text_plants.npy"),
           np.load(cfg["cache_dir"] / "bioclip_text_polls.npy"))
    ctx = plant_context(store)

    caches = {}
    for use_sum in (False, True):
        t0 = time.time()
        caches[use_sum] = SummaryData(store, edges, split, cfg, use_sum, emb=emb, seed=seeds[0])
        print(f"data(summaries={use_sum}) ready, wide_dim={caches[use_sum].wide_dim} ({time.time() - t0:.0f}s)", flush=True)

    def partners(pl):
        sub = edges[edges["plant"].isin(pl)]
        return {sp: set(store.idx_polls(g["pollinator"])) for sp, g in sub.groupby("plant")}

    val_part, test_part = partners(split["val"]), partners(split["test"])
    wide_cache = {u: {"val": {sp: caches[u].wide_plant2(store.p2i[sp]) for sp in val_part},
                      "test": {sp: caches[u].wide_plant2(store.p2i[sp]) for sp in test_part}}
                  for u in (False, True)}
    print("wide caches ready", flush=True)

    pos_te = edges[edges["plant"].isin(split["test"])]
    neg_te = sample_negatives(pos_te, sorted(split["test"]), store, ratio, mode, seeds[0] + 1, known)
    te = pd.concat([pos_te[["plant", "pollinator"]].assign(label=1), neg_te.assign(label=0)], ignore_index=True)
    tpi, tqi, y_te = store.idx_plants(te["plant"]), store.idx_polls(te["pollinator"]), te["label"].values

    GRID = [("no_temporal", False, False), ("summaries_only", False, True),
            ("curves_only", True, False), ("curves+summaries", True, True)]
    plant_order = list(test_part)
    rows, per_plant = [], {}

    for name, use_curves, use_sum in GRID:
        data = caches[use_sum]
        p_dense = data.p_dense if use_curves else data.p_dense[:, 52:]
        q_dense = data.q_dense if use_curves else data.q_dense[:, 52:]
        seed_vals = {h: [] for h in ("retrieval", "compatibility")}
        pooled_last = {}
        for sd in seeds:
            torch.manual_seed(sd)
            m = TwoHead(p_dense.shape[1], q_dense.shape[1], data.n_pgen, data.n_pfam,
                        data.n_qgen, data.n_qfam).to(args.device)
            m.head_r = torch.nn.Sequential(torch.nn.Linear(data.wide_dim, 64), torch.nn.ReLU(),
                                           torch.nn.Linear(64, 1)).to(args.device)
            m.head_c = torch.nn.Sequential(torch.nn.Linear(data.wide_dim + CTX_DIM, 64), torch.nn.ReLU(),
                                           torch.nn.Linear(64, 1)).to(args.device)
            opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
            P, Q = (torch.from_numpy(p_dense).to(args.device), torch.from_numpy(q_dense).to(args.device))
            CTX = torch.from_numpy(ctx).to(args.device)
            pgi, pfi = torch.from_numpy(data.p_gi).to(args.device), torch.from_numpy(data.p_fi).to(args.device)
            qgi, qfi = torch.from_numpy(data.q_gi).to(args.device), torch.from_numpy(data.q_fi).to(args.device)
            ppi, pqi = torch.from_numpy(data.pos_pi).to(args.device), torch.from_numpy(data.pos_qi).to(args.device)
            pools = torch.from_numpy(data.pools).to(args.device)
            pw = torch.from_numpy(data.pos_wide).to(args.device)
            plw = torch.from_numpy(data.pool_wide).to(args.device)
            n_pos, psz, wd = pools.shape[0], pools.shape[1], data.wide_dim
            rng = np.random.default_rng(sd)
            for ep in range(6):
                m.train()
                order = torch.from_numpy(rng.permutation(n_pos)).to(args.device)
                for s in range(0, n_pos, 512):
                    idx = order[s:s + 512]
                    cols = torch.from_numpy(rng.integers(0, psz, (len(idx), 63))).to(args.device)
                    qc = torch.cat([pqi[idx, None], torch.gather(pools[idx], 1, cols)], 1)
                    wide = torch.cat([pw[idx, None, :], torch.gather(
                        plw[idx], 1, cols[..., None].expand(-1, -1, wd))], 1)
                    pv = m.pt(P[ppi[idx]], pgi[ppi[idx]], pfi[ppi[idx]])[:, None, :]
                    qv = m.qt(Q[qc.reshape(-1)], qgi[qc.reshape(-1)], qfi[qc.reshape(-1)]).view(len(idx), 64, -1)
                    cb = CTX[ppi[idx]][:, None, :].expand(-1, 64, -1)
                    lr_ = m.score_r(pv, qv, wide)
                    lc = m.score_c(pv, qv, wide, cb)
                    lbl = torch.zeros_like(lc); lbl[:, 0] = 1.0
                    loss = Fn.cross_entropy(lr_, torch.zeros(len(idx), dtype=torch.long, device=args.device)) \
                        + Fn.binary_cross_entropy_with_logits(lc, lbl)
                    opt.zero_grad(); loss.backward(); opt.step()
            m.eval()
            with torch.no_grad():
                qv_all = torch.cat([m.qt(Q[i:i + 4096], qgi[i:i + 4096], qfi[i:i + 4096])
                                    for i in range(0, len(q_dense), 4096)])
            def score(sp, which):
                p = store.p2i[sp]
                with torch.no_grad():
                    pv = m.pt(P[p:p + 1], pgi[p:p + 1], pfi[p:p + 1])[0]
                    w = torch.from_numpy(wide_cache[use_sum][which][sp]).to(args.device)
                    inter = m.tau * (qv_all @ pv)
                    c = CTX[p:p + 1].expand(len(qv_all), -1)
                    return (inter + m.head_r(w).squeeze(-1)).cpu().numpy(), \
                           (inter + m.head_c(torch.cat([w, c], -1)).squeeze(-1)).cpu().numpy()
            for hi, h in enumerate(("retrieval", "compatibility")):
                seed_vals[h].append(np.array([
                    len(test_part[sp] & set(np.argpartition(-score(sp, "test")[hi], 10)[:10].tolist()))
                    / len(test_part[sp]) for sp in plant_order]))
            with torch.no_grad():
                sr, sc = [], []
                for s in range(0, len(tpi), 8192):
                    sl = slice(s, s + 8192)
                    pv = m.pt(P[tpi[sl]], pgi[tpi[sl]], pfi[tpi[sl]])
                    qv = m.qt(Q[tqi[sl]], qgi[tqi[sl]], qfi[tqi[sl]])
                    w = torch.from_numpy(data._wide2(tpi[sl], tqi[sl])).to(args.device)
                    sr.append(m.score_r(pv, qv, w).cpu().numpy())
                    sc.append(m.score_c(pv, qv, w, CTX[tpi[sl]]).cpu().numpy())
                pooled_last = {"retrieval": np.concatenate(sr), "compatibility": np.concatenate(sc)}
            print(f"  [{name}] seed {sd} R@10 {seed_vals['retrieval'][-1].mean():.4f}", flush=True)
        for h in seed_vals:
            per_plant[(name, h)] = np.mean(seed_vals[h], axis=0)
            mm, lo, hi, _ = bootstrap_mean(per_plant[(name, h)], nboot, seeds[0])
            rows.append({"variant": name, "head": h, "recall@10": mm, "lo": lo, "hi": hi,
                         "pooled_PR": pair_metrics(y_te, pooled_last[h])["pr_auc"]})
            print(f"  {name:<18} {h:<14} R@10 {mm:.4f} [{lo:.4f},{hi:.4f}] "
                  f"pooledPR {rows[-1]['pooled_PR']:.4f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(cfg["edges"].parent / "overlap_summaries_v1.csv", index=False)
    print("\n" + df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    rng = np.random.default_rng(seeds[0])
    idx = rng.integers(0, len(plant_order), size=(10000, len(plant_order)))
    print("\npaired bootstrap:")
    for h in ("retrieval", "compatibility"):
        for a, b in (("summaries_only", "no_temporal"), ("curves_only", "no_temporal"),
                     ("curves_only", "summaries_only"), ("curves+summaries", "curves_only")):
            d = per_plant[(a, h)] - per_plant[(b, h)]
            bs = d[idx].mean(1)
            pv = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
            print(f"  {h:<14} {a:<17} - {b:<15} = {d.mean():+.4f} "
                  f"[{np.percentile(bs, 2.5):+.4f},{np.percentile(bs, 97.5):+.4f}] p={max(pv, 1e-4):.4f}")


if __name__ == "__main__":
    main()
