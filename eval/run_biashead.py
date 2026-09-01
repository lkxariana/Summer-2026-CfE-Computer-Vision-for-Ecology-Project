import argparse
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from antheia.config import load_config
from antheia.metrics import bootstrap_mean
from antheia.pairs import load_split
from antheia.store import FeatureStore
from antheia.twotower import Data, make_ranker, train_model

# Decisive test for S1: if an EXCLUDED observation-process head closes the provenance transfer gap,
# then debiasing IS achievable here and S1 ("not identifiable, here are the diagnostics") is wrong.
# The head sees only species-level documentation-process descriptors and is dropped at inference.


def process_features(cfg, store):
    prof = pd.read_parquet(cfg["cache_dir"] / "obs_profiles.parquet")
    bor = [c for c in prof.columns if c.startswith("bor::")]
    tot = prof[bor].sum(1).clip(lower=1)
    f = pd.DataFrame(index=prof.index)
    f["specimen_share"] = prof.get("bor::preserved_specimen", 0) / tot
    f["obs_share"] = prof.get("bor::human_observation", 0) / tot
    f["log_images"] = np.log1p(prof["n_images"])
    f["src_div"] = (prof[[c for c in prof.columns if c.startswith("src::")]] > 0).sum(1)
    f["n_pub"] = prof.get("n_publishers", 0)
    med = f.median()
    P = f.reindex(store.plants).fillna(med).to_numpy(np.float32)
    Q = f.reindex(store.polls).fillna(med).to_numpy(np.float32)
    z = lambda a: (a - a.mean(0)) / np.clip(a.std(0), 1e-6, None)
    return z(P), z(Q)


def main():
    ap = argparse.ArgumentParser(description="Two-tower with an excluded observation-process bias head")
    ap.add_argument("--config", default=None)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--emb", default="bioclip")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = load_config(args.config)
    store = FeatureStore(cfg)
    edges = pd.read_parquet(cfg["edges"])
    prov = pd.read_csv(cfg["edges"].parent / "edge_provenance_v1.csv")
    edges = edges.merge(prov, on=["plant", "pollinator"], how="left")
    edges["tier1"] = edges["n_inat"].notna() & (edges["n_inat"] < edges["n"])
    split = load_split(cfg["edges"].parent / "split_v1.json")
    nboot, seed = cfg["eval"]["bootstrap_n"], args.seed

    prefix = {"bioclip": "bioclip_text", "bioclip2": "bioclip2_text", "none": None}[args.emb]
    emb = None if prefix is None else (np.load(cfg["cache_dir"] / f"{prefix}_plants.npy"),
                                       np.load(cfg["cache_dir"] / f"{prefix}_polls.npy"))
    proc = process_features(cfg, store)
    print(f"process features: {proc[0].shape[1]} per species", flush=True)

    g = pd.read_csv(cfg["paths"]["globi"], usecols=["sourceTaxonName", "sourceTaxonOrderName"]).dropna()
    order = g.groupby("sourceTaxonName")["sourceTaxonOrderName"].agg(lambda s: s.mode().iat[0])
    hym = {store.q2i[s] for s in store.polls if order.get(s, "UNK") == "Hymenoptera"}

    results = {}
    rng_perm = np.random.default_rng(0)
    proc_perm = (proc[0][rng_perm.permutation(len(proc[0]))],
                 proc[1][rng_perm.permutation(len(proc[1]))])
    # Control: same capacity, same feature scale, but species-process association destroyed.
    # If the permuted head hurts as much as the real one, the loss is capacity/overfitting,
    # not the process features doing (or failing to do) their job.
    for tag, use_proc in [("no_bias_head", None), ("bias_head", proc), ("bias_head_permuted", proc_perm)]:
        t0 = time.time()
        data = Data(store, edges, split, cfg, emb=emb, seed=seed, proc=use_proc)
        val_p = {sp: set(store.idx_polls(gg["pollinator"]))
                 for sp, gg in edges[edges["plant"].isin(split["val"])].groupby("plant")}
        val_w = {sp: data.wide_plant(store.p2i[sp]) for sp in val_p}

        def val_eval(m):
            sc = make_ranker(m, data, args.device)
            return float(np.mean([len(p & set(np.argpartition(-sc(store.p2i[sp], val_w[sp]), 10)[:10].tolist())) / len(p)
                                  for sp, p in val_p.items()]))

        m, _ = train_model(data, args.device, seed=seed, val_eval=val_eval,
                           log=lambda s: print(f"  [{tag}] {s}", flush=True))
        sc = make_ranker(m, data, args.device)
        test = edges[edges["plant"].isin(split["test"])]
        r_all, r_cur, r_inat = [], [], []
        for sp, gg in test.groupby("plant"):
            w = data.wide_plant(store.p2i[sp])
            s = sc(store.p2i[sp], w)
            recs = gg.to_dict("records")
            allp = {store.q2i[r["pollinator"]] for r in recs}
            t10 = set(np.argpartition(-s, 10)[:10].tolist())
            r_all.append(len(allp & t10) / len(allp))
            for bucket, want in ((r_cur, True), (r_inat, False)):
                tg = {store.q2i[r["pollinator"]] for r in recs
                      if bool(r["tier1"]) == want and store.q2i[r["pollinator"]] in hym}
                if not tg:
                    continue
                sm = s.copy()
                blk = np.ones(len(sm), bool); blk[list(hym)] = False
                sm[blk] = -np.inf
                sm[list(allp - tg)] = -np.inf
                bucket.append(len(tg & set(np.argpartition(-sm, 10)[:10].tolist())) / len(tg))
        a, lo, hi, _ = bootstrap_mean(np.array(r_all), nboot, seed)
        ratio = np.mean(r_cur) / np.mean(r_inat)
        results[tag] = (a, lo, hi, np.mean(r_cur), np.mean(r_inat), ratio)
        print(f"  [{tag}] R@10 {a:.4f} [{lo:.4f},{hi:.4f}] | Hym curated {np.mean(r_cur):.4f} "
              f"vs iNat {np.mean(r_inat):.4f} -> ratio {ratio:.2f} ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n{'model':<16}{'R@10':>10}{'Hym curated':>14}{'Hym iNat':>11}{'ratio':>8}")
    for k, v in results.items():
        print(f"{k:<16}{v[0]:>10.4f}{v[3]:>14.4f}{v[4]:>11.4f}{v[5]:>8.2f}")
    d = abs(1 - results["bias_head"][5]) - abs(1 - results["no_bias_head"][5])
    print(f"\ngap-to-parity change with bias head: {d:+.3f}  (negative = gap CLOSED -> S1 wrong)")
    pd.DataFrame([{"model": k, "r10": v[0], "lo": v[1], "hi": v[2], "hym_curated": v[3],
                   "hym_inat": v[4], "ratio": v[5]} for k, v in results.items()]).to_csv(
        cfg["edges"].parent / "biashead_v1.csv", index=False)


if __name__ == "__main__":
    main()
