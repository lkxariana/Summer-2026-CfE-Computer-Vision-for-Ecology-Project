"""Tables from bundles, and nothing else (plan §8.1).

  python eval/report_ladder.py --split cold_plant/val --ref M1.0_reference

For every (model, config) on the split: metrics averaged over seeds (bundles with < 3 seeds flagged
incomplete), and paired bootstraps against the reference on the same plants -- nrecall@10 from
per-query files, AUPR from saved score matrices with the O(N) weighted-AP bootstrap, both within seed
and averaged over seeds.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from antheia.bundle import RUNS, load_bundles, seed_pooled
from antheia.metrics import paired_bootstrap


def boot_ap(y_sorted, plant_sorted, n_plants, idx_list):
    out = np.empty(len(idx_list))
    for k, ii in enumerate(idx_list):
        wv = np.bincount(ii, minlength=n_plants).astype(np.float64)[plant_sorted]
        tp = np.cumsum(wv * y_sorted); tot = np.cumsum(wv)
        prec = np.divide(tp, tot, out=np.zeros_like(tp), where=tot > 0)
        out[k] = (wv * y_sorted * prec).sum() / tp[-1]
    return out


_STORE = {}


def rebuild_Y(path, split):
    """Label matrix for a bundle written before Y.npy was saved: tier-A partners of its queries over the
    split's candidate set, in the bundle's own row/column order."""
    if "store" not in _STORE:
        from antheia.store import UniverseStore
        _STORE["store"] = UniverseStore(curves="modelled")
        e = pd.read_parquet(ROOT / "data/network/edges.parquet")
        _STORE["A"] = e[(e.tier == "A") & e.plant.isin(_STORE["store"].p2i) & e.pollinator.isin(_STORE["store"].q2i)]
    store, A = _STORE["store"], _STORE["A"]
    cfg = json.load(open(Path(path) / "config.json"))
    pq = pd.read_parquet(Path(path) / "per_query.parquet")
    queries = pq.sort_values("idx")["query"].tolist()
    if not split.startswith("cold_plant"):
        raise NotImplementedError("Y rebuild only for cold_plant; newer bundles carry Y.npy")
    Y = np.zeros((cfg["n_queries"], cfg["n_candidates"]), np.int8)
    for i, p in enumerate(queries):
        Y[i, store.idx_polls(A[A.plant == p].pollinator)] = 1
    return Y


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="cold_plant/val")
    ap.add_argument("--ref", default="M1.0_reference")
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--models", nargs="*", default=None)
    args = ap.parse_args()
    df = load_bundles()
    df = df[df.split == args.split]
    if args.models:
        df = df[df.model.isin(args.models)]
    pooled = seed_pooled(df)
    cols = ["model", "n_seeds", "complete", "aupr", "aupr_at_0.25", "aupr_at_0.5", "auroc", "nrecall_at_10", "nrecall_at_50", "map",
            "nrecall_at_10__genus_unseen", "aupr__genus_unseen"]
    cols = [c for c in cols if c in pooled.columns]
    print(f"\n== {args.split}: seed-averaged ==")
    print(pooled[cols].sort_values("aupr", ascending=False).round(4).to_string(index=False))

    # paired bootstraps vs reference, within seed, averaged
    ref = df[df.model == args.ref]
    if ref.empty:
        print(f"\n(no reference bundle {args.ref}; skipping paired tests)"); return
    rng = np.random.default_rng(42)
    print(f"\n== paired bootstrap vs {args.ref} ({args.boot} plant resamples, averaged over shared seeds) ==")
    print(f"{'model':<34} {'d AUPR':>8} {'95% CI':>20} {'p':>7}   {'d nR@10':>8} {'p':>7}   {'d unseen nR@10':>14} {'p':>7}")
    rows = []
    for model, g in df.groupby("model"):
        if model == args.ref:
            continue
        seeds = sorted(set(g.seed) & set(ref.seed))
        if not seeds:
            continue
        d_ap, d_nr, d_un = [], [], []
        for sd in seeds:
            a = g[g.seed == sd].iloc[0]; r = ref[ref.seed == sd].iloc[0]
            pa, pr = pd.read_parquet(Path(a.path) / "per_query.parquet"), pd.read_parquet(Path(r.path) / "per_query.parquet")
            m = pa.merge(pr, on="query", suffixes=("_a", "_r"))
            n = len(m); idx = [rng.integers(0, n, n) for _ in range(args.boot)]
            d_nr.append(np.array([(m["nrecall@10_a"].to_numpy()[ii] - m["nrecall@10_r"].to_numpy()[ii]).mean() for ii in idx]))
            un = m["genus_unseen_a"].to_numpy()
            d_un.append(np.array([(m["nrecall@10_a"].to_numpy()[ii][un[ii]] - m["nrecall@10_r"].to_numpy()[ii][un[ii]]).mean() for ii in idx]))
            Sa = np.load(Path(a.path) / "scores.npy").astype(np.float32); Sr = np.load(Path(r.path) / "scores.npy").astype(np.float32)
            # Y from per_query rows: positives are unknown here, so rebuild from the ranks is not possible;
            # use the candidate order saved with the bundle instead: recover from metrics? -> rebuild Y via edges
            Y = np.load(Path(a.path) / "Y.npy") if (Path(a.path) / "Y.npy").exists() else rebuild_Y(a.path, args.split)
            plant_of = np.repeat(np.arange(Sa.shape[0]), Sa.shape[1]); y = Y.ravel()
            oa = np.argsort(-Sa.ravel(), kind="stable"); orr = np.argsort(-Sr.ravel(), kind="stable")
            d_ap.append(boot_ap(y[oa].astype(float), plant_of[oa], Sa.shape[0], idx) - boot_ap(y[orr].astype(float), plant_of[orr], Sr.shape[0], idx))
        d_nr = np.mean(d_nr, 0); d_un = np.mean(d_un, 0)
        pn = 2 * min((d_nr > 0).mean(), (d_nr < 0).mean()); pu = 2 * min((d_un > 0).mean(), (d_un < 0).mean())
        if d_ap:
            d_ap = np.mean(d_ap, 0); lo, hi = np.percentile(d_ap, [2.5, 97.5]); p = 2 * min((d_ap > 0).mean(), (d_ap < 0).mean())
            ap_s = f"{d_ap.mean():+8.4f} [{lo:+.4f},{hi:+.4f}] {min(p,1):7.4f}"
        else:
            ap_s = f"{'(no Y.npy)':>37}"
        print(f"{model:<34} {ap_s}   {d_nr.mean():+8.4f} {min(pn,1):7.4f}   {d_un.mean():+14.4f} {min(pu,1):7.4f}")
        rows.append(dict(model=model, seeds=len(seeds), d_aupr=float(np.mean(d_ap)) if len(d_ap) else np.nan,
                         d_nr10=float(d_nr.mean()), p_nr10=min(pn, 1), d_unseen=float(d_un.mean()), p_unseen=min(pu, 1)))
    out = ROOT / "results" / f"ladder_{args.split.replace('/', '_')}.csv"
    pooled[cols].to_csv(out, index=False); pd.DataFrame(rows).to_csv(out.with_name(out.stem + "_paired.csv"), index=False)
    print(f"\n[wrote] {out.relative_to(ROOT)}, {out.with_name(out.stem + '_paired.csv').name}")


if __name__ == "__main__":
    main()
