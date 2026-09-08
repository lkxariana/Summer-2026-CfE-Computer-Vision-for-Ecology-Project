"""Local-network completion: the connectivity evaluation with real non-interactions (plan §1.5, §5 X2).

Each surveyed field network (data/network/local_networks.parquet; scripts/extract_local_networks.py) lists
the plants and pollinators present at one site and the pairs recorded interacting. Because the site was
surveyed, an absent pair among present species is an observed non-interaction, not an unlabelled one.
The model is trained with **every local-network pair removed from the training edges** (whatever other
source also recorded it), then scores each network's plants x pollinators block.

Per network: AUPR, AUROC, connectance (= chance AUPR), and structure fidelity of the predicted network
at matched connectance -- take the top-L scored pairs, L = observed links, and compare plant-degree and
pollinator-degree Spearman correlations and nestedness (NODF) against the observed network. Aggregates:
mean over networks with a bootstrap over networks, pooled AUPR over all blocks, and per-dataset means.

  python eval/run_localnets.py --model embednet --name M1.0_reference --seed 42
"""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))
from antheia.baselines import REGISTRY
from antheia.bundle import RUNS, config_hash, git_commit
from antheia.store import UniverseStore
from eval.run_ladder import BASE_EMBED


def nodf(M):
    """Nestedness (NODF, Almeida-Neto et al. 2008) of a binary matrix, 0-100."""
    M = np.asarray(M, bool)
    def axis_nodf(A):
        deg = A.sum(1); n = len(A); vals = []
        for i in range(n):
            for j in range(n):
                if i != j and deg[i] > deg[j] and deg[j] > 0:
                    vals.append((A[i] & A[j]).sum() / deg[j])
        return np.mean(vals) * 100 if vals else 0.0
    n_r, n_c = M.shape
    pairs_r, pairs_c = n_r * (n_r - 1) / 2, n_c * (n_c - 1) / 2
    r, c = axis_nodf(M), axis_nodf(M.T)
    return float((r * pairs_r + c * pairs_c) / max(pairs_r + pairs_c, 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--config", default="{}")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--min-plants", type=int, default=10)
    ap.add_argument("--min-polls", type=int, default=10)
    args = ap.parse_args()
    cfg = json.loads(args.config); name = args.name or args.model
    store = UniverseStore(curves="modelled")

    nets = pd.read_parquet(ROOT / "data/network/local_networks.parquet")
    nets = nets[nets.plant.isin(store.p2i) & nets.pollinator.isin(store.q2i)]
    local_pairs = set(zip(nets.plant, nets.pollinator))
    e = pd.read_parquet(ROOT / "data/network/edges.parquet")
    e = e[e.plant.isin(store.p2i) & e.pollinator.isin(store.q2i)]
    is_local = pd.Series(list(zip(e.plant, e.pollinator))).isin(local_pairs).to_numpy()
    train = e[~is_local]
    assert not (set(zip(train.plant, train.pollinator)) & local_pairs), "local-network pairs leak into training"
    print(f"[localnet] {len(local_pairs):,} local-network pairs removed ({int(is_local.sum()):,} were in edges); "
          f"training on {len(train):,} edges", flush=True)

    t0 = time.time()
    Model = REGISTRY[args.model]
    kw = dict(BASE_EMBED, **cfg, seed=args.seed, device=args.device) if args.model == "embednet" else dict(cfg)
    try:
        m = Model(**kw).fit(train, store)
    except TypeError:
        m = Model().fit(train, store)
    fit_s = time.time() - t0

    warm_plants = set(train.plant)          # a plant is warm if any edge survives the removal
    rows = []; pooled_y, pooled_s = [], []
    grid = pd.read_parquet(ROOT / "data/features/grid.parquet")
    from scipy.spatial import cKDTree
    tree = cKDTree(np.c_[grid.centroid_lat, grid.centroid_lon])
    cache = {}
    for net, g in nets.groupby("network"):
        P = sorted(set(g.plant)); Q = sorted(set(g.pollinator))
        if len(P) < args.min_plants or len(Q) < args.min_polls:
            continue
        qi = store.idx_polls(Q)
        S = np.vstack([cache.setdefault(p, m.score_plant(store.p2i[p]))[qi] for p in P]).astype(np.float64)
        Y = np.zeros(S.shape, np.int8)
        pos = {(p, q) for p, q in zip(g.plant, g.pollinator)}
        for i, p in enumerate(P):
            for j, q in enumerate(Q):
                if (p, q) in pos:
                    Y[i, j] = 1
        L = int(Y.sum()); conn = L / Y.size
        y, s = Y.ravel(), S.ravel()
        aupr = average_precision_score(y, s); auroc = roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan
        # predicted network at matched connectance
        top = np.argsort(-s)[:L]; Yhat = np.zeros_like(y); Yhat[top] = 1; Yhat = Yhat.reshape(Y.shape)
        dp = spearmanr(Y.sum(1), Yhat.sum(1)).correlation; dq = spearmanr(Y.sum(0), Yhat.sum(0)).correlation
        lat, lon = g.lat.median(), g.lon.median()
        d, _ = tree.query([lat, lon]) if np.isfinite(lat) and np.isfinite(lon) else (np.inf, None)
        wm = np.array([p in warm_plants for p in P])
        def sub_aupr(mask):
            Ys, Ss = Y[mask], S[mask]
            return float(average_precision_score(Ys.ravel(), Ss.ravel())) if mask.any() and 0 < Ys.sum() < Ys.size else np.nan
        rows.append(dict(network=net, dataset=g.dataset.iloc[0], n_plants=len(P), n_polls=len(Q), n_links=L, connectance=conn,
                         aupr=aupr, aupr_lift=aupr / conn, auroc=auroc, deg_spearman_plants=dp, deg_spearman_polls=dq,
                         n_warm_plants=int(wm.sum()), aupr_warm=sub_aupr(wm), aupr_cold=sub_aupr(~wm),
                         conn_warm=float(Y[wm].mean()) if wm.any() else np.nan, conn_cold=float(Y[~wm].mean()) if (~wm).any() else np.nan,
                         nodf_obs=nodf(Y), nodf_pred=nodf(Yhat), link_precision_at_L=float((Yhat & Y).sum() / L),
                         in_grid=bool(d <= 0.5), year_min=g.year.min(), year_max=g.year.max()))
        pooled_y.append(y); pooled_s.append(s)
    df = pd.DataFrame(rows)
    y_all, s_all = np.concatenate(pooled_y), np.concatenate(pooled_s)
    rng = np.random.default_rng(42)
    boots = np.array([df.aupr.to_numpy()[rng.integers(0, len(df), len(df))].mean() for _ in range(2000)])
    met = dict(n_networks=len(df), n_in_grid=int(df.in_grid.sum()),
               mean_aupr=float(df.aupr.mean()), mean_aupr_lo=float(np.percentile(boots, 2.5)), mean_aupr_hi=float(np.percentile(boots, 97.5)),
               mean_connectance=float(df.connectance.mean()), mean_aupr_lift=float(df.aupr_lift.mean()),
               mean_auroc=float(df.auroc.mean()), pooled_aupr=float(average_precision_score(y_all, s_all)),
               pooled_auroc=float(roc_auc_score(y_all, s_all)), pooled_connectance=float(y_all.mean()),
               mean_deg_spearman_plants=float(df.deg_spearman_plants.mean()), mean_deg_spearman_polls=float(df.deg_spearman_polls.mean()),
               mean_nodf_obs=float(df.nodf_obs.mean()), mean_nodf_pred=float(df.nodf_pred.mean()),
               mean_link_precision_at_L=float(df.link_precision_at_L.mean()), fit_s=fit_s, wall_s=time.time() - t0,
               mean_aupr_warm=float(df.aupr_warm.mean()), mean_aupr_cold=float(df.aupr_cold.mean()),
               mean_conn_warm=float(df.conn_warm.mean()), mean_conn_cold=float(df.conn_cold.mean()),
               n_networks_with_cold=int(df.aupr_cold.notna().sum()))
    for ds, gd in df.groupby("dataset", dropna=False):
        met[f"mean_aupr__{ds if isinstance(ds, str) else 'other'}"] = float(gd.aupr.mean())
    h = config_hash({"model": args.model, **cfg})
    out = RUNS / name / h / "localnet" / f"s{args.seed}"; out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "per_network.csv", index=False)
    json.dump(met, open(out / "metrics.json", "w"), indent=1)
    json.dump({"model": name, "config": {"model": args.model, **cfg}, "config_hash": h, "split": "localnet", "seed": args.seed,
               "git": git_commit(), "n_queries": len(df), "n_candidates": int(y_all.size)}, open(out / "config.json", "w"), indent=1)
    print(f"  {len(df)} networks ({met['n_in_grid']} in grid) | mean AUPR {met['mean_aupr']:.3f} [{met['mean_aupr_lo']:.3f},{met['mean_aupr_hi']:.3f}] "
          f"(mean connectance {met['mean_connectance']:.3f}, lift {met['mean_aupr_lift']:.2f}x) | pooled AUPR {met['pooled_aupr']:.3f} "
          f"AUROC {met['pooled_auroc']:.3f} | degree rho plants {met['mean_deg_spearman_plants']:.3f} polls {met['mean_deg_spearman_polls']:.3f} "
          f"| NODF obs {met['mean_nodf_obs']:.1f} pred {met['mean_nodf_pred']:.1f} | precision@L {met['mean_link_precision_at_L']:.3f} "
          f"| warm-plant AUPR {met['mean_aupr_warm']:.3f} (conn {met['mean_conn_warm']:.3f}) cold-plant AUPR {met['mean_aupr_cold']:.3f} (conn {met['mean_conn_cold']:.3f}, {met['n_networks_with_cold']} nets)  ({met['wall_s']:.0f}s)\n[bundle] {out}", flush=True)


if __name__ == "__main__":
    main()
