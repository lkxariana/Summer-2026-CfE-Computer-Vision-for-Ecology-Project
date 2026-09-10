"""Local-network completion for the fusion re-ranker (plan §1.5 applied to model 2).

Same protocol as eval/run_localnets.py: every local-network pair is removed from training, the retriever
(reference embedding model) and the fusion model are fit on what remains, and each surveyed network's
plants x pollinators block is scored. Blocks are small (<= 421 x 682), so every pair in a block is
re-scored by the fusion model rather than only a retriever top-K: logit = retriever + fusion delta.

  python eval/run_localnets_fusion.py --name M2.1_fusion_identity --config '{"field_dir": ..., "identity_only": true}' --seed 42
"""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))
from antheia.bundle import RUNS, config_hash, git_commit
from antheia.baselines import REGISTRY
from antheia.embednet import EmbedRanker
from antheia.fusion import FusionReranker
from antheia.store import UniverseStore
from eval.run_ladder import BASE_EMBED
from eval.run_localnets import nodf


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True)
    ap.add_argument("--retriever-config", default="{}")
    ap.add_argument("--retriever-model", default="embednet")
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--topk", type=int, default=500)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--min-plants", type=int, default=10)
    ap.add_argument("--min-polls", type=int, default=10)
    args = ap.parse_args()
    rcfg = json.loads(args.retriever_config); fcfg = json.loads(args.config)
    store = UniverseStore(curves="modelled")
    nets = pd.read_parquet(ROOT / "data/network/local_networks.parquet")
    nets = nets[nets.plant.isin(store.p2i) & nets.pollinator.isin(store.q2i)]
    local_pairs = set(zip(nets.plant, nets.pollinator))
    e = pd.read_parquet(ROOT / "data/network/edges.parquet")
    e = e[e.plant.isin(store.p2i) & e.pollinator.isin(store.q2i)]
    is_local = pd.Series(list(zip(e.plant, e.pollinator))).isin(local_pairs).to_numpy()
    train = e[~is_local]
    assert not (set(zip(train.plant, train.pollinator)) & local_pairs)
    print(f"[localnet-fusion] {len(local_pairs):,} local pairs removed; training on {len(train):,} edges", flush=True)

    t0 = time.time()
    if args.retriever_model == "embednet":
        ret = EmbedRanker(**dict(BASE_EMBED, **rcfg, seed=args.seed, device=args.device)).fit(train, store)
    else:
        try:
            ret = REGISTRY[args.retriever_model](**rcfg).fit(train, store)
        except TypeError:
            ret = REGISTRY[args.retriever_model]().fit(train, store)
    train_plants = sorted(set(train.plant)); tpi = store.idx_plants(train_plants)
    K = args.topk
    s_tr = np.empty((len(tpi), K), np.float32); c_tr = np.empty((len(tpi), K), np.int64)
    for i, p in enumerate(tpi):
        s = ret.score_plant(int(p)); o = np.argsort(-s)[:K]; s_tr[i] = s[o]; c_tr[i] = o
    ret_cache = {}
    def ret_scores(p):
        if p not in ret_cache:
            ret_cache[p] = ret.score_plant(int(p)).astype(np.float32)
        return ret_cache[p]
    id_override = None
    src = fcfg.get("id_source", "text")
    if src != "text":
        with torch.no_grad():
            n_p = len(store.plants)
            if src == "retriever_proj":
                pq = (ret.proj_text_q if getattr(ret.cfg, "text_proj", "shared") == "kingdom" else ret.proj_text)
                id_override = (ret.proj_text(ret.text_p).cpu().numpy(), pq(ret.text_q).cpu().numpy())
            elif src == "retriever_h":
                id_override = (ret.h_all[:n_p].cpu().numpy(), ret.hq_all.cpu().numpy())
            else:
                raise ValueError(src)
    fus = FusionReranker(**dict(fcfg, seed=args.seed, device=args.device)).fit(train, store, s_tr, c_tr, tpi, id_override=id_override)

    grid = pd.read_parquet(ROOT / "data/features/grid.parquet"); tree = cKDTree(np.c_[grid.centroid_lat, grid.centroid_lon])
    rows = []; pooled_y, pooled_s = [], []
    for net, g in nets.groupby("network"):
        P = sorted(set(g.plant)); Q = sorted(set(g.pollinator))
        if len(P) < args.min_plants or len(Q) < args.min_polls:
            continue
        qi = store.idx_polls(Q)
        S = np.empty((len(P), len(Q)), np.float64)
        for i, p in enumerate(P):
            pi = store.p2i[p]; base = ret_scores(pi)[qi]
            new, _ = fus.rerank(pi, base, qi, len(store.polls))          # re-score every pair in the block
            S[i] = new
        Y = np.zeros(S.shape, np.int8); pos = set(zip(g.plant, g.pollinator))
        for i, p in enumerate(P):
            for j, q in enumerate(Q):
                if (p, q) in pos: Y[i, j] = 1
        L = int(Y.sum()); conn = L / Y.size; y, s = Y.ravel(), S.ravel()
        aupr = average_precision_score(y, s); auroc = roc_auc_score(y, s)
        top = np.argsort(-s)[:L]; Yhat = np.zeros_like(y); Yhat[top] = 1; Yhat = Yhat.reshape(Y.shape)
        lat, lon = g.lat.median(), g.lon.median()
        d, _ = tree.query([lat, lon]) if np.isfinite(lat) and np.isfinite(lon) else (np.inf, None)
        rows.append(dict(network=net, dataset=g.dataset.iloc[0], n_plants=len(P), n_polls=len(Q), n_links=L, connectance=conn,
                         aupr=aupr, aupr_lift=aupr / conn, auroc=auroc,
                         deg_spearman_plants=spearmanr(Y.sum(1), Yhat.sum(1)).correlation, deg_spearman_polls=spearmanr(Y.sum(0), Yhat.sum(0)).correlation,
                         nodf_obs=nodf(Y), nodf_pred=nodf(Yhat), link_precision_at_L=float((Yhat & Y).sum() / L), in_grid=bool(d <= 0.5),
                         year_min=g.year.min(), year_max=g.year.max()))
        pooled_y.append(y); pooled_s.append(s)
    df = pd.DataFrame(rows); y_all, s_all = np.concatenate(pooled_y), np.concatenate(pooled_s)
    rng = np.random.default_rng(42)
    boots = np.array([df.aupr.to_numpy()[rng.integers(0, len(df), len(df))].mean() for _ in range(2000)])
    met = dict(n_networks=len(df), n_in_grid=int(df.in_grid.sum()), mean_aupr=float(df.aupr.mean()),
               mean_aupr_lo=float(np.percentile(boots, 2.5)), mean_aupr_hi=float(np.percentile(boots, 97.5)),
               mean_connectance=float(df.connectance.mean()), mean_aupr_lift=float(df.aupr_lift.mean()), mean_auroc=float(df.auroc.mean()),
               pooled_aupr=float(average_precision_score(y_all, s_all)), pooled_auroc=float(roc_auc_score(y_all, s_all)),
               pooled_connectance=float(y_all.mean()), mean_deg_spearman_plants=float(df.deg_spearman_plants.mean()),
               mean_deg_spearman_polls=float(df.deg_spearman_polls.mean()), mean_nodf_obs=float(df.nodf_obs.mean()),
               mean_nodf_pred=float(df.nodf_pred.mean()), mean_link_precision_at_L=float(df.link_precision_at_L.mean()), wall_s=time.time() - t0)
    cfg_all = {"model": "fusion", "retriever_model": args.retriever_model, "retriever": rcfg, **fcfg, "topk": K}
    h = config_hash(cfg_all); out = RUNS / args.name / h / "localnet" / f"s{args.seed}"; out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "per_network.csv", index=False); json.dump(met, open(out / "metrics.json", "w"), indent=1)
    json.dump({"model": args.name, "config": cfg_all, "config_hash": h, "split": "localnet", "seed": args.seed, "git": git_commit(),
               "n_queries": len(df), "n_candidates": int(y_all.size)}, open(out / "config.json", "w"), indent=1)
    print(f"  {len(df)} networks | mean AUPR {met['mean_aupr']:.3f} [{met['mean_aupr_lo']:.3f},{met['mean_aupr_hi']:.3f}] (lift {met['mean_aupr_lift']:.2f}x) "
          f"| pooled AUPR {met['pooled_aupr']:.3f} AUROC {met['pooled_auroc']:.3f} | deg rho {met['mean_deg_spearman_plants']:.3f}/{met['mean_deg_spearman_polls']:.3f} "
          f"| NODF pred {met['mean_nodf_pred']:.1f} | precision@L {met['mean_link_precision_at_L']:.3f} ({met['wall_s']:.0f}s)\n[bundle] {out}", flush=True)


if __name__ == "__main__":
    main()
