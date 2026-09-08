"""The artifact contract of the connectivity ladder (plan §8.1).

Every run is (model, config, split, seed) and writes one bundle under runs/. Tables and bootstraps
read bundles and nothing else. A bundle holds the full score matrix, every metric the protocol
reports, the strata, and enough provenance to reproduce it.
"""
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"


def config_hash(cfg: dict) -> str:
    return hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:10]


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def ap_at_prevalence(y, s, prev):
    """Average precision with negatives re-weighted so positives are `prev` of the data (full ranking, no sampling)."""
    order = np.argsort(-s, kind="stable"); y = y[order]
    n_pos, n_neg = y.sum(), (1 - y).sum()
    w_neg = (n_pos * (1 - prev) / prev) / n_neg
    tp = np.cumsum(y); fp = np.cumsum(1 - y) * w_neg
    return float(((tp / (tp + fp)) * y).sum() / n_pos)


def per_query(scores, relevant, ks=(10, 50)):
    order = np.argsort(-scores)
    rel = np.isin(order, list(relevant)).astype(float)
    R = len(relevant); out = {}
    for k in ks:
        hits = rel[:k].sum()
        out[f"recall@{k}"] = hits / R; out[f"nrecall@{k}"] = hits / min(R, k)
    hits_cum = np.cumsum(rel); ranks = np.flatnonzero(rel) + 1
    out["ap"] = float((hits_cum[ranks - 1] / ranks).mean()) if len(ranks) else 0.0
    out["rank_first"] = float(ranks[0]) if len(ranks) else np.nan
    return out


def evaluate_scores(S, Y, query_names, strata: dict, prevalences=(0.25, 0.5)):
    """S, Y: [n_queries, n_candidates]. strata: name -> bool array over queries. Returns (metrics, per_query_df)."""
    y, s = Y.ravel().astype(np.int8), S.ravel().astype(np.float64)
    m = {"n_queries": int(S.shape[0]), "n_candidates": int(S.shape[1]), "n_positives": int(y.sum()),
         "prevalence": float(y.mean()),
         "aupr": float(average_precision_score(y, s)), "auroc": float(roc_auc_score(y, s))}
    for p in prevalences:
        m[f"aupr_at_{p:g}"] = ap_at_prevalence(y, s, p)
    rows = []
    for i in range(S.shape[0]):
        rel = set(np.flatnonzero(Y[i]).tolist())
        if not rel:
            continue
        r = per_query(S[i], rel); r["query"] = query_names[i]; r["idx"] = i
        for k, v in strata.items():
            r[k] = bool(v[i])
        rows.append(r)
    pq = pd.DataFrame(rows)
    for c in ("nrecall@10", "nrecall@50", "recall@10", "ap"):
        m[c.replace("@", "_at_")] = float(pq[c].mean())
    m["map"] = m.pop("ap")
    for k in strata:
        for c in ("nrecall@10", "nrecall@50"):
            for val, tag in ((True, k), (False, f"not_{k}")):
                sub = pq[pq[k] == val]
                if len(sub):
                    m[f"{c.replace('@', '_at_')}__{tag}"] = float(sub[c].mean())
        # pooled AUPR within stratum
        idx = np.flatnonzero(strata[k])
        if len(idx) and Y[idx].sum() > 0:
            m[f"aupr__{k}"] = float(average_precision_score(Y[idx].ravel(), S[idx].ravel().astype(np.float64)))
    return m, pq


def write_bundle(model: str, cfg: dict, split: str, seed: int, S, Y, query_names, candidate_names, strata: dict,
                 extra: dict | None = None, candidates_topk=None, wall_s=None):
    h = config_hash(cfg)
    out = RUNS / model / h / split / f"s{seed}"
    out.mkdir(parents=True, exist_ok=True)
    m, pq = evaluate_scores(S, Y, query_names, strata)
    m.update(extra or {})
    m["wall_s"] = wall_s
    np.save(out / "scores.npy", S.astype(np.float16))
    np.save(out / "Y.npy", Y.astype(np.int8))
    if candidates_topk is not None:
        np.save(out / "candidates.npy", candidates_topk.astype(np.int32))
    pq.to_parquet(out / "per_query.parquet", index=False)
    json.dump({"model": model, "config": cfg, "config_hash": h, "split": split, "seed": seed,
               "git": git_commit(), "host": platform.node(), "time": time.strftime("%Y-%m-%d %H:%M:%S"),
               "python": platform.python_version(), "n_queries": len(query_names), "n_candidates": len(candidate_names)},
              open(out / "config.json", "w"), indent=1)
    json.dump(m, open(out / "metrics.json", "w"), indent=1)
    return out, m


def load_bundles(model=None):
    rows = []
    for f in RUNS.glob("**/s*/metrics.json"):
        c = json.load(open(f.with_name("config.json"))); m = json.load(open(f))
        if model and c["model"] != model:
            continue
        rows.append({**{k: c[k] for k in ("model", "config_hash", "split", "seed", "git")}, **m, "path": str(f.parent)})
    return pd.DataFrame(rows)


def seed_pooled(df, group=("model", "config_hash", "split"), min_seeds=3):
    """Mean over seeds per metric; bundles with fewer than `min_seeds` are marked incomplete."""
    num = df.select_dtypes("number").columns.difference(["seed"])
    g = df.groupby(list(group))
    out = g[num].mean(); out["n_seeds"] = g.seed.nunique(); out["complete"] = out.n_seeds >= min_seeds
    return out.reset_index()
