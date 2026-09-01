import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def pair_metrics(y, scores):
    return {"pr_auc": average_precision_score(y, scores), "roc_auc": roc_auc_score(y, scores)}


def rank_metrics(scorer, partners_by_plant, store, ks):
    """Ranks every pollinator in the universe for each test plant and scores the ranking.

    Args:
        scorer: callable(plant_index) -> score vector over all pollinators.
        partners_by_plant: dict plant name -> set of true partner pollinator indices (held out).
        ks: iterable of cutoffs.

    Returns:
        DataFrame with one row per test plant: degree, recall@k and hit@k per cutoff.
    """
    kmax = max(ks)
    rows = []
    for sp, partners in partners_by_plant.items():
        s = scorer(store.p2i[sp])
        top = np.argpartition(-s, kmax)[:kmax]
        top = top[np.argsort(-s[top])]
        row = {"plant": sp, "degree": len(partners)}
        for k in ks:
            hits = len(partners.intersection(top[:k]))
            row[f"recall@{k}"] = hits / len(partners)
            row[f"hit@{k}"] = float(hits > 0)
        rows.append(row)
    return pd.DataFrame(rows)


def ranking_metrics(scores, relevant, ks=(10, 50)):
    """All per-query ranking metrics from one score vector and a relevant-index set.

    recall@k  - fraction of KNOWN partners retrieved; degree-capped at k/|relevant|
    nrecall@k - recall divided by its achievable maximum, removing the degree cap
    hit@k     - was the shortlist worth opening at all
    mrr       - 1/rank of the FIRST partner (ignores all others; single-answer metric)
    map       - mean average precision: the multi-relevant generalisation of MRR
    ndcg@k    - position-discounted, normalised by the ideal ranking (degree-fair)
    """
    order = np.argsort(-scores)
    rel = np.fromiter((1.0 if i in relevant else 0.0 for i in order[:max(ks)]), float, max(ks))
    R = len(relevant)
    out = {}
    for k in ks:
        hits = rel[:k].sum()
        out[f"recall@{k}"] = hits / R
        out[f"nrecall@{k}"] = hits / min(R, k)
        out[f"hit@{k}"] = float(hits > 0)
        disc = 1.0 / np.log2(np.arange(2, k + 2))
        idcg = disc[:min(R, k)].sum()
        out[f"ndcg@{k}"] = float((rel[:k] * disc).sum() / idcg) if idcg > 0 else 0.0
    ranks = np.flatnonzero(np.isin(order, list(relevant))) + 1
    out["mrr"] = 1.0 / ranks[0] if len(ranks) else 0.0
    out["map"] = float(np.mean([(j + 1) / r for j, r in enumerate(ranks)])) if len(ranks) else 0.0
    out["median_rank_first"] = float(ranks[0]) if len(ranks) else np.nan
    return out


def bootstrap_mean(values, n, seed):
    """Bootstrap mean with 95% CI: returns (mean, lo, hi, std)."""
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n, len(values)))
    means = values[idx].mean(1)
    return values.mean(), *np.percentile(means, [2.5, 97.5]), means.std()


def bootstrap_pr_by_plant(pairs, n, seed):
    """Bootstrap PR-AUC by resampling test plants (cluster bootstrap): returns (pr, lo, hi, std)."""
    rng = np.random.default_rng(seed)
    groups = {sp: g for sp, g in pairs.groupby("plant")}
    names = list(groups)
    stats = []
    for _ in range(n):
        sample = pd.concat([groups[names[i]] for i in rng.integers(0, len(names), len(names))])
        if sample["label"].nunique() == 2:
            stats.append(average_precision_score(sample["label"], sample["score"]))
    base = average_precision_score(pairs["label"], pairs["score"])
    return base, *np.percentile(stats, [2.5, 97.5]), np.std(stats)
