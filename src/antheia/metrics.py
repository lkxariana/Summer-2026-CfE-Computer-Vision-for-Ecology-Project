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
