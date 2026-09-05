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


def retrieval_metrics(scores, relevant, ks=(10, 20)):
    """All per-query ranking metrics from one score vector and a relevant-index set.

    Primary metric is nrecall@k, recall normalised by min(R, k). Plain recall@k is capped
    below 1 whenever a plant has more than k recorded partners -- 29% of held-out plants --
    so it measures the degree distribution as much as the model. Both are returned;
    recall@k is reported alongside for comparability with the wider literature.

    nDCG uses binary gains (graded gains give an identical ordering). k<10 and MRR are
    excluded: both reward the popularity shortcut.
    """
    order = np.argsort(-scores)
    rel = np.fromiter((1.0 if i in relevant else 0.0 for i in order[:max(ks)]), float, max(ks))
    R = len(relevant)
    out = {}
    for k in ks:
        hits = rel[:k].sum()
        out[f"recall@{k}"] = hits / R
        out[f"nrecall@{k}"] = hits / min(R, k)
        disc = 1.0 / np.log2(np.arange(2, k + 2))
        idcg = disc[:min(R, k)].sum()
        out[f"ndcg@{k}"] = float((rel[:k] * disc).sum() / idcg) if idcg > 0 else 0.0
    ranks = np.flatnonzero(np.isin(order, list(relevant))) + 1
    out["rank_first"] = float(ranks[0]) if len(ranks) else np.nan
    # Average precision over the whole ranking: cutoff-free, and unlike pooled PR-AUC it asks
    # nothing about whether this plant's scores are on the same scale as another's.
    out["ap"] = float((np.arange(1, len(ranks) + 1) / ranks).sum() / R) if len(ranks) else 0.0
    return out


def pooled_metrics(y, scores):
    """Pooled discrimination: PR-AUC is primary, ROC-AUC reported for comparability only.

    ROC-AUC is insensitive to the vast negative class and was the metric on which the
    original 139-pair benchmark scored 0.99 for an N-only model; treat it as context, not evidence.
    """
    return {"pr_auc": average_precision_score(y, scores), "roc_auc": roc_auc_score(y, scores)}


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


def paired_bootstrap(a, b, n, seed):
    """Paired bootstrap over the same queries: returns (mean difference, lo, hi, p).

    Both methods are evaluated on the identical test plants, so resampling the plants rather than
    the scores keeps the pairing and removes the between-plant variance that dominates an unpaired
    comparison. `p` is two-sided, the fraction of resamples whose difference has the opposite sign
    to the observed mean.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert a.shape == b.shape, "paired comparison needs the same queries in the same order"
    d = a - b
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(n, len(d)))].mean(1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return d.mean(), lo, hi, min(p, 1.0)
