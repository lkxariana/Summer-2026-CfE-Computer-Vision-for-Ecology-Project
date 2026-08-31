import json
import numpy as np
import pandas as pd


def _stratified_groups(edges, seed):
    """Shuffled plant lists per degree-quartile stratum."""
    deg = edges.groupby("plant").size()
    strata = pd.qcut(deg, q=4, labels=False, duplicates="drop")
    rng = np.random.default_rng(seed)
    groups = []
    for s in sorted(strata.unique()):
        plants = strata.index[strata == s].to_numpy()
        rng.shuffle(plants)
        groups.append(plants)
    return groups


def plant_split(edges, seed, test_frac, val_frac=0.0):
    """Degree-stratified split of plants-with-positives into train/val/test plant lists."""
    out = {"train": [], "val": [], "test": []}
    for plants in _stratified_groups(edges, seed):
        n = len(plants)
        n_te, n_va = int(round(n * test_frac)), int(round(n * val_frac))
        out["test"] += list(plants[:n_te])
        out["val"] += list(plants[n_te:n_te + n_va])
        out["train"] += list(plants[n_te + n_va:])
    return {k: sorted(v) for k, v in out.items()}


def kfold_plants(edges, k, seed):
    """Degree-stratified k-fold partition of plants-with-positives; returns k sorted plant lists."""
    folds = [[] for _ in range(k)]
    for plants in _stratified_groups(edges, seed):
        for i, sp in enumerate(plants):
            folds[i % k].append(sp)
    return [sorted(f) for f in folds]


def save_split(split, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(split, f, indent=1)


def load_split(path):
    with open(path) as f:
        return json.load(f)


def sample_negatives(pos_pairs, plant_pool, store, ratio, mode, seed, known_pos):
    """Samples unlabeled (plant, pollinator) pairs as negatives for the given positive set.

    Modes: 'uniform' draws both sides uniformly; 'cooc' draws plants from the positives' plant
    multiset and pollinators uniformly among that plant's co-occurring (N>0) pollinators, matching
    the encounter profile of the positives; 'mixed' is half and half.

    Args:
        pos_pairs: positive pairs df with plant/pollinator columns (defines the cooc plant multiset).
        plant_pool: plant names negatives may use (train plants only, to avoid split leakage).
        known_pos: set of all documented (plant, pollinator) tuples to exclude, across all splits.

    Returns:
        DataFrame with plant, pollinator columns, len == ratio * len(pos_pairs).
    """
    rng = np.random.default_rng(seed)
    n_target = ratio * len(pos_pairs)
    quotas = {"uniform": {"uniform": n_target}, "cooc": {"cooc": n_target},
              "mixed": {"uniform": n_target // 2, "cooc": n_target - n_target // 2}}[mode]
    plant_pool = np.asarray(plant_pool)
    cooc_plants = pos_pairs["plant"].to_numpy()
    cooc_cache = {}
    out = []

    seen = set()
    for kind, quota in quotas.items():
        taken = 0
        while taken < quota:
            need = quota - taken
            if kind == "uniform":
                pl = rng.choice(plant_pool, size=need * 2)
                po = rng.choice(store.polls, size=need * 2)
                cands = zip(pl, po)
            else:
                pl = rng.choice(cooc_plants, size=need * 2)
                cands = []
                for sp in pl:
                    if sp not in cooc_cache:
                        cooc_cache[sp] = np.flatnonzero(store.N_full[store.p2i[sp]] > 0)
                    idxs = cooc_cache[sp]
                    if len(idxs) == 0:
                        continue
                    cands.append((sp, store.polls[rng.choice(idxs)]))
            for pair in cands:
                if pair not in known_pos and pair not in seen:
                    seen.add(pair)
                    out.append(pair)
                    taken += 1
                    if taken >= quota:
                        break

    return pd.DataFrame(out, columns=["plant", "pollinator"])
