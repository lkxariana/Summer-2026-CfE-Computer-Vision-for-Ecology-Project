import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODELS = {
    "n_only": ("n",),
    "range_only": ("frs", "prs"),
    "spatial": ("vf", "vp", "n"),
    "scalar": ("vf", "vp", "n", "delta"),
    "antheia_4d": ("vf", "vp", "n", "vd4"),
    "antheia_15d": ("vf", "vp", "n", "vd15"),
    "pmf": ("vfp", "vp", "n"),
    "scalar_local": ("vf", "vp", "n", "delta_local"),
}

# Ranking specs use only pollinator-varying components: plant-side features cancel
# in within-plant score differences under a linear scorer.
RANK_SPECS = {
    "rank_n": ("n",),
    "rank_n_delta": ("n", "delta"),
    "rank_spatial": ("vp", "n"),
    "rank_scalar": ("vp", "n", "delta"),
    "rank_full": ("vp", "n", "delta", "prs"),
    "rank_ldelta": ("delta_local",),
    "rank_n_ldelta": ("n", "delta_local"),
    "rank_scalar_local": ("vp", "n", "delta_local"),
    "rank_full_local": ("vp", "n", "delta", "delta_local", "prs"),
}


def train(store, name, pairs, y, seed):
    """Fits the standardized logistic-regression probe for a named model on pair rows."""
    pi = store.idx_plants(pairs["plant"])
    qi = store.idx_polls(pairs["pollinator"])
    X = store.assemble(pi, qi, MODELS[name])
    pipe = Pipeline([("scale", StandardScaler()),
                     ("lr", LogisticRegression(max_iter=1000, random_state=seed))])
    pipe.fit(X, y)
    return pipe


def pair_scores(store, name, pipe, pairs):
    pi = store.idx_plants(pairs["plant"])
    qi = store.idx_polls(pairs["pollinator"])
    return pipe.decision_function(store.assemble(pi, qi, MODELS[name]))


def plant_scorer(store, name, pipe):
    """Returns callable(plant_index) -> decision scores over every pollinator."""
    return spec_scorer(store, MODELS[name], pipe)


def spec_scorer(store, spec, pipe):
    return lambda pi: pipe.decision_function(store.assemble_plant(pi, spec))


def rank_contrast_sets(pos_pairs, store, j, seed, known_pos):
    """Builds within-plant contrast index arrays for pairwise ranking training.

    For each positive (plant, q+), samples j negative candidate pollinators for the same plant
    (half uniform, half among the plant's co-occurring N>0 pollinators), excluding documented pairs.

    Returns:
        (pi, qi_pos, qi_neg) index arrays, each of length len(pos_pairs) * j.
    """
    rng = np.random.default_rng(seed)
    n_po = len(store.polls)
    cooc_cache = {}
    pi, qi_pos, qi_neg = [], [], []
    for plant, q_pos in zip(pos_pairs["plant"], pos_pairs["pollinator"]):
        p = store.p2i[plant]
        if p not in cooc_cache:
            cooc_cache[p] = np.flatnonzero(store.N_full[p] > 0)
        cooc = cooc_cache[p]
        qp = store.q2i[q_pos]
        got = 0
        while got < j:
            use_cooc = (got % 2 == 0) and len(cooc) > 0
            q = int(cooc[rng.integers(len(cooc))]) if use_cooc else int(rng.integers(n_po))
            if q != qp and (plant, store.polls[q]) not in known_pos:
                pi.append(p)
                qi_pos.append(qp)
                qi_neg.append(q)
                got += 1
    return np.array(pi), np.array(qi_pos), np.array(qi_neg)


def train_pairwise(store, spec, pi, qi_pos, qi_neg, seed):
    """Fits a linear pairwise ranker (logistic regression on within-plant feature differences)."""
    d = store.assemble(pi, qi_pos, spec) - store.assemble(pi, qi_neg, spec)
    X = np.vstack([d, -d])
    y = np.concatenate([np.ones(len(d)), np.zeros(len(d))])
    pipe = Pipeline([("scale", StandardScaler(with_mean=False)),
                     ("lr", LogisticRegression(max_iter=1000, random_state=seed, fit_intercept=False))])
    pipe.fit(X, y)
    return pipe
