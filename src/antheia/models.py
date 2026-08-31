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
    spec = MODELS[name]
    return lambda pi: pipe.decision_function(store.assemble_plant(pi, spec))
