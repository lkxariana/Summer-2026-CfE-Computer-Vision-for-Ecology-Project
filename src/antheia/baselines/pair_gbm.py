import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from antheia.baselines.base import Baseline
from antheia import negpool


class PairGBM(Baseline):
    """Gradient boosting over concatenated plant and pollinator features.

    The machine-learning-in-ecology reference point. Pichler, Boreux, Klein, Schleuning & Hartig
    (2020) compare GLMs, random forests, boosted trees, SVMs and neural networks for trait
    matching, treating every plant-pollinator pair as a row whose features are the two species'
    traits plus their interactions, and evaluate under species-blocked cross-validation.

    We keep the framework and substitute the features available at continental scale — occupancy
    summary, annual curves, taxonomic affinity and co-occurrence — for the morphological traits
    that do not exist for more than a few dozen species. Trees supply the feature interactions
    that a linear model cannot express.
    """

    name = "Gradient boosting on pair features"
    reference = "Pichler et al. 2020, Methods Ecol Evol 11:281"

    def __init__(self, n_neg=10, seed=42, **kw):
        self.n_neg, self.seed = n_neg, seed
        self.params = dict(max_iter=400, learning_rate=0.05, min_samples_leaf=100,
                           l2_regularization=5.0, random_state=seed)
        self.params.update(kw)

    def _features(self, pi, qi):
        st = self.store
        return np.hstack([
            st.FC[pi], st.AC[qi],
            np.log1p(st.N_full[pi, qi].astype(np.float64))[:, None],
            np.log1p(st.Frs[pi])[:, None], np.log1p(st.Prs[qi])[:, None],
            np.minimum(st.FC[pi], st.AC[qi]).sum(1)[:, None],
        ])

    def fit(self, edges, store):
        self.store = store
        rng = np.random.default_rng(self.seed)
        pi, qi = store.idx_plants(edges["plant"]), store.idx_polls(edges["pollinator"])
        known = set(zip(pi.tolist(), qi.tolist()))
        neg_p = np.repeat(pi, self.n_neg)
        neg_q = negpool.sample(rng, len(neg_p), len(store.polls))
        keep = [i for i, (a, b) in enumerate(zip(neg_p, neg_q)) if (a, b) not in known]
        neg_p, neg_q = neg_p[keep], neg_q[keep]
        X = np.vstack([self._features(pi, qi), self._features(neg_p, neg_q)])
        y = np.concatenate([np.ones(len(pi)), np.zeros(len(neg_p))])
        self.clf = HistGradientBoostingClassifier(**self.params).fit(X, y)
        return self

    def score_plant(self, p):
        n_q = len(self.store.polls)
        pi = np.full(n_q, p)
        return self.clf.predict_proba(self._features(pi, np.arange(n_q)))[:, 1]
