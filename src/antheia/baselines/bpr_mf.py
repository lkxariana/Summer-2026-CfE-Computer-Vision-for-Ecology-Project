import numpy as np

from antheia.baselines.base import Baseline


class BPRMatrixFactorisation(Baseline):
    """Bayesian personalised ranking over free per-species embeddings. Transductive.

    Included as an upper reference, not a competitor. BPR (Rendle et al. 2009) optimises the
    pairwise ordering of an observed interaction above an unobserved one, learning a free latent
    vector for every node. Because a held-out plant has no vector, the method cannot score it at
    all — which is precisely the limitation that motivates feature-based approaches in this
    setting, so we report it on the warm subset to bound what an edge-requiring method achieves.

    Seo & Hutchinson (2018) applied this family to plant-pollinator data with implicit-feedback
    weighting, the closest prior framing to ours.
    """

    name = "Matrix factorisation (BPR)"
    reference = "Rendle et al. 2009, UAI; cf. Seo & Hutchinson 2018, AAAI-18"
    cold_start = False

    def __init__(self, factors=64, iterations=100, seed=42):
        self.factors, self.iterations, self.seed = factors, iterations, seed

    def fit(self, edges, store):
        from implicit.bpr import BayesianPersonalizedRanking
        from scipy.sparse import coo_matrix
        self.store = store
        pi, qi = store.idx_plants(edges["plant"]), store.idx_polls(edges["pollinator"])
        self.seen = np.zeros(len(store.plants), bool)
        self.seen[np.unique(pi)] = True
        m = coo_matrix((np.ones(len(pi), dtype=np.float32), (pi, qi)),
                       shape=(len(store.plants), len(store.polls))).tocsr()
        self.model = BayesianPersonalizedRanking(factors=self.factors, iterations=self.iterations,
                                                 random_state=self.seed)
        self.model.fit(m, show_progress=False)
        return self

    def score_plant(self, p):
        if not self.seen[p]:
            return np.zeros(len(self.store.polls))      # undefined for an unseen plant
        return self.model.user_factors[p] @ self.model.item_factors.T
