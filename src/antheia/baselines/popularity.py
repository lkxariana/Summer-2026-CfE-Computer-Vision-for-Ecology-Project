import numpy as np

from antheia.baselines.base import Baseline


class Popularity(Baseline):
    """Rank pollinators by training degree, ignoring the plant entirely.

    The mandatory null. Aiyappa et al. (ICML 2025) show that under uniform negative sampling a
    degree-only predictor is near-optimal on standard link-prediction benchmarks, so any model
    that does not clearly beat this has demonstrated nothing. Because the score is constant
    across plants it is trivially cold-start capable, and its retrieval performance is an upper
    bound on what popularity alone can explain.
    """

    name = "Pollinator popularity"
    reference = "Aiyappa et al. 2025, ICML (arXiv:2405.14985)"

    def fit(self, edges, store):
        self.deg = np.zeros(len(store.polls))
        idx = store.idx_polls(edges["pollinator"])
        np.add.at(self.deg, idx, 1.0)
        return self

    def score_plant(self, p):
        return self.deg
