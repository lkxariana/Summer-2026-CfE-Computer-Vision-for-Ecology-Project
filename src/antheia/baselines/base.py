import numpy as np


class Baseline:
    """Common interface for every comparison method.

    A baseline is fitted on the training interactions and then scores, for one plant, every
    pollinator in the candidate universe. Scores need not be probabilities — only their order
    within a plant matters for retrieval, and only their order across pairs matters for PR-AUC.

    Attributes set by subclasses:
        name: label used in results tables.
        reference: citation for the method.
        cold_start: True if the method can score a plant with no training edges. Methods that
            cannot are still reported, restricted to plants seen in training, as an upper
            reference on what an edge-requiring method achieves.
    """

    name = "baseline"
    reference = ""
    cold_start = True

    def fit(self, edges, store):
        """Fit on training interactions.

        Args:
            edges: DataFrame of training interactions with plant/pollinator columns.
            store: FeatureStore giving species order, occupancy, curves and co-occurrence.
        """
        raise NotImplementedError

    def score_plant(self, p):
        """Return a score for every pollinator, given a plant index into `store.polls`."""
        raise NotImplementedError

    def score_pairs(self, pi, qi):
        """Score arbitrary (plant, pollinator) index pairs. Default: gather from per-plant scores."""
        out = np.empty(len(pi), dtype=np.float64)
        for p in np.unique(pi):
            m = pi == p
            out[m] = self.score_plant(int(p))[qi[m]]
        return out


def jitter(scores, seed):
    """Break ties reproducibly so that argpartition does not favour low indices."""
    return scores + np.random.default_rng(seed).uniform(0, 1e-9, len(scores))
