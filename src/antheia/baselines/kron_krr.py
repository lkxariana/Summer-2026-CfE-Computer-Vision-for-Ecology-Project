import numpy as np

from antheia.baselines.base import Baseline


class TwoStepKronRLS(Baseline):
    """Kernel ridge regression over the Kronecker product of a plant kernel and a pollinator kernel.

    The formalism that matches this evaluation most closely. Pairwise learning with Kronecker
    kernels was developed for drug-target prediction (van Laarhoven et al. 2011) and applied to
    pollination by Stock et al. (2021), who also formalised the four cold-start settings; our
    retrieval task is their Setting B, a held-out left-hand species.

    The two-step form solves ridge regression independently on each side and composes them, which
    avoids ever materialising the (n_plants x n_pollinators) pairwise kernel and gives closed-form
    predictions for unseen species provided the kernels are built from features rather than from
    observed edges.

    Kernels here are cosine similarities over occupancy and over the annual curves; substituting
    an edge-derived kernel would make the method warm-start only.
    """

    name = "Two-step Kronecker kernel ridge regression"
    reference = "Stock et al. 2021, Ecol Modelling 451:109508; van Laarhoven et al. 2011"

    def __init__(self, lam_p=1.0, lam_q=1.0, n_components=256):
        self.lam_p, self.lam_q, self.n_components = lam_p, lam_q, n_components

    @staticmethod
    def _cosine(X):
        X = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
        return X @ X.T

    def fit(self, edges, store):
        self.store = store
        pi, qi = store.idx_plants(edges["plant"]), store.idx_polls(edges["pollinator"])
        Y = np.zeros((len(store.plants), len(store.polls)), dtype=np.float32)
        Y[pi, qi] = 1.0
        # Side kernels from features only, so held-out species remain scorable.
        Kp = self._cosine(np.hstack([store.F.astype(np.float32), store.FC.astype(np.float32)]))
        Kq = self._cosine(np.hstack([store.P.astype(np.float32), store.AC.astype(np.float32)]))
        # Two-step: ridge on the pollinator side, then on the plant side.
        Aq = np.linalg.solve(Kq + self.lam_q * np.eye(len(Kq)), Kq)
        Ap = np.linalg.solve(Kp + self.lam_p * np.eye(len(Kp)), Kp)
        self.pred = Ap @ Y @ Aq.T
        return self

    def score_plant(self, p):
        return self.pred[p].astype(np.float64)
