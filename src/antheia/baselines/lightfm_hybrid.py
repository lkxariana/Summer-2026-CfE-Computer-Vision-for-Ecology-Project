import numpy as np
from scipy.sparse import coo_matrix, hstack, identity

from antheia.baselines.base import Baseline


class LightFMHybrid(Baseline):
    """Hybrid factorisation where a species' embedding is the sum of its feature embeddings.

    The canonical cold-start recommender, and the baseline an ML reviewer will expect. LightFM
    (Kula 2015) represents each node as the sum of latent vectors of its content features rather
    than as a free per-node vector, so a species never seen in training still receives an
    embedding through its features. Trained with WARP, which samples negatives until one violates
    the margin and therefore optimises the top of the ranking directly.

    Features here are taxonomic indicators plus discretised occupancy and phenology, since LightFM
    expects sparse categorical input. Requires the `lightfm` package; skipped if unavailable.
    """

    name = "LightFM (WARP)"
    reference = "Kula 2015, arXiv:1507.08439"

    def __init__(self, components=64, epochs=30, seed=42, n_bins=16):
        self.components, self.epochs, self.seed, self.n_bins = components, epochs, seed, n_bins

    def _side_features(self, dense, n):
        """Identity block plus quantile-binned dense features, as a sparse indicator matrix."""
        blocks = [identity(n, format="csr")]
        for col in range(dense.shape[1]):
            v = dense[:, col]
            edges_ = np.quantile(v, np.linspace(0, 1, self.n_bins + 1)[1:-1])
            b = np.digitize(v, edges_)
            blocks.append(coo_matrix((np.ones(n), (np.arange(n), b)),
                                     shape=(n, self.n_bins)).tocsr())
        return hstack(blocks, format="csr")

    def fit(self, edges, store):
        from lightfm import LightFM
        self.store = store
        pi, qi = store.idx_plants(edges["plant"]), store.idx_polls(edges["pollinator"])
        n_p, n_q = len(store.plants), len(store.polls)
        inter = coo_matrix((np.ones(len(pi)), (pi, qi)), shape=(n_p, n_q))
        self.pf = self._side_features(np.hstack([store.FC, np.log1p(store.Frs)[:, None]]), n_p)
        self.qf = self._side_features(np.hstack([store.AC, np.log1p(store.Prs)[:, None]]), n_q)
        self.model = LightFM(no_components=self.components, loss="warp", random_state=self.seed)
        self.model.fit(inter, user_features=self.pf, item_features=self.qf,
                       epochs=self.epochs, num_threads=8)
        return self

    def score_plant(self, p):
        n_q = len(self.store.polls)
        return self.model.predict(np.full(n_q, p, dtype=np.int32), np.arange(n_q, dtype=np.int32),
                                  user_features=self.pf, item_features=self.qf, num_threads=8)
