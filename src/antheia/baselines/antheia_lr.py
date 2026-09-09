"""The original ANTHEIA link predictors: logistic regression on occupancy PCA and co-occurrence.

Li, Cher & Jacobs (2026, README ANTHEIA2026.07): each species is a 15-D PCA of its 0.5-degree
occupancy grid (V_f for plants, V_p for pollinators); a pair is [V_f, V_p, N] with N the number of
shared occupied cells; ANTHEIA-Scalar adds the phenological overlap Delta = sum_t min(f_t, a_t).
Trained pointwise with negatives sampled 3:1 per seed, as in the original notebooks. Re-implemented
here on the corrected network and the frozen cold-plant split; the PCA is fit on the training plants
and all pollinators, so a held-out plant's coordinates are a projection, not a refit.
"""
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from antheia.baselines.base import Baseline
from antheia import negpool


class AntheiaSpatial(Baseline):
    name = "ANTHEIA spatial baseline (PCA-15 + N, logistic)"
    reference = "Li, Cher & Jacobs 2026 (this project, v1)"
    use_delta = False

    def __init__(self, n_neg=3, seed=42, dims=15, **kw):
        self.n_neg, self.seed, self.dims = n_neg, seed, dims

    def _pair_feats(self, pi, qi):
        st = self.store
        cols = [self.Vf[pi], self.Vp[qi], np.log1p(st.N_full[pi, qi].astype(np.float64))[:, None]]
        if self.use_delta:
            cols.append(np.minimum(st.FC[pi], st.AC[qi]).sum(1)[:, None])
        return np.hstack(cols)

    def fit(self, edges, store):
        self.store = store
        rng = np.random.default_rng(self.seed)
        pi = store.idx_plants(edges["plant"]); qi = store.idx_polls(edges["pollinator"])
        train_plants = np.unique(pi)
        svd_f = TruncatedSVD(self.dims, random_state=self.seed).fit(store.F[train_plants].astype(np.float32))
        self.Vf = svd_f.transform(store.F.astype(np.float32))
        self.Vp = TruncatedSVD(self.dims, random_state=self.seed).fit_transform(store.P.astype(np.float32))
        pos = set(zip(pi.tolist(), qi.tolist()))
        n_neg = self.n_neg * len(pi)
        np_, nq_ = rng.choice(train_plants, n_neg), negpool.sample(rng, n_neg, len(store.polls))
        keep = np.fromiter(((a, b) not in pos for a, b in zip(np_.tolist(), nq_.tolist())), bool, n_neg)
        X = np.vstack([self._pair_feats(pi, qi), self._pair_feats(np_[keep], nq_[keep])])
        y = np.r_[np.ones(len(pi)), np.zeros(int(keep.sum()))]
        self.scaler = StandardScaler().fit(X)
        self.clf = LogisticRegression(C=1.0, max_iter=2000).fit(self.scaler.transform(X), y)
        return self

    def score_plant(self, p):
        n_q = len(self.store.polls)
        X = self._pair_feats(np.full(n_q, p), np.arange(n_q))
        return self.clf.decision_function(self.scaler.transform(X))


class AntheiaScalar(AntheiaSpatial):
    name = "ANTHEIA-Scalar (PCA-15 + N + Delta, logistic)"
    use_delta = True
