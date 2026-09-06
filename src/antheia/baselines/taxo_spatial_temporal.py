import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier

from antheia.baselines.base import Baseline
from antheia.taxonomy import genus

EPS = 1e-12


class TaxoSpatialTemporal(Baseline):
    """Taxonomic affinity, spatial co-occurrence and phenology in one pair ranker.

    Taxonomic affinity enters as a single composite, `genus_count + 1e-3 * family_count`, not as two
    columns. The two carry the same information but not the same ordering: a plant's genus has only a
    handful of distinct affinity values across the candidate set, so most candidates tie on it, and
    the family term resolves those ties. That ordering is lexicographic, and a tree ensemble cannot
    express it -- boosting on the two columns separately yields ~38 distinct scores per plant over
    13,124 candidates and loses most of the family signal (0.254 against 0.293 for the composite
    scored directly). Pre-combining hands the ordering to the model intact.

    Affinity tables are built from training edges only, so a held-out plant contributes nothing to
    its own score; its genus and family are known metadata, which is what makes the method cold-start.
    """

    name = "Taxonomy + spatial + temporal (ours)"
    reference = "this work"

    def __init__(self, n_neg=10, seed=42, pca_dim=15, family_weight=1e-3, **kw):
        self.n_neg, self.seed, self.pca_dim, self.family_weight = n_neg, seed, pca_dim, family_weight
        self.params = dict(max_iter=400, learning_rate=0.05, min_samples_leaf=100,
                           l2_regularization=5.0, random_state=seed)
        self.params.update(kw)

    def _features(self, pi, qi):
        st = self.store
        comp = self.Cg[qi, self.GI[pi]] + self.family_weight * self.Cf[qi, self.FI[pi]]
        return np.hstack([
            np.log1p(st.Prs[qi])[:, None],
            comp[:, None],
            np.log1p(np.asarray(st.N_full[pi, qi], dtype=np.float64))[:, None],
            self.Fp[pi] * self.Pp[qi],
            st.FC[pi], st.AC[qi],
        ])

    def fit(self, edges, store):
        self.store = store
        gen = np.array([genus(s) for s in store.plants])
        fam = np.array([store.family.get(s, "UNK") for s in store.plants])
        gi = {g: i for i, g in enumerate(sorted(set(gen)))}
        fi = {f: i for i, f in enumerate(sorted(set(fam)))}
        self.GI = np.array([gi[g] for g in gen])
        self.FI = np.array([fi[f] for f in fam])
        self.Cg = np.zeros((len(store.polls), len(gi)), np.float32)
        self.Cf = np.zeros((len(store.polls), len(fi)), np.float32)
        for pl, po in zip(edges["plant"], edges["pollinator"]):
            q = store.q2i[po]
            self.Cg[q, self.GI[store.p2i[pl]]] += 1
            self.Cf[q, self.FI[store.p2i[pl]]] += 1

        f = TruncatedSVD(self.pca_dim, random_state=self.seed).fit_transform(store.F.astype(np.float32))
        p = TruncatedSVD(self.pca_dim, random_state=self.seed).fit_transform(store.P.astype(np.float32))
        self.Fp = f / (np.linalg.norm(f, axis=1, keepdims=True) + EPS)
        self.Pp = p / (np.linalg.norm(p, axis=1, keepdims=True) + EPS)

        rng = np.random.default_rng(self.seed)
        pi, qi = store.idx_plants(edges["plant"]), store.idx_polls(edges["pollinator"])
        known = set(zip(pi.tolist(), qi.tolist()))
        neg_p = np.repeat(pi, self.n_neg)
        neg_q = rng.integers(0, len(store.polls), len(neg_p))
        keep = [i for i, (a, b) in enumerate(zip(neg_p, neg_q)) if (a, b) not in known]
        neg_p, neg_q = neg_p[keep], neg_q[keep]
        X = np.vstack([self._features(pi, qi), self._features(neg_p, neg_q)])
        y = np.concatenate([np.ones(len(pi)), np.zeros(len(neg_p))])
        self.clf = HistGradientBoostingClassifier(**self.params).fit(X, y)
        return self

    def score_plant(self, p):
        n_q = len(self.store.polls)
        return self.clf.predict_proba(self._features(np.full(n_q, p), np.arange(n_q)))[:, 1]
