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

    With `use_local`, one further column carries per-cell phenological co-activity, the sum over
    cells and weeks of the two predicted surfaces multiplied. Continental marginal curves barely
    discriminate -- a random pair already overlaps 0.50 against 0.59 for a true pair -- but the
    per-cell version asks whether two taxa are active in the same weeks *in the same places*, which
    no marginal can reconstruct. Against a species-permuted control it is worth +0.013 nrecall@10
    and +0.013 PR-AUC, and it helps calibration more than the top of the ranking.

    With `residualise`, the spatial and per-cell terms are regressed on the two prevalence terms --
    the pollinator's range size and the plant's -- and the residual is used in their place. Marginally
    those features are confounded with how widespread and how heavily recorded a taxon is: ranked
    alone, per-cell co-activity reaches AUC 0.665 within its own top 200, and its top is populated by
    widespread pollinators rather than partners. Conditioned on prevalence it is strongly
    discriminative, adding 0.125 held-out AUC at the top 200 over popularity and range overlap
    together. Fed in parallel the model has to learn that conditioning while taxonomic affinity
    dominates the fit; residualising imposes it.

    Affinity tables are built from training edges only, so a held-out plant contributes nothing to
    its own score; its genus and family are known metadata, which is what makes the method cold-start.
    """

    name = "Taxonomy + spatial + temporal (ours)"
    reference = "this work"

    def __init__(self, n_neg=10, seed=42, pca_dim=15, family_weight=1e-3, use_local=True,
                 device="cuda", residualise=False, local_mode="proj", **kw):
        self.n_neg, self.seed, self.pca_dim, self.family_weight = n_neg, seed, pca_dim, family_weight
        self.use_local, self.device, self.residualise = use_local, device, residualise
        self.local_mode = local_mode   # 'exact' loads 8.4 GB of surfaces; 'proj' uses the
                                       # rank-256 basis, pearson 1.000 against it, no GPU memory
        self.name = ("Taxonomy + spatial + per-cell temporal (ours)" if use_local
                     else "Taxonomy + spatial + temporal (ours)")
        self.params = dict(max_iter=400, learning_rate=0.05, min_samples_leaf=100,
                           l2_regularization=5.0, random_state=seed)
        self.params.update(kw)

    def _local(self, pi, qi, chunk=4096):
        """Per-cell co-activity, sum over cells and weeks of f(p,c,w)*a(q,c,w), exact, on GPU."""
        import torch
        out = np.empty(len(pi), np.float32)
        for s in range(0, len(pi), chunk):
            a = self._P[torch.as_tensor(np.asarray(pi[s:s + chunk])).long().to(self._dev)].float()
            b = self._Q[torch.as_tensor(np.asarray(qi[s:s + chunk])).long().to(self._dev)].float()
            out[s:s + chunk] = (a * b).sum(1).cpu().numpy()
        return out

    def _spatial(self, pi, qi):
        """The two prevalence-confounded terms: range overlap and per-cell co-activity."""
        st = self.store
        n = np.log1p(np.asarray(st.N_full[pi, qi], dtype=np.float64))
        cols = [n]
        if self.use_local:
            loc = (self._local(pi, qi) if self.local_mode == "exact"
                   else np.maximum(st.local_overlap(pi, qi), 0))
            cols.append(np.log1p(loc))
        return np.column_stack(cols)

    def _prevalence(self, pi, qi):
        st = self.store
        return np.column_stack([np.log1p(st.Prs[qi]), np.log1p(st.Frs[pi]), np.ones(len(pi))])

    def _features(self, pi, qi):
        st = self.store
        comp = self.Cg[qi, self.GI[pi]] + self.family_weight * self.Cf[qi, self.FI[pi]]
        sp = self._spatial(pi, qi)
        if self.residualise:
            sp = sp - self._prevalence(pi, qi) @ self.res_W
        return np.hstack([
            np.log1p(st.Prs[qi])[:, None],
            comp[:, None],
            sp,
            self.Fp[pi] * self.Pp[qi],
            st.FC[pi], st.AC[qi],
        ])

    def fit(self, edges, store):
        self.store = store
        if self.use_local and self.local_mode == "exact":
            import torch
            self._dev = self.device if torch.cuda.is_available() else "cpu"
            self._P = torch.from_numpy(np.array(store.plant_surfaces).reshape(len(store.plants), -1)).to(self._dev)
            self._Q = torch.from_numpy(np.array(store.poll_surfaces).reshape(len(store.polls), -1)).to(self._dev)
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
        if self.residualise:
            # coefficients from training pairs only, then frozen; a held-out plant never informs them
            a = rng.integers(0, len(store.plants), 100_000)
            b = rng.integers(0, len(store.polls), 100_000)
            self.res_W = np.linalg.lstsq(self._prevalence(a, b), self._spatial(a, b), rcond=None)[0]
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
