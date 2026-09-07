from pathlib import Path

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

    With `genus_fallback`, family affinity is carried as its own column beside the composite, along
    with an indicator for whether the plant's genus was seen in training. The composite alone weights
    family at 1e-3, which is enough to break ties among candidates but not to carry a prediction: on
    the 12% of held-out plants whose genus never appears in training, this model scores 0.086
    nrecall@10 against 0.256 for truncated SVD with taxonomic imputation and 0.213 for congeneric
    transfer, both of which degrade gracefully. Splitting the signals lets the trees condition on the
    indicator and fall back to family where the genus lookup is empty. Whether a genus was seen is
    known at inference and uses no test label.

    With `use_mf`, the interaction matrix is factorised and the plant's latent vector is imputed by
    taxonomy when the plant is unseen -- its genus mean, else its family mean, else the global mean --
    following Strydom et al. (2022). The affinity table has no such fallback: it returns zeros for an
    unseen genus, which is why this model scores 0.086 nrecall@10 on those plants against 0.256 for
    the factorisation. Carrying the latent product as features rather than routing between models
    keeps one score scale, which pooled PR-AUC depends on.

    With `use_phylo`, taxonomic affinity is smoothed over a dated phylogeny instead of a flat genus
    key: a candidate's score contribution from a training plant is weighted exp(-d/tau) in its
    patristic distance. Genus membership is a two-level key that treats plants in different genera of
    one family as equidistant whether they diverged five or eighty million years ago, and it returns
    exactly zero for a genus absent from training -- the stratum where this model scores 0.086
    nrecall@10 against 0.256 for truncated SVD. A continuous distance degrades instead of vanishing.
    Weights exclude the plant itself, so a training plant never contributes to its own feature.

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
                 device="cuda", residualise=False, local_mode="proj", genus_fallback=True, use_mf=False, mf_rank=12,
                 use_phylo=False, phylo_tau=50.0, **kw):
        self.n_neg, self.seed, self.pca_dim, self.family_weight = n_neg, seed, pca_dim, family_weight
        self.use_local, self.device, self.residualise = use_local, device, residualise
        self.genus_fallback, self.use_mf, self.mf_rank = genus_fallback, use_mf, mf_rank
        self.use_phylo, self.phylo_tau = use_phylo, phylo_tau
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
        extra = ([np.log1p(self.Cf[qi, self.FI[pi]])[:, None],
                  self.seen_genus[pi][:, None].astype(np.float64)]
                 if self.genus_fallback else [])
        if self.use_mf:
            pv = self.P_imp[pi]                                   # [n, rank], taxonomy-imputed
            extra += [pv * self.Q_lat[qi], (pv * self.Q_lat[qi]).sum(1)[:, None]]
        if self.use_phylo:
            extra.append(self.A_phylo[pi, qi][:, None])
        return np.hstack([
            np.log1p(st.Prs[qi])[:, None],
            comp[:, None],
            *extra,
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
        seen = {g for g in gen[store.idx_plants(edges["plant"])]}
        self.seen_genus = np.array([g in seen for g in gen])

        if self.use_mf:
            from scipy.sparse import csr_matrix
            from scipy.sparse.linalg import svds
            pi = store.idx_plants(edges["plant"]); qi = store.idx_polls(edges["pollinator"])
            A = csr_matrix((np.ones(len(pi)), (pi, qi)),
                           shape=(len(store.plants), len(store.polls))).astype(np.float64)
            U, Sg, Vt = svds(A, k=self.mf_rank)
            P_lat, self.Q_lat = (U * Sg)[:, ::-1], Vt[::-1].T
            seen_p = np.zeros(len(store.plants), bool); seen_p[np.unique(pi)] = True
            by_gen = {g: P_lat[seen_p & (gen == g)].mean(0) for g in np.unique(gen[seen_p])}
            by_fam = {ff: P_lat[seen_p & (fam == ff)].mean(0) for ff in np.unique(fam[seen_p])}
            glob = P_lat[seen_p].mean(0)
            # Every plant gets the taxonomy-imputed vector, including plants that are in training.
            # Giving a training plant its own latent -- which encodes its own partners -- while a
            # held-out plant necessarily gets an imputed one is a train/test feature shift, and the
            # model learns to trust a feature that degrades at inference. Genus means are computed
            # leave-one-out for the same reason.
            gsum, gcnt = {}, {}
            for i in np.flatnonzero(seen_p):
                gsum[gen[i]] = gsum.get(gen[i], 0) + P_lat[i]
                gcnt[gen[i]] = gcnt.get(gen[i], 0) + 1
            imp = []
            for i in range(len(store.plants)):
                g = gen[i]
                if g in gcnt and (gcnt[g] - (1 if seen_p[i] else 0)) > 0:
                    tot = gsum[g] - (P_lat[i] if seen_p[i] else 0)
                    imp.append(tot / (gcnt[g] - (1 if seen_p[i] else 0)))
                else:
                    imp.append(by_fam.get(fam[i], glob))
            self.P_imp = np.vstack(imp)

        if self.use_phylo:
            D = np.load(Path(__file__).resolve().parents[3] / "data/features/phylo_dist.npy")
            tr_p = np.unique(store.idx_plants(edges["plant"]))
            K = np.exp(-D[:, tr_p] / self.phylo_tau)              # [P, n_train]
            K[np.isnan(K)] = 0.0
            K[np.arange(len(K))[:, None] == tr_p[None, :]] = 0.0  # never weight a plant by itself
            K /= np.maximum(K.sum(1, keepdims=True), 1e-9)
            M = np.zeros((len(tr_p), len(store.polls)), np.float32)
            pos = {p_: j for j, p_ in enumerate(tr_p)}
            for p_, q_ in zip(store.idx_plants(edges["plant"]), store.idx_polls(edges["pollinator"])):
                M[pos[p_], q_] += 1.0
            self.A_phylo = (K.astype(np.float32) @ M)
            print(f"    phylo affinity: tau={self.phylo_tau} Myr, "
                  f"{int((self.A_phylo > 0).sum()):,} non-zero cells", flush=True)

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
