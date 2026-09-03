import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

from antheia.baselines.base import Baseline
from antheia.taxonomy import family_map, genus


class SVDTaxonomic(Baseline):
    """Low-rank factorisation of the interaction matrix, with latent positions transferred by taxonomy.

    The reference cold-start method in ecology. Strydom et al. (2022) take a truncated SVD of a
    known metaweb, treat the left and right factors as latent traits (a random dot-product graph),
    and infer latent positions for species absent from the training network by ancestral-state
    estimation over a phylogeny. Predicted interactions are then dot products in that space.

    We substitute taxonomic distance for the phylogeny: an unseen plant inherits the mean latent
    position of its congeners, then its confamilials, then the global mean. Rank follows the
    original paper's practice of retaining roughly the leading dozen components.
    """

    name = "Truncated SVD + taxonomic imputation"
    reference = "Strydom et al. 2022, Methods Ecol Evol 13:2308"

    def __init__(self, rank=12):
        self.rank = rank

    def fit(self, edges, store):
        pi, qi = store.idx_plants(edges["plant"]), store.idx_polls(edges["pollinator"])
        A = csr_matrix((np.ones(len(pi)), (pi, qi)),
                       shape=(len(store.plants), len(store.polls))).astype(np.float64)
        U, S, Vt = svds(A, k=self.rank)
        self.P_lat, self.Q_lat = (U * S)[:, ::-1], Vt[::-1].T
        fam = family_map(store.cfg["paths"]["globi"], set(store.plants)) if hasattr(store, "cfg") else {}
        self.p_gen = np.array([genus(s) for s in store.plants])
        self.p_fam = np.array([fam.get(s, "UNK") for s in store.plants])
        seen = np.zeros(len(store.plants), bool)
        seen[np.unique(pi)] = True
        self.by_genus = {g: self.P_lat[seen & (self.p_gen == g)].mean(0)
                         for g in np.unique(self.p_gen[seen])}
        self.by_family = {f: self.P_lat[seen & (self.p_fam == f)].mean(0)
                          for f in np.unique(self.p_fam[seen])}
        self.global_lat = self.P_lat[seen].mean(0)
        self.seen = seen
        return self

    def score_plant(self, p):
        if self.seen[p]:
            v = self.P_lat[p]
        else:
            v = self.by_genus.get(self.p_gen[p],
                                  self.by_family.get(self.p_fam[p], self.global_lat))
        return self.Q_lat @ v
