import numpy as np

from antheia.baselines.base import Baseline
from antheia.taxonomy import family_map, genus


class CongenericTransfer(Baseline):
    """Predict a plant's partners from those of its close relatives.

    What a field ecologist does by hand, and the phylogenetic-signal baseline embedded in
    Strydom et al. (2022) and Stock et al. (2021). Interaction partners are strongly conserved
    within plant genera, so for an unseen plant we take the interaction vector of its congeners,
    falling back to confamilials when the genus is unseen.

    Cold-start capable, needs no fitting beyond counting, and with a median plant degree of a
    few partners it is usually competitive — omitting it is the fastest way to lose a referee.
    """

    name = "Congeneric transfer"
    reference = "phylogenetic-signal baseline; cf. Strydom et al. 2022, Methods Ecol Evol 13:2308"

    def __init__(self, family_weight=1e-3):
        self.family_weight = family_weight

    def fit(self, edges, store):
        self.store = store
        n_q = len(store.polls)
        fam = family_map(store.cfg["paths"]["globi"], set(store.plants)) if hasattr(store, "cfg") else {}
        self.p_gen = np.array([genus(s) for s in store.plants])
        self.p_fam = np.array([fam.get(s, "UNK") for s in store.plants])
        qi = store.idx_polls(edges["pollinator"])
        pi = store.idx_plants(edges["plant"])
        self.by_genus, self.by_family = {}, {}
        for g, q in zip(self.p_gen[pi], qi):
            self.by_genus.setdefault(g, np.zeros(n_q))[q] += 1.0
        for f, q in zip(self.p_fam[pi], qi):
            self.by_family.setdefault(f, np.zeros(n_q))[q] += 1.0
        self.zero = np.zeros(n_q)
        return self

    def score_plant(self, p):
        g = self.by_genus.get(self.p_gen[p])
        f = self.by_family.get(self.p_fam[p], self.zero)
        return f * self.family_weight if g is None else g + self.family_weight * f
