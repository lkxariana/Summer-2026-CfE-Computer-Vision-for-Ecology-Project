import numpy as np

from antheia.baselines.base import Baseline


class PhenologyAbundance(Baseline):
    """Multiply candidate probability matrices: encounter opportunity times phenological overlap.

    The ecological ancestor of the phenology term this paper is testing. Vizentin-Bugoni,
    Maruyama & Sazima (2014) build separate probability matrices for abundance, morphology and
    phenology, multiply them elementwise, and rank models by likelihood; they conclude that
    phenological and morphological constraints matter more than abundance. Bartomeus et al.
    (2016) give the same decomposition as a general linkage-rule framework.

    We reproduce the abundance x phenology form, which is what our data supports: no morphology
    term exists at this scale. The phenology matrix is the overlap of the two normalised annual
    curves, i.e. the hand-computed scalar our learned encodings are compared against.
    """

    name = "Phenology x abundance likelihood"
    reference = "Vizentin-Bugoni, Maruyama & Sazima 2014, Proc R Soc B 281:20132397"

    def fit(self, edges, store):
        self.store = store
        self.abund = store.Prs.astype(np.float64)
        self.abund = self.abund / self.abund.sum()
        return self

    def score_plant(self, p):
        st = self.store
        f = st.FC[p] / max(st.FC[p].sum(), 1e-12)
        a = st.AC / np.clip(st.AC.sum(1, keepdims=True), 1e-12, None)
        overlap = np.minimum(f[None, :], a).sum(1)          # shared area under the two curves
        return overlap * self.abund
