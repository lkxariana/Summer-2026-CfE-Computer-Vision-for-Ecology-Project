import numpy as np

from antheia.baselines.base import Baseline


class AbundanceNeutral(Baseline):
    """Neutral model: interaction probability proportional to the product of species abundances.

    Vázquez, Chacoff & Cagnolo (2009) formalised the idea that interaction frequency can be
    explained by encounter rates alone, with no trait or preference term. Dormann et al. (2025)
    find this parameter-free benchmark dominates traits and phylogeny across fourteen networks,
    which makes it the strongest non-learned hypothesis in the field.

    Within a single plant the plant's own abundance is constant, so the ranking reduces to
    pollinator abundance. We therefore use observation counts rather than training degree —
    degree is unavailable for a held-out plant, and using it would leak the label.
    """

    name = "Abundance neutral model"
    reference = "Vázquez, Chacoff & Cagnolo 2009, Ecology 90:2039"

    def fit(self, edges, store):
        self.abund = store.Prs.astype(np.float64)          # occupied cells as an abundance proxy
        return self

    def score_plant(self, p):
        return self.abund
