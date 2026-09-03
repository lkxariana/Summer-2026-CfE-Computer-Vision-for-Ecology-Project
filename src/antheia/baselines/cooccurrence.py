import numpy as np

from antheia.baselines.base import Baseline


class CoOccurrence(Baseline):
    """Rank pollinators by the number of grid cells they share with the plant.

    The encounter-opportunity null: two species cannot interact where they never co-occur, so
    shared range is a necessary condition. It is emphatically not sufficient — Blanchet, Cazelles
    & Gravel (2020) show co-occurrence is not evidence of interaction — which is exactly why it
    belongs in every table as the quantity a model must beat rather than reproduce.

    Uses no training edges at all, so it is unaffected by the split.
    """

    name = "Co-occurrence count"
    reference = "Blanchet, Cazelles & Gravel 2020, Ecology Letters 23:1050"

    def fit(self, edges, store):
        self.store = store
        return self

    def score_plant(self, p):
        return self.store.N_full[p].astype(np.float64)
