import numpy as np

from antheia.baselines.base import Baseline
from antheia.baselines.svd_taxonomic import SVDTaxonomic
from antheia.baselines.taxo_spatial_temporal import TaxoSpatialTemporal
from antheia.taxonomy import genus


class GenusRouted(Baseline):
    """Two experts, gated on whether the plant's genus was seen in training.

    Error analysis localised the failure of the boosted ranker to plants whose genus never appears in
    training: 79 of 663 held-out plants, at 0.086 nrecall@10 against 0.351 for the rest. Within that
    stratum it is the weakest of the methods compared, not the strongest -- truncated SVD with
    taxonomic imputation reaches 0.256 and congeneric transfer 0.213 -- because its affinity feature
    returns zeros for an unseen genus while a factorisation imputes the plant's latent vector from its
    genus, family, or the global mean and so degrades rather than collapsing.

    Three attempts to repair this inside one model failed. Carrying family affinity as its own column
    lifts the stratum to 0.122 and nets nothing overall (p=0.97); carrying the factorisation as
    features costs 0.011; smoothing affinity over a dated phylogeny costs 0.018 to 0.029 at every
    bandwidth. Each smoother trades accuracy on the 88% of plants whose genus is seen for a little on
    the tail. Routing keeps the sharp model where it is sharp.

    Whether a genus appeared in training is a property of the training set, known at inference and
    using no test label, so the gate is deployable rather than an oracle.

    Scores from the two experts are not comparable, and splicing them raw collapsed pooled PR-AUC from
    0.10 to 0.02. The fallback's scores are therefore mapped onto the primary's global distribution by
    quantile, fitted on the seen-genus plants where both experts apply, which preserves cross-plant
    ordering. With that, routing improves both objectives.
    """

    name = "Genus-routed experts (ours)"
    reference = "this work"
    cold_start = True

    def __init__(self, primary=None, fallback=None, **kw):
        self.primary = primary or TaxoSpatialTemporal(**kw)
        self.fallback = fallback or SVDTaxonomic()

    def fit(self, edges, store):
        self.store = store
        self.primary.fit(edges, store)
        self.fallback.fit(edges, store)
        seen = {genus(s) for s in set(edges["plant"])}
        self.seen_genus = np.array([genus(s) in seen for s in store.plants])

        # quantile map, fitted only where both experts apply
        rng = np.random.default_rng(0)
        sample = rng.choice(np.flatnonzero(self.seen_genus),
                            min(400, int(self.seen_genus.sum())), replace=False)
        src = np.sort(np.concatenate([self.fallback.score_plant(int(p)) for p in sample]))
        dst = np.sort(np.concatenate([self.primary.score_plant(int(p)) for p in sample]))
        self.src, self.dst = src, dst
        return self

    def score_plant(self, p):
        if self.seen_genus[p]:
            return self.primary.score_plant(p)
        return np.interp(self.fallback.score_plant(p), self.src, self.dst)
