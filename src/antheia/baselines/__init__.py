from antheia.baselines.abundance import AbundanceNeutral
from antheia.baselines.base import Baseline
from antheia.baselines.congeneric import CongenericTransfer
from antheia.baselines.cooccurrence import CoOccurrence
from antheia.baselines.lightfm_hybrid import LightFMHybrid
from antheia.baselines.pair_gbm import PairGBM
from antheia.baselines.phenology_likelihood import PhenologyAbundance
from antheia.baselines.popularity import Popularity
from antheia.baselines.svd_taxonomic import SVDTaxonomic

REGISTRY = {
    "popularity": Popularity,
    "cooccurrence": CoOccurrence,
    "abundance": AbundanceNeutral,
    "congeneric": CongenericTransfer,
    "phenology_abundance": PhenologyAbundance,
    "svd_taxonomic": SVDTaxonomic,
    "pair_gbm": PairGBM,
    "lightfm": LightFMHybrid,
}
