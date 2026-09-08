from antheia.baselines.abundance import AbundanceNeutral
from antheia.baselines.base import Baseline
from antheia.baselines.congeneric import CongenericTransfer
from antheia.baselines.cooccurrence import CoOccurrence
from antheia.baselines.lightfm_hybrid import LightFMHybrid
from antheia.baselines.pair_gbm import PairGBM
from antheia.baselines.phenology_likelihood import PhenologyAbundance
from antheia.baselines.popularity import Popularity
from antheia.baselines.routed import GenusRouted
from antheia.baselines.svd_taxonomic import SVDTaxonomic
from antheia.baselines.taxo_spatial_temporal import TaxoSpatialTemporal
from antheia.neural import NeuralRanker
from antheia.embednet import EmbedRanker
from antheia.rgcn import RGCNRanker
from antheia.baselines.antheia_lr import AntheiaSpatial, AntheiaScalar
from antheia.pairnet import PairRanker

REGISTRY = {
    "popularity": Popularity,
    "antheia_spatial": AntheiaSpatial,
    "antheia_scalar": AntheiaScalar,
    "cooccurrence": CoOccurrence,
    "abundance": AbundanceNeutral,
    "congeneric": CongenericTransfer,
    "phenology_abundance": PhenologyAbundance,
    "svd_taxonomic": SVDTaxonomic,
    "pair_gbm": PairGBM,
    "lightfm": LightFMHybrid,
    "ours_gbm": TaxoSpatialTemporal,
    "routed": GenusRouted,
    "two_tower": lambda **kw: NeuralRanker(use_surface=False, **kw),
    "two_tower_percell": NeuralRanker,
    "two_tower_text": lambda **kw: NeuralRanker(use_surface=False, use_text=True, **kw),
    "pairnet": PairRanker,
    "embednet": EmbedRanker,
    "rgcn": RGCNRanker,
}
