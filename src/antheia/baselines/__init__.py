from antheia.baselines.abundance import AbundanceNeutral
from antheia.baselines.base import Baseline
from antheia.baselines.congeneric import CongenericTransfer
from antheia.baselines.cooccurrence import CoOccurrence
from antheia.baselines.pair_gbm import PairGBM
from antheia.baselines.phenology_likelihood import PhenologyAbundance
from antheia.baselines.popularity import Popularity
from antheia.baselines.routed import GenusRouted
from antheia.baselines.svd_taxonomic import SVDTaxonomic
from antheia.baselines.taxo_spatial_temporal import TaxoSpatialTemporal
from antheia.models.neural import NeuralRanker
from antheia.models.embednet import EmbedRanker
from antheia.models.rgcn import RGCNRanker
from antheia.baselines.antheia_lr import AntheiaSpatial, AntheiaScalar
from antheia.baselines.nectar_like import NectarLike, NectarLikeUngated
from antheia.models.pairnet import PairRanker

REGISTRY = {
    "popularity": Popularity,
    "antheia_spatial": AntheiaSpatial,
    "antheia_scalar": AntheiaScalar,
    "nectar_like": NectarLike,
    "nectar_ungated": NectarLikeUngated,
    "cooccurrence": CoOccurrence,
    "abundance": AbundanceNeutral,
    "congeneric": CongenericTransfer,
    "phenology_abundance": PhenologyAbundance,
    "svd_taxonomic": SVDTaxonomic,
    "pair_gbm": PairGBM,
    "ours_gbm": TaxoSpatialTemporal,
    "routed": GenusRouted,
    "two_tower": lambda **kw: NeuralRanker(use_surface=False, **kw),
    "two_tower_percell": NeuralRanker,
    "two_tower_text": lambda **kw: NeuralRanker(use_surface=False, use_text=True, **kw),
    "pairnet": PairRanker,
    "embednet": EmbedRanker,
    "rgcn": RGCNRanker,
}
