"""Strategy registry for the CCluster pipeline.

The CCluster pipeline is composed of three pluggable steps:

* ``enrichment``: optional semantic enrichment of the graph database
  (:class:`EnrichGraphSemanticsStrategy`).
* ``embedding``: turns the graphs into a pairwise distance matrix
  (:class:`GraphDistanceStrategy`).
* ``clustering``: partitions the graphs based on the distance matrix
  (:class:`ClusteringStrategy`).

This module re-exports the most commonly used classes from the three
sub-packages so callers can do ``from CCluster.strategies import ...``
without having to know the internal package layout.
"""

from .clustering import (
    ClusteringContext,
    ClusteringStrategy,
    KMedoidsClusteringStrategy,
)
from .embedding import (
    DistanceMatrixStrategyContext,
    FlexibleSubgraphDistanceStrategy,
    GraphDistanceStrategy,
    MCSDistanceStrategy,
    SimpleStructuralDistanceStrategy,
)
from .enrichment import (
    EnrichGraphSemanticsStrategy,
    EnrichmentContext,
    NoOpEnrichmentStrategy,
    SemanticLabelClusteringEnrichmentStrategy,
)

__all__ = [
    # Enrichment
    "EnrichGraphSemanticsStrategy",
    "EnrichmentContext",
    "NoOpEnrichmentStrategy",
    "SemanticLabelClusteringEnrichmentStrategy",
    # Embedding (a.k.a. distance) strategies
    "DistanceMatrixStrategyContext",
    "GraphDistanceStrategy",
    "SimpleStructuralDistanceStrategy",
    "FlexibleSubgraphDistanceStrategy",
    "MCSDistanceStrategy",
    # Clustering algorithms
    "ClusteringContext",
    "ClusteringStrategy",
    "KMedoidsClusteringStrategy",
]
