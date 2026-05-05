"""Interface for graph semantic enrichment strategies.

An enrichment strategy receives the list of graphs read from the database and
returns an enriched list (same order, same identities, possibly with extra
node/edge attributes or new nodes/edges that augment the semantic content).

The strategy is run BEFORE the embedding step in the CCluster pipeline:

    read graphs -> enrich semantics -> compute embeddings -> cluster

Implementations should NOT mutate the original graphs in place unless they
explicitly document this behavior; returning the (possibly augmented) graphs
keeps the pipeline composable.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List


@dataclass
class EnrichmentContext:
    """Runtime information passed to an enrichment strategy.

    Attributes
    ----------
    db_graphs:
        The list of graphs read from the database that will be enriched.
    db_file:
        The original database file path (useful for caching or auxiliary IO).
    directed_graph:
        Whether the graph database is composed of directed graphs.
    strategy_params:
        Free-form, strategy-specific parameters (verbose flag, model name,
        thresholds, etc.).
    """

    db_graphs: list
    db_file: str
    directed_graph: bool
    strategy_params: dict = field(default_factory=dict)


class EnrichGraphSemanticsStrategy(ABC):
    """Abstract base class for graph semantic enrichment strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the strategy identifier used by the clustering registry."""

    @abstractmethod
    def enrich(self, context: EnrichmentContext) -> List:
        """Return the (possibly) semantically enriched list of graphs.

        Implementations may add labels, attributes, derived nodes/edges, or
        any other semantic information useful to downstream embedding and
        clustering stages. The returned list MUST preserve the order of
        ``context.db_graphs`` so that name-based lookups remain valid.
        """
