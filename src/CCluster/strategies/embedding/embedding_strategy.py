"""Interface for graph embedding / distance strategies.

An embedding strategy turns the (possibly enriched) graph database into a
pairwise distance matrix consumed by a clustering algorithm.

The interface is named :class:`GraphDistanceStrategy` for backward
compatibility with the rest of the CCluster code base; conceptually it
represents an "embedding" step that produces a distance matrix.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class DistanceMatrixStrategyContext:
    """Runtime information passed to a distance / embedding strategy."""

    db_graphs: list
    db_file: str
    directed_graph: bool
    num_clusters: int
    init_method: str
    max_iter: int
    tolerance: float
    strategy_params: dict = field(default_factory=dict)


class GraphDistanceStrategy(ABC):
    """Abstract base class for embedding strategies producing distance matrices."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def compute_distance_matrix(self, context: DistanceMatrixStrategyContext):
        """Return a tuple: (distance_matrix, graph_names)."""
        pass
