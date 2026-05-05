"""Interface for clustering algorithms operating on a precomputed distance matrix.

A clustering strategy receives the pairwise distance matrix produced by the
embedding step and partitions the graphs into ``num_clusters`` groups.

The interface is deliberately minimal: it returns both the human-readable
``clusters`` mapping (cluster_id -> list of graph names) and the raw
``assignments`` (list of lists of indices) so the orchestrator can compute
metrics like the silhouette score on the same partition.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class ClusteringContext:
    """Runtime information passed to a clustering strategy."""

    num_clusters: int
    init_method: str = "random"
    max_iter: int = 100
    tolerance: float = 1e-4
    strategy_params: dict = field(default_factory=dict)


class ClusteringStrategy(ABC):
    """Abstract base class for distance-matrix-based clustering algorithms."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the strategy identifier used by the clustering registry."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Return a human-readable description of what this strategy does."""

    @abstractmethod
    def cluster(
        self,
        distance_matrix,
        graph_names: List[str],
        context: ClusteringContext,
    ) -> Tuple[Dict[int, List[str]], List[List[int]]]:
        """Partition the graphs into clusters.

        Parameters
        ----------
        distance_matrix:
            Symmetric pairwise distance matrix (list-of-lists or 2D array).
        graph_names:
            Names of the graphs, aligned with the rows/columns of the matrix.
        context:
            Algorithm-agnostic configuration (target number of clusters,
            initialization method, iteration limits, ...).

        Returns
        -------
        clusters, assignments:
            ``clusters`` maps the cluster id to the list of graph names;
            ``assignments`` lists, for each cluster id, the integer indices
            of the assigned points (so callers can compute metrics like the
            silhouette score without re-deriving the partition).
        """
