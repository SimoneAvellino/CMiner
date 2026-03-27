from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class DistanceMatrixStrategyContext:
    db_graphs: list
    db_file: str
    directed_graph: bool
    num_clusters: int
    init_method: str
    max_iter: int
    tolerance: float
    strategy_params: dict = field(default_factory=dict)


class GraphDistanceStrategy(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def compute_distance_matrix(self, context: DistanceMatrixStrategyContext):
        """Return a tuple: (distance_matrix, graph_names)."""
        pass
