import math

from .strategy_interface import DistanceMatrixStrategyContext, GraphDistanceStrategy


class SimpleStructuralDistanceStrategy(GraphDistanceStrategy):
    @property
    def name(self) -> str:
        return "simple_structural"

    @staticmethod
    def _safe_avg_degree(graph):
        n_nodes = graph.number_of_nodes()
        if n_nodes == 0:
            return 0.0
        return (2.0 * graph.number_of_edges()) / n_nodes

    @staticmethod
    def _graph_feature_vector(graph):
        node_labels = set()
        for _, node_data in graph.nodes(data=True):
            node_labels.update(node_data.get("labels", []))

        edge_labels = set()
        for _, _, _, edge_data in graph.edges(keys=True, data=True):
            edge_type = edge_data.get("type", "")
            if edge_type:
                edge_labels.add(edge_type)

        return [
            float(graph.number_of_nodes()),
            float(graph.number_of_edges()),
            float(len(node_labels)),
            float(len(edge_labels)),
            float(SimpleStructuralDistanceStrategy._safe_avg_degree(graph)),
        ]

    @staticmethod
    def _normalize_feature_vectors(feature_vectors):
        if not feature_vectors:
            return []

        dim = len(feature_vectors[0])
        maxima = [0.0] * dim
        for vector in feature_vectors:
            for idx, value in enumerate(vector):
                if value > maxima[idx]:
                    maxima[idx] = value

        normalized = []
        for vector in feature_vectors:
            normalized.append(
                [
                    (value / maxima[idx]) if maxima[idx] > 0 else 0.0
                    for idx, value in enumerate(vector)
                ]
            )
        return normalized

    @staticmethod
    def _euclidean_distance(vector_a, vector_b):
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(vector_a, vector_b)))

    def compute_distance_matrix(self, context: DistanceMatrixStrategyContext):
        graphs = context.db_graphs
        graph_names = [graph.get_name() for graph in graphs]

        feature_vectors = [self._graph_feature_vector(graph) for graph in graphs]
        normalized_vectors = self._normalize_feature_vectors(feature_vectors)

        n_graphs = len(normalized_vectors)
        distance_matrix = [[0.0 for _ in range(n_graphs)] for _ in range(n_graphs)]

        for i in range(n_graphs):
            for j in range(i + 1, n_graphs):
                distance = self._euclidean_distance(
                    normalized_vectors[i], normalized_vectors[j]
                )
                distance_matrix[i][j] = distance
                distance_matrix[j][i] = distance

        return distance_matrix, graph_names
