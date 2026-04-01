"""Graph Edit Distance (GED) strategy.

Pseudocode overview
-------------------

INPUT:
- db_graphs: list of graphs
- GED parameters: timeout, normalize, edit costs

OUTPUT:
- distance_matrix ready for clustering

ALGORITHM compute_distance_matrix(db_graphs, params):
    n <- number_of_graphs(db_graphs)
    distance_matrix <- zero_matrix(n, n)

    for each pair (i, j) with i < j:
        g1 <- db_graphs[i]
        g2 <- db_graphs[j]

        # Use NetworkX GED with custom node/edge similarity
        d <- graph_edit_distance(
                g1, g2,
                node_match=match_node_labels,
                edge_match=match_edge_type,
                node_del_cost=node_cost,
                node_ins_cost=node_cost,
                edge_del_cost=edge_cost,
                edge_ins_cost=edge_cost,
                timeout=timeout
             )

        if d is None:
            d <- fallback_upper_bound(g1, g2, costs)

        if normalize:
            d <- d / normalization_factor(g1, g2, costs)

        distance_matrix[i][j] <- d
        distance_matrix[j][i] <- d

    return distance_matrix
"""

import networkx as nx

from .strategy_interface import DistanceMatrixStrategyContext, GraphDistanceStrategy


class GEDDistanceStrategy(GraphDistanceStrategy):
    @property
    def name(self) -> str:
        return "ged"

    @staticmethod
    def _is_verbose(context: DistanceMatrixStrategyContext) -> bool:
        return bool(context.strategy_params.get("verbose", False))

    @staticmethod
    def _as_float(context: DistanceMatrixStrategyContext, key: str, default: float) -> float:
        value = context.strategy_params.get(key, default)
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid GED parameter '{key}': {value}") from exc

    @staticmethod
    def _labels_signature(node_attrs):
        labels = node_attrs.get("labels", [])
        if labels is None:
            return tuple()
        return tuple(sorted(str(label) for label in labels))

    @staticmethod
    def _node_match(node_a, node_b):
        return GEDDistanceStrategy._labels_signature(node_a) == GEDDistanceStrategy._labels_signature(node_b)

    @staticmethod
    def _edge_match(edge_a, edge_b):
        return str(edge_a.get("type", "")) == str(edge_b.get("type", ""))

    @staticmethod
    def _fallback_upper_bound(g1, g2, node_del_cost, node_ins_cost, edge_del_cost, edge_ins_cost):
        return (
            g1.number_of_nodes() * node_del_cost
            + g2.number_of_nodes() * node_ins_cost
            + g1.number_of_edges() * edge_del_cost
            + g2.number_of_edges() * edge_ins_cost
        )

    @staticmethod
    def _normalization_factor(g1, g2, node_del_cost, node_ins_cost, edge_del_cost, edge_ins_cost):
        return GEDDistanceStrategy._fallback_upper_bound(
            g1,
            g2,
            node_del_cost,
            node_ins_cost,
            edge_del_cost,
            edge_ins_cost,
        )

    def _graph_edit_distance(self, g1, g2, timeout, node_del_cost, node_ins_cost, edge_del_cost, edge_ins_cost):
        return nx.graph_edit_distance(
            g1,
            g2,
            node_match=self._node_match,
            edge_match=self._edge_match,
            node_del_cost=lambda _attrs: node_del_cost,
            node_ins_cost=lambda _attrs: node_ins_cost,
            edge_del_cost=lambda _attrs: edge_del_cost,
            edge_ins_cost=lambda _attrs: edge_ins_cost,
            timeout=timeout,
        )

    def compute_distance_matrix(self, context: DistanceMatrixStrategyContext):
        verbose = self._is_verbose(context)

        timeout = context.strategy_params.get("ged_timeout", None)
        if timeout is not None:
            timeout = self._as_float(context, "ged_timeout", 0.0)
            if timeout <= 0:
                timeout = None

        normalize = bool(context.strategy_params.get("ged_normalize", True))
        node_del_cost = self._as_float(context, "ged_node_del_cost", 1.0)
        node_ins_cost = self._as_float(context, "ged_node_ins_cost", 1.0)
        edge_del_cost = self._as_float(context, "ged_edge_del_cost", 1.0)
        edge_ins_cost = self._as_float(context, "ged_edge_ins_cost", 1.0)

        if min(node_del_cost, node_ins_cost, edge_del_cost, edge_ins_cost) < 0:
            raise ValueError("GED costs must be non-negative.")

        graphs = context.db_graphs
        graph_names = [graph.get_name() for graph in graphs]
        n_graphs = len(graphs)
        distance_matrix = [[0.0 for _ in range(n_graphs)] for _ in range(n_graphs)]

        if verbose:
            print(
                "[ged] Computing pairwise Graph Edit Distance "
                f"for {n_graphs} graphs..."
            )

        total_pairs = (n_graphs * (n_graphs - 1)) // 2
        pair_index = 0

        for i in range(n_graphs):
            for j in range(i + 1, n_graphs):
                pair_index += 1
                if verbose:
                    print(
                        "[ged] "
                        f"Pair {pair_index}/{total_pairs}: {graph_names[i]} vs {graph_names[j]}"
                    )

                raw_distance = self._graph_edit_distance(
                    graphs[i],
                    graphs[j],
                    timeout,
                    node_del_cost,
                    node_ins_cost,
                    edge_del_cost,
                    edge_ins_cost,
                )

                if raw_distance is None:
                    raw_distance = self._fallback_upper_bound(
                        graphs[i],
                        graphs[j],
                        node_del_cost,
                        node_ins_cost,
                        edge_del_cost,
                        edge_ins_cost,
                    )

                distance = float(raw_distance)
                if normalize:
                    normalizer = self._normalization_factor(
                        graphs[i],
                        graphs[j],
                        node_del_cost,
                        node_ins_cost,
                        edge_del_cost,
                        edge_ins_cost,
                    )
                    if normalizer > 0:
                        distance /= normalizer
                    else:
                        distance = 0.0

                distance_matrix[i][j] = distance
                distance_matrix[j][i] = distance

        if verbose:
            print("[ged] Distance matrix completed.")

        return distance_matrix, graph_names
