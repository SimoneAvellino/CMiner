import networkx as nx

from .strategy_interface import DistanceMatrixStrategyContext, GraphDistanceStrategy


class MCSDistanceStrategy(GraphDistanceStrategy):
    @property
    def name(self) -> str:
        return "mcs"

    @staticmethod
    def _is_verbose(context: DistanceMatrixStrategyContext) -> bool:
        return bool(context.strategy_params.get("verbose", False))

    @staticmethod
    def _use_mcis(context: DistanceMatrixStrategyContext) -> bool:
        return bool(context.strategy_params.get("mcs_use_mcis", False))

    @staticmethod
    def _labels_signature(node_attrs):
        labels = node_attrs.get("labels", [])
        if labels is None:
            return tuple()
        return tuple(sorted(str(label) for label in labels))

    @classmethod
    def _nodes_compatible(cls, graph_a, node_a, graph_b, node_b):
        return cls._labels_signature(graph_a.nodes[node_a]) == cls._labels_signature(
            graph_b.nodes[node_b]
        )

    @staticmethod
    def _edge_type_multiset(graph, source, target):
        edge_bundle = graph.get_edge_data(source, target, default={})
        if not edge_bundle:
            return tuple()

        if isinstance(edge_bundle, dict) and "type" in edge_bundle:
            return (str(edge_bundle.get("type", "")),)

        labels = []
        for edge_data in edge_bundle.values():
            labels.append(str(edge_data.get("type", "")))
        labels.sort()
        return tuple(labels)

    @classmethod
    def _relation_signature(cls, graph, node_u, node_v):
        if graph.is_directed():
            return (
                cls._edge_type_multiset(graph, node_u, node_v),
                cls._edge_type_multiset(graph, node_v, node_u),
            )
        return cls._edge_type_multiset(graph, node_u, node_v)

    @staticmethod
    def _normalized_similarity(common_size, graph_a, graph_b):
        denominator = min(graph_a.number_of_nodes(), graph_b.number_of_nodes())
        if denominator == 0:
            return 1.0
        return float(common_size) / float(denominator)

    @classmethod
    def _mcs_relation_compatible(cls, graph_a, node_a1, node_a2, graph_b, node_b1, node_b2):
        relation_a = cls._relation_signature(graph_a, node_a1, node_a2)
        relation_b = cls._relation_signature(graph_b, node_b1, node_b2)

        if graph_a.is_directed():
            forward_a, _ = relation_a
            forward_b, _ = relation_b
            return forward_a == forward_b and len(forward_a) > 0

        return relation_a == relation_b and len(relation_a) > 0

    def _association_graph(self, graph_a, graph_b):
        association = nx.Graph()
        candidates = []

        for node_a in graph_a.nodes():
            for node_b in graph_b.nodes():
                if self._nodes_compatible(graph_a, node_a, graph_b, node_b):
                    pair = (node_a, node_b)
                    association.add_node(pair)
                    candidates.append(pair)

        for idx in range(len(candidates)):
            node_a1, node_b1 = candidates[idx]
            for jdx in range(idx + 1, len(candidates)):
                node_a2, node_b2 = candidates[jdx]
                if node_a1 == node_a2 or node_b1 == node_b2:
                    continue
                if self._mcs_relation_compatible(
                    graph_a, node_a1, node_a2, graph_b, node_b1, node_b2
                ):
                    association.add_edge((node_a1, node_b1), (node_a2, node_b2))

        return association

    @staticmethod
    def _maximum_clique_size(association_graph):
        best_size = 0
        for clique in nx.find_cliques(association_graph):
            if len(clique) > best_size:
                best_size = len(clique)
        return best_size

    def _mcs_size(self, graph_a, graph_b):
        association_graph = self._association_graph(graph_a, graph_b)
        return self._maximum_clique_size(association_graph)

    def _mcis_size(self, graph_a, graph_b):
        nodes_a = list(graph_a.nodes())
        nodes_b = list(graph_b.nodes())

        best_mapping = {}
        current_mapping = {}

        def induced_relations_coherent(node_a, node_b):
            for mapped_a, mapped_b in current_mapping.items():
                rel_a = self._relation_signature(graph_a, node_a, mapped_a)
                rel_b = self._relation_signature(graph_b, node_b, mapped_b)
                if rel_a != rel_b:
                    return False
            return True

        def backtrack(remaining_a, available_b):
            nonlocal best_mapping

            upper_bound = len(current_mapping) + min(len(remaining_a), len(available_b))
            if upper_bound <= len(best_mapping):
                return

            if not remaining_a or not available_b:
                if len(current_mapping) > len(best_mapping):
                    best_mapping = current_mapping.copy()
                return

            node_a = remaining_a[0]
            rest_a = remaining_a[1:]

            for idx, node_b in enumerate(available_b):
                if not self._nodes_compatible(graph_a, node_a, graph_b, node_b):
                    continue
                if not induced_relations_coherent(node_a, node_b):
                    continue

                current_mapping[node_a] = node_b
                next_available_b = available_b[:idx] + available_b[idx + 1 :]
                backtrack(rest_a, next_available_b)
                del current_mapping[node_a]

            # Skip branch to explore subsets when no valid pairing exists for node_a.
            backtrack(rest_a, available_b)

        backtrack(nodes_a, nodes_b)
        return len(best_mapping)

    def compute_distance_matrix(self, context: DistanceMatrixStrategyContext):
        graphs = context.db_graphs
        graph_names = [graph.get_name() for graph in graphs]
        verbose = self._is_verbose(context)
        use_mcis = self._use_mcis(context)

        n_graphs = len(graphs)
        distance_matrix = [[0.0 for _ in range(n_graphs)] for _ in range(n_graphs)]
        mode = "MCIS" if use_mcis else "MCS"

        if verbose:
            print(f"[mcs] Computing pairwise {mode} distance for {n_graphs} graphs...")

        for i in range(n_graphs):
            for j in range(i + 1, n_graphs):
                common_size = (
                    self._mcis_size(graphs[i], graphs[j])
                    if use_mcis
                    else self._mcs_size(graphs[i], graphs[j])
                )
                similarity = self._normalized_similarity(common_size, graphs[i], graphs[j])
                distance = 1.0 - similarity
                distance_matrix[i][j] = distance
                distance_matrix[j][i] = distance

        if verbose:
            print("[mcs] Distance matrix completed.")

        return distance_matrix, graph_names