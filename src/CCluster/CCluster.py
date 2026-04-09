import os
import random
import re

from sklearn.metrics import silhouette_score

from .strategies import (
    DistanceMatrixStrategyContext,
    FlexibleSubgraphDistanceStrategy,
    GraphDistanceStrategy,
    MCSDistanceStrategy,
    SimpleStructuralDistanceStrategy,
)


class CCluster:
    def __init__(
        self,
        db_file,
        num_clusters,
        directed_graph=0,
        output_path=None,
        strategy="simple_structural",
        init_method="random",
        max_iter=100,
        tolerance=1e-4,
        strategy_params=None,
    ):
        self.db_file = db_file
        self.num_clusters = num_clusters
        self.directed_graph = directed_graph
        self.output_path = output_path
        self.strategy = strategy
        self.init_method = init_method
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.strategy_params = strategy_params or {}
        self.db = []

    @staticmethod
    def _silhouette_score(distance_matrix, assignments, n_items):
        labels = [-1] * n_items
        for cluster_idx, points in enumerate(assignments):
            for point in points:
                labels[point] = cluster_idx

        # Ignore failed candidate clusterings (for example when empty clusters
        # reduce the number of effective labels below 2).
        try:
            return float(silhouette_score(distance_matrix, labels, metric="precomputed"))
        except ValueError:
            return float("-inf")

    def _read_graphs_from_file(self):
        from Graph.DBGraph import DirectedDBGraph, UndirectedDBGraph
        from NetworkLoader.NetworkConfigurator import NetworkConfigurator
        from NetworkLoader.NetworksLoading import NetworksLoading

        constructor_db_graph = (
            DirectedDBGraph if self.directed_graph else UndirectedDBGraph
        )
        type_file = self.db_file.split(".")[-1]
        configurator = NetworkConfigurator(self.db_file, type_file)
        for name, network in NetworksLoading(
            type_file, configurator.config, self.directed_graph
        ).Networks.items():
            self.db.append(constructor_db_graph(network, name))

    def _resolve_strategy(self) -> GraphDistanceStrategy:
        strategy_registry = {
            "simple_structural": SimpleStructuralDistanceStrategy(),
            "flexible_subgraph": FlexibleSubgraphDistanceStrategy(),
            "mcs": MCSDistanceStrategy(),
        }
        if self.strategy not in strategy_registry:
            available = ", ".join(sorted(strategy_registry))
            raise ValueError(
                f"Unknown strategy '{self.strategy}'. Available strategies: {available}"
            )
        return strategy_registry[self.strategy]

    @staticmethod
    def _distance(i, j, distance_matrix):
        if i == j:
            return 0.0
        if i < j:
            return distance_matrix[i][j]
        return distance_matrix[j][i]

    def _initialize_medoids(self, distance_matrix, n_items, num_clusters):
        if self.init_method == "random":
            return random.sample(range(n_items), num_clusters)

        if self.init_method != "kmeans++":
            raise ValueError(
                f"Unknown init_method '{self.init_method}'. Use 'random' or 'kmeans++'."
            )

        medoids = [random.randrange(n_items)]
        while len(medoids) < num_clusters:
            distances_to_nearest_medoid = []
            for idx in range(n_items):
                if idx in medoids:
                    distances_to_nearest_medoid.append(0.0)
                    continue
                nearest = min(
                    self._distance(idx, medoid, distance_matrix) for medoid in medoids
                )
                distances_to_nearest_medoid.append(nearest**2)

            total = sum(distances_to_nearest_medoid)
            if total == 0:
                remaining = [idx for idx in range(n_items) if idx not in medoids]
                medoids.extend(random.sample(remaining, num_clusters - len(medoids)))
                break

            target = random.random() * total
            cumulative = 0.0
            for idx, value in enumerate(distances_to_nearest_medoid):
                cumulative += value
                if cumulative >= target:
                    if idx not in medoids:
                        medoids.append(idx)
                    break

        return medoids

    def _select_num_clusters_auto(self, distance_matrix, graph_names):
        n_items = len(graph_names)
        if n_items == 0:
            raise ValueError("Graph database is empty.")

        if n_items <= 2:
            print(
                "Auto mode fallback: silhouette needs at least 3 graphs; "
                f"using num_clusters={n_items}."
            )
            return n_items

        best_k = 2
        best_score = float("-inf")

        for candidate_k in range(2, n_items):
            _, assignments = self._run_clustering_algorithm(
                distance_matrix,
                graph_names,
                num_clusters=candidate_k,
                return_assignments=True,
            )

            if any(len(points) == 0 for points in assignments):
                continue

            score = self._silhouette_score(distance_matrix, assignments, n_items)

            if score > best_score:
                best_score = score
                best_k = candidate_k

        print(
            "Auto mode selected "
            f"num_clusters={best_k} using silhouette score={best_score:.6f}."
        )
        return best_k

    def _repair_empty_clusters(self, medoids, assignments, distance_matrix, n_items):
        empty_cluster_indices = [idx for idx, points in enumerate(assignments) if not points]
        if not empty_cluster_indices:
            return medoids, False

        new_medoids = medoids[:]
        used_medoids = set(new_medoids)
        available_points = [point_idx for point_idx in range(n_items) if point_idx not in used_medoids]

        if not available_points:
            return new_medoids, False

        for empty_cluster_idx in empty_cluster_indices:
            if not available_points:
                break

            # Choose a point far from existing medoids to maximize separation.
            replacement = max(
                available_points,
                key=lambda point_idx: min(
                    self._distance(point_idx, medoid_idx, distance_matrix)
                    for medoid_idx in new_medoids
                ),
            )
            new_medoids[empty_cluster_idx] = replacement
            available_points.remove(replacement)

        return new_medoids, True

    def _assign_points(self, medoids, distance_matrix, n_items):
        assignments = [[] for _ in medoids]
        for idx in range(n_items):
            best_cluster = min(
                range(len(medoids)),
                key=lambda cluster_idx: self._distance(
                    idx, medoids[cluster_idx], distance_matrix
                ),
            )
            assignments[best_cluster].append(idx)
        return assignments

    def _recompute_medoids(self, assignments, medoids, distance_matrix):
        new_medoids = medoids[:]
        for cluster_idx, cluster_points in enumerate(assignments):
            if not cluster_points:
                continue
            best_medoid = min(
                cluster_points,
                key=lambda candidate: sum(
                    self._distance(candidate, other, distance_matrix)
                    for other in cluster_points
                ),
            )
            new_medoids[cluster_idx] = best_medoid
        return new_medoids

    def _run_clustering_algorithm(
        self,
        distance_matrix,
        graph_names,
        num_clusters=None,
        return_assignments=False,
    ):
        effective_num_clusters = (
            self.num_clusters if num_clusters is None else num_clusters
        )
        n_items = len(graph_names)
        if n_items == 0:
            raise ValueError("Graph database is empty.")
        if effective_num_clusters <= 0:
            raise ValueError("num_clusters must be greater than zero.")
        if effective_num_clusters > n_items:
            raise ValueError(
                "num_clusters "
                f"({effective_num_clusters}) cannot be greater than number of graphs ({n_items})."
            )

        medoids = self._initialize_medoids(
            distance_matrix, n_items, effective_num_clusters
        )

        for _ in range(self.max_iter):
            assignments = self._assign_points(medoids, distance_matrix, n_items)
            medoids, repaired = self._repair_empty_clusters(
                medoids, assignments, distance_matrix, n_items
            )
            if repaired:
                assignments = self._assign_points(medoids, distance_matrix, n_items)

            new_medoids = self._recompute_medoids(assignments, medoids, distance_matrix)
            changed = sum(1 for old, new in zip(medoids, new_medoids) if old != new)
            medoids = new_medoids

            if (changed / effective_num_clusters) <= self.tolerance:
                break

        clusters = {}
        final_assignments = self._assign_points(medoids, distance_matrix, n_items)
        for cluster_idx, points in enumerate(final_assignments):
            clusters[cluster_idx] = [graph_names[point_idx] for point_idx in points]

        if return_assignments:
            return clusters, final_assignments

        return clusters

    def _emit_results(self, clusters):
        print("Clustering result:")
        for cluster_id in sorted(clusters):
            print(f"  Cluster {cluster_id}: {clusters[cluster_id]}")

        if self.output_path is None:
            return

        source_name = os.path.splitext(os.path.basename(self.db_file))[0]
        safe_source_name = re.sub(r"[^A-Za-z0-9._-]", "_", source_name)
        cluster_folder = f"cluster_{safe_source_name}"
        target_folder = os.path.join(self.output_path, cluster_folder)

        os.makedirs(target_folder, exist_ok=True)

        db_by_name = {graph.get_name(): graph for graph in self.db}
        for cluster_id in sorted(clusters):
            graph_names = clusters[cluster_id]
            file_name = f"cluster_{cluster_id}_{len(graph_names)}"
            file_path = os.path.join(target_folder, file_name)

            with open(file_path, "w", encoding="utf-8") as handle:
                for local_idx, graph_name in enumerate(graph_names):
                    graph = db_by_name.get(graph_name)
                    if graph is None:
                        continue

                    handle.write(f"t # {local_idx} {graph_name}\n")

                    for node in sorted(graph.nodes(), key=lambda node_id: str(node_id)):
                        labels = graph.nodes[node].get("labels", [])
                        labels_part = " ".join(str(label) for label in labels if str(label))
                        if labels_part:
                            handle.write(f"v {node} {labels_part}\n")
                        else:
                            handle.write(f"v {node}\n")

                    edges = list(graph.edges(keys=True, data=True))
                    edges.sort(
                        key=lambda edge: (
                            str(edge[0]),
                            str(edge[1]),
                            str(edge[2]),
                        )
                    )
                    for src, dst, _, edge_data in edges:
                        edge_type = str(edge_data.get("type", "")).strip()
                        if edge_type:
                            handle.write(f"e {src} {dst} {edge_type}\n")
                        else:
                            handle.write(f"e {src} {dst}\n")

                    handle.write("\n")

        readme_path = os.path.join(target_folder, "README.md")
        with open(readme_path, "w", encoding="utf-8") as handle:
            handle.write("# Clustering Output\n\n")
            handle.write("Each file follows this naming convention:\n\n")
            handle.write("- `cluster_i_j`\n")
            handle.write("- `i`: cluster index\n")
            handle.write("- `j`: number of graphs stored in that cluster file\n\n")
            handle.write(
                "Each `cluster_i_j` file contains complete graphs in the `.data` "
                "style (`t`, `v`, `e` rows).\n"
            )
            handle.write("\n")
            handle.write(
                "This folder is named `cluster_[filename]`, where `[filename]` "
                "is the input database filename without extension.\n"
            )

        print(f"Results written to folder: {target_folder}")

    def cluster(self):
        print("Reading graphs from file...", end=" ")
        self._read_graphs_from_file()
        print("done.")

        strategy = self._resolve_strategy()
        context_num_clusters = (
            2 if isinstance(self.num_clusters, str) else self.num_clusters
        )
        context = DistanceMatrixStrategyContext(
            db_graphs=self.db,
            db_file=self.db_file,
            directed_graph=bool(self.directed_graph),
            num_clusters=context_num_clusters,
            init_method=self.init_method,
            max_iter=self.max_iter,
            tolerance=self.tolerance,
            strategy_params=self.strategy_params,
        )

        distance_matrix, graph_names = strategy.compute_distance_matrix(context)

        if isinstance(self.num_clusters, str):
            if self.num_clusters != "auto":
                raise ValueError(
                    "num_clusters string value is invalid. Supported value: 'auto'."
                )
            self.num_clusters = self._select_num_clusters_auto(distance_matrix, graph_names)

        clusters = self._run_clustering_algorithm(distance_matrix, graph_names)
        self._emit_results(clusters)
