import json
import random

from .strategies import (
    DistanceMatrixStrategyContext,
    GraphDistanceStrategy,
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

    def _initialize_medoids(self, distance_matrix, n_items):
        if self.init_method == "random":
            return random.sample(range(n_items), self.num_clusters)

        if self.init_method != "kmeans++":
            raise ValueError(
                f"Unknown init_method '{self.init_method}'. Use 'random' or 'kmeans++'."
            )

        medoids = [random.randrange(n_items)]
        while len(medoids) < self.num_clusters:
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
                medoids.extend(
                    random.sample(remaining, self.num_clusters - len(medoids))
                )
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

    def _run_clustering_algorithm(self, distance_matrix, graph_names):
        n_items = len(graph_names)
        if n_items == 0:
            raise ValueError("Graph database is empty.")
        if self.num_clusters <= 0:
            raise ValueError("num_clusters must be greater than zero.")
        if self.num_clusters > n_items:
            raise ValueError(
                f"num_clusters ({self.num_clusters}) cannot be greater than number of graphs ({n_items})."
            )

        medoids = self._initialize_medoids(distance_matrix, n_items)

        for _ in range(self.max_iter):
            assignments = self._assign_points(medoids, distance_matrix, n_items)
            new_medoids = self._recompute_medoids(assignments, medoids, distance_matrix)
            changed = sum(1 for old, new in zip(medoids, new_medoids) if old != new)
            medoids = new_medoids

            if (changed / self.num_clusters) <= self.tolerance:
                break

        clusters = {}
        final_assignments = self._assign_points(medoids, distance_matrix, n_items)
        for cluster_idx, points in enumerate(final_assignments):
            clusters[cluster_idx] = [graph_names[point_idx] for point_idx in points]

        return clusters

    def _emit_results(self, clusters):
        print("Clustering result:")
        for cluster_id in sorted(clusters):
            print(f"  Cluster {cluster_id}: {clusters[cluster_id]}")

        if self.output_path is None:
            return

        payload = {
            "num_clusters": self.num_clusters,
            "strategy": self.strategy,
            "clusters": clusters,
        }
        with open(self.output_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"Results written to {self.output_path}")

    def cluster(self):
        print("Reading graphs from file...", end=" ")
        self._read_graphs_from_file()
        print("done.")

        strategy = self._resolve_strategy()
        context = DistanceMatrixStrategyContext(
            db_graphs=self.db,
            db_file=self.db_file,
            directed_graph=bool(self.directed_graph),
            num_clusters=self.num_clusters,
            init_method=self.init_method,
            max_iter=self.max_iter,
            tolerance=self.tolerance,
            strategy_params=self.strategy_params,
        )

        distance_matrix, graph_names = strategy.compute_distance_matrix(context)
        clusters = self._run_clustering_algorithm(distance_matrix, graph_names)
        self._emit_results(clusters)
