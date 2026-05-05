"""K-medoids clustering strategy.

This module encapsulates the k-medoids algorithm previously inlined in
``CCluster.CCluster``. It works on any precomputed symmetric distance matrix
and is therefore decoupled from the embedding strategy that produced it.
"""

import random
import time
from typing import Dict, List, Tuple

from .clustering_strategy import ClusteringContext, ClusteringStrategy


class KMedoidsClusteringStrategy(ClusteringStrategy):
    """Classic Partitioning Around Medoids (PAM) on a precomputed distance matrix."""

    @property
    def name(self) -> str:
        return "kmedoids"

    # ------------------------------------------------------------------ utils

    @staticmethod
    def _distance(i, j, distance_matrix):
        if i == j:
            return 0.0
        if i < j:
            return distance_matrix[i][j]
        return distance_matrix[j][i]

    # ------------------------------------------------------------------ init

    def _initialize_medoids(self, distance_matrix, n_items, num_clusters, init_method):
        if init_method == "random":
            return random.sample(range(n_items), num_clusters)

        if init_method != "kmeans++":
            raise ValueError(
                f"Unknown init_method '{init_method}'. Use 'random' or 'kmeans++'."
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
                distances_to_nearest_medoid.append(nearest ** 2)

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

    # --------------------------------------------------------------- main loop

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

    def _repair_empty_clusters(self, medoids, assignments, distance_matrix, n_items):
        empty_cluster_indices = [
            idx for idx, points in enumerate(assignments) if not points
        ]
        if not empty_cluster_indices:
            return medoids, False

        new_medoids = medoids[:]
        used_medoids = set(new_medoids)
        available_points = [
            point_idx for point_idx in range(n_items) if point_idx not in used_medoids
        ]

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

    # ---------------------------------------------------------------- public

    def cluster(
        self,
        distance_matrix,
        graph_names: List[str],
        context: ClusteringContext,
    ) -> Tuple[Dict[int, List[str]], List[List[int]]]:
        n_items = len(graph_names)
        num_clusters = context.num_clusters
        verbose = bool(context.strategy_params.get("verbose", False))

        if n_items == 0:
            raise ValueError("Graph database is empty.")
        if num_clusters <= 0:
            raise ValueError("num_clusters must be greater than zero.")
        if num_clusters > n_items:
            raise ValueError(
                "num_clusters "
                f"({num_clusters}) cannot be greater than number of graphs ({n_items})."
            )

        if verbose:
            print(
                f"[kmedoids] starting: n_items={n_items}, k={num_clusters}, "
                f"init={context.init_method}, max_iter={context.max_iter}",
                flush=True,
            )

        medoids = self._initialize_medoids(
            distance_matrix, n_items, num_clusters, context.init_method
        )

        t_start = time.time()
        for iteration in range(context.max_iter):
            assignments = self._assign_points(medoids, distance_matrix, n_items)
            medoids, repaired = self._repair_empty_clusters(
                medoids, assignments, distance_matrix, n_items
            )
            if repaired:
                assignments = self._assign_points(medoids, distance_matrix, n_items)

            new_medoids = self._recompute_medoids(
                assignments, medoids, distance_matrix
            )
            changed = sum(1 for old, new in zip(medoids, new_medoids) if old != new)
            medoids = new_medoids

            if verbose:
                cluster_sizes = [len(points) for points in assignments]
                print(
                    f"[kmedoids]   iter {iteration + 1}/{context.max_iter}: "
                    f"{changed}/{num_clusters} medoids changed, "
                    f"sizes={cluster_sizes}",
                    flush=True,
                )

            if (changed / num_clusters) <= context.tolerance:
                if verbose:
                    print(
                        f"[kmedoids] converged at iter {iteration + 1} "
                        f"(elapsed {time.time() - t_start:.2f}s)",
                        flush=True,
                    )
                break
        else:
            if verbose:
                print(
                    f"[kmedoids] reached max_iter={context.max_iter} without "
                    f"full convergence (elapsed {time.time() - t_start:.2f}s)",
                    flush=True,
                )

        final_assignments = self._assign_points(medoids, distance_matrix, n_items)
        clusters: Dict[int, List[str]] = {}
        for cluster_idx, points in enumerate(final_assignments):
            clusters[cluster_idx] = [graph_names[point_idx] for point_idx in points]

        return clusters, final_assignments
