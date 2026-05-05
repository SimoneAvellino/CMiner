"""Agglomerative hierarchical clustering strategy.

Hierarchical clustering on a precomputed distance matrix via
``sklearn.cluster.AgglomerativeClustering``.  Contrary to k-medoids this
method is fully deterministic (no random initialisation) and tends to produce
better results when clusters have unequal densities or sizes.

Supported linkage criteria (``linkage`` parameter):

* ``average`` *(default)* — UPGMA; minimises average inter-cluster distance.
  A solid general-purpose choice.
* ``complete`` — maximises the maximum inter-cluster distance (diameter
  linkage); produces compact, roughly equal-sized clusters.
* ``single`` — minimises the minimum inter-cluster distance (nearest-
  neighbour linkage); can detect elongated / chain-like clusters but is
  sensitive to outliers.

Note: ``ward`` linkage is NOT supported because it requires raw Euclidean
coordinates, not a precomputed distance matrix.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .clustering_strategy import ClusteringContext, ClusteringStrategy

_VALID_LINKAGES = ("average", "complete", "single")


class AgglomerativeClusteringStrategy(ClusteringStrategy):
    """Hierarchical agglomerative clustering on a precomputed distance matrix."""

    @property
    def name(self) -> str:
        return "agglomerative"

    @property
    def description(self) -> str:
        return (
            "Hierarchical agglomerative clustering (average linkage by default) on "
            "the precomputed distance matrix. Fully deterministic — no random "
            "initialisation — and tends to produce better results than k-medoids "
            "when clusters have unequal sizes or densities."
        )

    def cluster(
        self,
        distance_matrix,
        graph_names: List[str],
        context: ClusteringContext,
    ) -> Tuple[Dict[int, List[str]], List[List[int]]]:
        from sklearn.cluster import AgglomerativeClustering

        n_items = len(graph_names)
        num_clusters = context.num_clusters
        linkage = str(context.strategy_params.get("linkage", "average")).lower()
        verbose = bool(context.strategy_params.get("verbose", False))

        if n_items == 0:
            raise ValueError("Graph database is empty.")
        if num_clusters <= 0:
            raise ValueError("num_clusters must be greater than zero.")
        if num_clusters > n_items:
            raise ValueError(
                f"num_clusters ({num_clusters}) cannot be greater than "
                f"number of graphs ({n_items})."
            )
        if linkage not in _VALID_LINKAGES:
            raise ValueError(
                f"Unknown linkage '{linkage}'. "
                f"Valid options: {', '.join(_VALID_LINKAGES)}."
            )

        if verbose:
            print(
                f"[agglomerative] starting: n_items={n_items}, "
                f"k={num_clusters}, linkage={linkage}",
                flush=True,
            )

        dist = np.asarray(distance_matrix, dtype=np.float64)

        model = AgglomerativeClustering(
            n_clusters=num_clusters,
            metric="precomputed",
            linkage=linkage,
        )
        flat_labels = model.fit_predict(dist)

        assignments: List[List[int]] = [[] for _ in range(num_clusters)]
        for idx, label in enumerate(flat_labels):
            assignments[int(label)].append(idx)

        clusters: Dict[int, List[str]] = {
            cluster_idx: [graph_names[i] for i in points]
            for cluster_idx, points in enumerate(assignments)
        }

        if verbose:
            sizes = [len(points) for points in assignments]
            print(f"[agglomerative] done. cluster sizes={sizes}", flush=True)

        return clusters, assignments
