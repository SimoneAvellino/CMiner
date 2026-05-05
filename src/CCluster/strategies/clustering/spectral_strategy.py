"""Spectral clustering strategy.

Spectral clustering on a precomputed affinity matrix derived from the
pairwise distance matrix.  It projects the graphs onto a low-dimensional
eigenspace before running k-means, which lets it find clusters with complex,
non-spherical shapes that k-medoids or agglomerative methods may miss.

Distance → affinity conversion (``affinity_mode`` parameter):

* ``gaussian`` *(default)* — ``A = exp(-D² / σ²)``; the bandwidth ``sigma``
  defaults to the median of all pairwise distances.  Set ``sigma`` explicitly
  when the automatic estimate is too coarse or too fine.
* ``reciprocal`` — ``A = 1 / (1 + D)``; a simple monotone conversion that
  requires no tuning and works well when distances already span a bounded
  range.

Because the final step is k-means on the spectral embedding, results can vary
across runs.  Use ``random_state`` to make them reproducible.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np

from .clustering_strategy import ClusteringContext, ClusteringStrategy

_VALID_AFFINITY_MODES = ("gaussian", "reciprocal")


class SpectralClusteringStrategy(ClusteringStrategy):
    """Spectral clustering via a precomputed affinity matrix."""

    @property
    def name(self) -> str:
        return "spectral"

    @property
    def description(self) -> str:
        return (
            "Spectral clustering via a Gaussian affinity matrix derived from the "
            "distance matrix. Projects graphs into a low-dimensional eigenspace "
            "before k-means, which lets it detect complex, non-spherical cluster "
            "shapes that distance-based methods may miss."
        )

    # -------------------------------------------------------------- helpers

    @staticmethod
    def _to_affinity(dist: np.ndarray, mode: str, sigma: Optional[float]) -> np.ndarray:
        if mode == "gaussian":
            if sigma is None:
                # Use median of all pairwise distances as bandwidth.
                flat = dist[dist > 0]
                sigma = float(np.median(flat)) if flat.size > 0 else 1.0
            affinity = np.exp(-(dist ** 2) / (sigma ** 2))
        elif mode == "reciprocal":
            affinity = 1.0 / (1.0 + dist)
        else:
            raise ValueError(
                f"Unknown affinity_mode '{mode}'. "
                f"Valid options: {', '.join(_VALID_AFFINITY_MODES)}."
            )
        # Ensure perfect symmetry and zero diagonal (numerical safety).
        np.fill_diagonal(affinity, 1.0)
        affinity = (affinity + affinity.T) / 2.0
        return affinity

    # -------------------------------------------------------------- public

    def cluster(
        self,
        distance_matrix,
        graph_names: List[str],
        context: ClusteringContext,
    ) -> Tuple[Dict[int, List[str]], List[List[int]]]:
        from sklearn.cluster import SpectralClustering

        n_items = len(graph_names)
        num_clusters = context.num_clusters
        affinity_mode = str(
            context.strategy_params.get("affinity_mode", "gaussian")
        ).lower()
        sigma_param = context.strategy_params.get("sigma", None)
        sigma = float(sigma_param) if sigma_param is not None else None
        random_state = int(context.strategy_params.get("random_state", 42))
        n_init = int(context.strategy_params.get("n_init", 10))
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
        if affinity_mode not in _VALID_AFFINITY_MODES:
            raise ValueError(
                f"Unknown affinity_mode '{affinity_mode}'. "
                f"Valid options: {', '.join(_VALID_AFFINITY_MODES)}."
            )

        if verbose:
            print(
                f"[spectral] starting: n_items={n_items}, k={num_clusters}, "
                f"affinity_mode={affinity_mode}, random_state={random_state}",
                flush=True,
            )

        dist = np.asarray(distance_matrix, dtype=np.float64)
        affinity = self._to_affinity(dist, affinity_mode, sigma)

        if verbose and affinity_mode == "gaussian":
            effective_sigma = sigma if sigma is not None else float(
                np.median(dist[dist > 0]) if (dist > 0).any() else 1.0
            )
            print(f"[spectral] gaussian sigma={effective_sigma:.4f}", flush=True)

        model = SpectralClustering(
            n_clusters=num_clusters,
            affinity="precomputed",
            random_state=random_state,
            n_init=n_init,
        )
        with warnings.catch_warnings(), np.errstate(
            invalid="ignore", divide="ignore", over="ignore"
        ):
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            flat_labels = model.fit_predict(affinity)

        assignments: List[List[int]] = [[] for _ in range(num_clusters)]
        for idx, label in enumerate(flat_labels):
            assignments[int(label)].append(idx)

        clusters: Dict[int, List[str]] = {
            cluster_idx: [graph_names[i] for i in points]
            for cluster_idx, points in enumerate(assignments)
        }

        if verbose:
            sizes = [len(points) for points in assignments]
            print(f"[spectral] done. cluster sizes={sizes}", flush=True)

        return clusters, assignments
