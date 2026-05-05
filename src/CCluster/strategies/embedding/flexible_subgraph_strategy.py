"""Flexible subgraph distance strategy.

Pseudocode overview
-------------------

INPUT:
- db_graphs: list of multigraphs
- method: "nodes" or "edges"

OUTPUT:
- distance_matrix ready for clustering

ALGORITHM compute_distance_matrix(db_graphs, method):
    graph_profiles <- []
    global_vocabulary <- set()

    # Phase 1: subgraph extraction and counting
    for each graph in db_graphs:
        local_counts <- {}

        if method == "nodes":
            for k from 2 to number_of_nodes(graph):
                for each node_combination of size k:
                    subgraph <- node_induced_subgraph(graph, node_combination)
                    if is_connected(subgraph):
                        code <- canonical_code(subgraph)
                        local_counts[code] += 1
                        global_vocabulary.add(code)

        else if method == "edges":
            for m from 1 to number_of_edges(graph):
                for each edge_combination of size m:
                    subgraph <- edge_induced_subgraph(graph, edge_combination)
                    if is_connected(subgraph):
                        code <- canonical_code(subgraph)
                        local_counts[code] += 1
                        global_vocabulary.add(code)

        graph_profiles.append(local_counts)

    # Phase 2: feature matrix
    feature_matrix <- build_matrix(graph_profiles, global_vocabulary)

    # Phase 3: TF-IDF weighting
    weighted_matrix <- apply_tfidf(feature_matrix)

    # Phase 4: pairwise cosine distance
    distance_matrix <- cosine_distance(weighted_matrix)

    return distance_matrix
"""

import itertools
import math
import time

import networkx as nx
import numpy as np

from .embedding_strategy import DistanceMatrixStrategyContext, GraphDistanceStrategy


# Heartbeat tuning: stamp progress at most every this many combinations OR
# every this many seconds (whichever comes first).
_HEARTBEAT_EVERY_N_COMBOS = 50_000
_HEARTBEAT_EVERY_SECONDS = 5.0


class FlexibleSubgraphDistanceStrategy(GraphDistanceStrategy):
    """Compute graph distances from connected subgraph profiles.

    This strategy builds a bag-of-subgraphs representation for each graph,
    optionally using node-induced or edge-induced connected subgraphs.
    Canonical codes are used as features, then TF-IDF weighting is applied,
    and pairwise cosine distances are returned.
    """

    @property
    def name(self) -> str:
        """Return the strategy identifier used by the clustering registry."""
        return "flexible_subgraph"

    @staticmethod
    def _is_verbose(context: DistanceMatrixStrategyContext) -> bool:
        """Return whether verbose logging is enabled for this strategy."""
        return bool(context.strategy_params.get("verbose", False))

    @staticmethod
    def _is_connected(graph) -> bool:
        """Check connectivity for both directed and undirected multigraphs.

        Directed graphs are tested with weak connectivity.
        Single-node graphs are considered connected.
        """
        n_nodes = graph.number_of_nodes()
        if n_nodes == 0:
            return False
        if n_nodes == 1:
            return True
        if nx.is_directed(graph):
            return nx.is_weakly_connected(graph)
        return nx.is_connected(graph)

    @staticmethod
    def _new_empty_graph_like(graph):
        """Create an empty graph instance compatible with the input graph type.

        Some DB graph wrappers require constructor arguments `(graph, name)`.
        This helper abstracts constructor differences so subgraph builders can
        work with both base graph classes and DB wrapper classes.
        """
        graph_cls = type(graph)
        try:
            return graph_cls()
        except TypeError:
            # DB graph wrappers require (graph, name).
            if hasattr(graph, "get_name"):
                return graph_cls(None, f"{graph.get_name()}_subgraph")
            raise

    @staticmethod
    def _node_induced_subgraph(graph, node_combo):
        """Build the node-induced subgraph for the given node combination."""
        subgraph = FlexibleSubgraphDistanceStrategy._new_empty_graph_like(graph)

        node_set = set(node_combo)
        for node in node_combo:
            subgraph.add_node(node, **dict(graph.nodes[node]))

        for src, dst, key, edge_data in graph.edges(keys=True, data=True):
            if src in node_set and dst in node_set:
                subgraph.add_edge(src, dst, key=key, **dict(edge_data))

        return subgraph

    @staticmethod
    def _count_connected_node_induced_subgraphs(
        graph, *, min_size=2, max_size=None, verbose=False, log_prefix=""
    ):
        """Count connected node-induced subgraphs grouped by canonical code.

        Parameters
        ----------
        min_size:
            Lower bound (inclusive) on the size of the enumerated subgraphs.
            Defaults to ``2``.
        max_size:
            Upper bound (inclusive). ``None`` means ``n_nodes`` (no cap).

        When ``verbose`` is true and ``log_prefix`` is set, prints a heartbeat
        every ``_HEARTBEAT_EVERY_N_COMBOS`` combinations or every
        ``_HEARTBEAT_EVERY_SECONDS`` seconds (whichever comes first), so the
        user can see progress on graphs whose subset enumeration takes a
        while.
        """
        counts = {}
        nodes = list(graph.nodes())
        n_nodes = len(nodes)

        lo = max(2, int(min_size))
        hi = n_nodes if max_size is None else min(n_nodes, int(max_size))
        if hi < lo:
            # Empty effective range: nothing to enumerate.
            if verbose:
                print(
                    f"{log_prefix}n_nodes={n_nodes}, "
                    f"effective size range [{lo}, {hi}] is empty -> 0 subsets",
                    flush=True,
                )
            return counts

        total = sum(math.comb(n_nodes, k) for k in range(lo, hi + 1))

        if verbose and total > 0:
            print(
                f"{log_prefix}n_nodes={n_nodes}, "
                f"size range [{lo}, {hi}], "
                f"node-subsets to test={total:,}",
                flush=True,
            )

        processed = 0
        connected_kept = 0
        last_log = time.time()

        for size in range(lo, hi + 1):
            for node_combo in itertools.combinations(nodes, size):
                processed += 1
                subgraph = FlexibleSubgraphDistanceStrategy._node_induced_subgraph(
                    graph, node_combo
                )
                if FlexibleSubgraphDistanceStrategy._is_connected(subgraph):
                    code = subgraph.canonical_code()
                    counts[code] = counts.get(code, 0) + 1
                    connected_kept += 1

                if verbose and (
                    processed % _HEARTBEAT_EVERY_N_COMBOS == 0
                    or (time.time() - last_log) >= _HEARTBEAT_EVERY_SECONDS
                ):
                    pct = (processed * 100 // total) if total else 100
                    print(
                        f"{log_prefix}  {processed:,}/{total:,} ({pct}%) "
                        f"node-subsets tested, connected so far={connected_kept:,}",
                        flush=True,
                    )
                    last_log = time.time()

        return counts

    @staticmethod
    def _edge_induced_subgraph(graph, edge_combo):
        """Build an edge-induced subgraph from the selected multiedges."""
        subgraph = FlexibleSubgraphDistanceStrategy._new_empty_graph_like(graph)

        used_nodes = set()
        for src, dst, _ in edge_combo:
            used_nodes.add(src)
            used_nodes.add(dst)

        for node in used_nodes:
            subgraph.add_node(node, **dict(graph.nodes[node]))

        for src, dst, key in edge_combo:
            edge_data = dict(graph.get_edge_data(src, dst, key) or {})
            subgraph.add_edge(src, dst, key=key, **edge_data)

        return subgraph

    @staticmethod
    def _count_connected_edge_induced_subgraphs(
        graph, *, min_size=1, max_size=None, verbose=False, log_prefix=""
    ):
        """Count connected edge-induced subgraphs grouped by canonical code.

        Parameters
        ----------
        min_size:
            Lower bound (inclusive) on the number of edges per subgraph.
            Defaults to ``1``.
        max_size:
            Upper bound (inclusive). ``None`` means ``n_edges`` (no cap).

        Mirrors the heartbeat behavior of the node-induced variant.
        """
        counts = {}
        edges = list(graph.edges(keys=True))
        n_edges = len(edges)

        lo = max(1, int(min_size))
        hi = n_edges if max_size is None else min(n_edges, int(max_size))
        if hi < lo:
            if verbose:
                print(
                    f"{log_prefix}n_edges={n_edges}, "
                    f"effective size range [{lo}, {hi}] is empty -> 0 subsets",
                    flush=True,
                )
            return counts

        total = sum(math.comb(n_edges, k) for k in range(lo, hi + 1))

        if verbose and total > 0:
            print(
                f"{log_prefix}n_edges={n_edges}, "
                f"size range [{lo}, {hi}], "
                f"edge-subsets to test={total:,}",
                flush=True,
            )

        processed = 0
        connected_kept = 0
        last_log = time.time()

        for size in range(lo, hi + 1):
            for edge_combo in itertools.combinations(edges, size):
                processed += 1
                subgraph = FlexibleSubgraphDistanceStrategy._edge_induced_subgraph(
                    graph, edge_combo
                )
                if FlexibleSubgraphDistanceStrategy._is_connected(subgraph):
                    code = subgraph.canonical_code()
                    counts[code] = counts.get(code, 0) + 1
                    connected_kept += 1

                if verbose and (
                    processed % _HEARTBEAT_EVERY_N_COMBOS == 0
                    or (time.time() - last_log) >= _HEARTBEAT_EVERY_SECONDS
                ):
                    pct = (processed * 100 // total) if total else 100
                    print(
                        f"{log_prefix}  {processed:,}/{total:,} ({pct}%) "
                        f"edge-subsets tested, connected so far={connected_kept:,}",
                        flush=True,
                    )
                    last_log = time.time()

        return counts

    @staticmethod
    def _build_feature_matrix(graph_profiles, global_vocabulary):
        """Convert graph profile dictionaries into a dense feature matrix."""
        ordered_codes = sorted(global_vocabulary)
        matrix = []
        for profile in graph_profiles:
            matrix.append([float(profile.get(code, 0)) for code in ordered_codes])
        return matrix

    @staticmethod
    def _apply_tfidf(feature_matrix):
        """Apply TF-IDF weighting to the raw subgraph-count feature matrix."""
        if not feature_matrix:
            return []

        n_docs = len(feature_matrix)
        n_features = len(feature_matrix[0]) if feature_matrix[0] else 0
        if n_features == 0:
            return [[0.0] * 0 for _ in range(n_docs)]

        doc_freq = [0] * n_features
        for row in feature_matrix:
            for idx, value in enumerate(row):
                if value > 0:
                    doc_freq[idx] += 1

        idf = [
            math.log((1.0 + n_docs) / (1.0 + doc_freq[idx])) + 1.0
            for idx in range(n_features)
        ]

        weighted_matrix = []
        for row in feature_matrix:
            row_sum = sum(row)
            if row_sum == 0:
                weighted_matrix.append([0.0] * n_features)
                continue

            tfidf_row = []
            for idx, value in enumerate(row):
                tf = value / row_sum
                tfidf_row.append(tf * idf[idx])
            weighted_matrix.append(tfidf_row)

        return weighted_matrix

    @staticmethod
    def _cosine_distance_matrix(weighted_matrix):
        """
        Compute a symmetric cosine-distance matrix using vectorized NumPy operations.
        Maintains the exact same logic for zero-norm vectors as the original implementation.
        """
        # 1. Convert to contiguous C-array (much faster memory access)
        X = np.array(weighted_matrix, dtype=np.float64)

        # 2. Vectorized norm computation
        norms = np.linalg.norm(X, axis=1)

        # 3. Handle zero norms safely to avoid division by zero
        zero_mask = (norms == 0.0)
        safe_norms = np.where(zero_mask, 1.0, norms) # Replace 0 with 1 for division

        # Broadcased division to normalize all vectors at once
        X_normalized = X / safe_norms[:, np.newaxis]

        # 4. Matrix multiplication (Dot product of normalized vectors) -> O(N^2) in C
        cosine_similarity = np.dot(X_normalized, X_normalized.T)
        cosine_similarity = np.clip(cosine_similarity, -1.0, 1.0)

        # 5. Convert similarity to distance
        distances = 1.0 - cosine_similarity

        # 6. Apply your exact zero-norm business logic using boolean masks
        mask_i = zero_mask[:, np.newaxis] # Shape (N, 1)
        mask_j = zero_mask[np.newaxis, :] # Shape (1, N)

        # If either i or j is 0-norm -> distance is 1.0
        distances = np.where(mask_i | mask_j, 1.0, distances)

        # If BOTH i and j are 0-norm -> distance is 0.0
        distances = np.where(mask_i & mask_j, 0.0, distances)

        # Force exact 0.0 on the diagonal to avoid floating point inaccuracies (e.g., 1e-16)
        np.fill_diagonal(distances, 0.0)

        # Return as list of lists to match your original return type,
        # though returning the NumPy array directly is highly recommended for downstream tasks.
        return distances.tolist()

    def compute_distance_matrix(self, context: DistanceMatrixStrategyContext):
        """Build the strategy distance matrix and return `(distance_matrix, names)`.

        Strategy parameters (``context.strategy_params``):

        ``subgraph_method``  ``"nodes"`` (default) or ``"edges"`` — selects the
                             enumeration mode (node-induced vs edge-induced).
        ``min_size``         Lower bound (inclusive) on subgraph size.
                             Defaults: ``2`` for nodes, ``1`` for edges.
        ``max_size``         Upper bound (inclusive). ``None`` = no cap.
        ``verbose``          Print progress logs.
        """
        params = context.strategy_params
        verbose = self._is_verbose(context)
        method = (
            params.get("subgraph_method", "nodes") or "nodes"
        ).strip().lower()
        if method not in {"nodes", "edges"}:
            raise ValueError(
                "Unknown flexible_subgraph subgraph_method "
                f"'{method}'. Supported values: nodes, edges"
            )

        # Optional size bounds. ``None`` keeps the legacy default
        # (min=2 nodes / 1 edge, no upper cap).
        min_size = params.get("min_size", None)
        max_size = params.get("max_size", None)
        default_min = 2 if method == "nodes" else 1
        effective_min = default_min if min_size is None else int(min_size)
        if effective_min < 1:
            raise ValueError("flexible_subgraph min_size must be >= 1")
        if max_size is not None and int(max_size) < effective_min:
            raise ValueError(
                "flexible_subgraph max_size "
                f"({max_size}) must be >= min_size ({effective_min})"
            )

        if verbose:
            print(
                "[flexible_subgraph] Phase 1/4: extracting connected "
                f"{method}-induced subgraphs and computing canonical-code counts "
                f"(min_size={effective_min}, "
                f"max_size={'inf' if max_size is None else int(max_size)})..."
            )

        graph_profiles = []
        global_vocabulary = set()

        total_graphs = len(context.db_graphs)
        for index, graph in enumerate(context.db_graphs, start=1):
            graph_name = (
                graph.get_name() if hasattr(graph, "get_name") else f"#{index}"
            )
            if verbose:
                print(
                    f"[flexible_subgraph] Processing graph {index}/{total_graphs} "
                    f"({graph_name})",
                    flush=True,
                )

            t_start = time.time()
            log_prefix = f"[flexible_subgraph] graph {index}/{total_graphs}: "
            if method == "nodes":
                counts = self._count_connected_node_induced_subgraphs(
                    graph,
                    min_size=effective_min,
                    max_size=max_size,
                    verbose=verbose,
                    log_prefix=log_prefix,
                )
            else:
                counts = self._count_connected_edge_induced_subgraphs(
                    graph,
                    min_size=effective_min,
                    max_size=max_size,
                    verbose=verbose,
                    log_prefix=log_prefix,
                )
            elapsed = time.time() - t_start

            graph_profiles.append(counts)
            global_vocabulary.update(counts.keys())

            if verbose:
                extracted_subgraphs = sum(counts.values())
                print(
                    f"[flexible_subgraph] graph {index}/{total_graphs} done "
                    f"({extracted_subgraphs:,} connected subgraphs in {elapsed:.1f}s)",
                    flush=True,
                )

        if verbose:
            print(
                "[flexible_subgraph] Phase 2/4: building feature matrix from "
                f"{len(global_vocabulary)} canonical codes..."
            )

        feature_matrix = self._build_feature_matrix(graph_profiles, global_vocabulary)

        if verbose:
            print("[flexible_subgraph] Phase 3/4: applying TF-IDF weighting...")

        weighted_matrix = self._apply_tfidf(feature_matrix)

        if verbose:
            print("[flexible_subgraph] Phase 4/4: computing cosine distance matrix...")

        distance_matrix = self._cosine_distance_matrix(weighted_matrix)
        graph_names = [graph.get_name() for graph in context.db_graphs]

        if verbose:
            print("[flexible_subgraph] Distance matrix completed.")

        return distance_matrix, graph_names
