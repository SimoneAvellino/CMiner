"""Semantic-label clustering enrichment strategy.

Augments each graph with synthetic "super-nodes" that group together node
labels and edge labels with similar semantics, computed once on the entire
database to guarantee a deterministic mapping (no train/test leakage).

Pipeline:

1. Collect every unique node label and every unique edge label across the
   whole database.
2. Embed each label with a SentenceTransformer model (default
   ``all-MiniLM-L6-v2``) and L2-normalize.
3. Run a K-Means grid search over ``k`` in ``[2, K_max]`` (with
   ``K_max = ceil(sqrt(n_unique_labels))`` by default) and pick the ``k`` with
   the highest mean silhouette score.
4. Outlier handling: any label whose per-sample silhouette is negative is
   reassigned to a synthetic ``-1`` "noise" cluster, so it does not pollute
   the semantics of the healthy clusters.
5. For every graph, deep-copy it and add:
   - one super-node per cluster id actually present in the graph, with label
     ``semantic_cluster_<cluster_id>``;
   - a ``BELONGS_TO_SEMANTIC_NODE`` edge from each original node to the
     super-node of its label cluster;
   - a ``PARTICIPATES_IN_SEMANTIC_EDGE`` edge from both endpoints of each
     original edge to the super-node of that edge's cluster.

The original graphs are NOT mutated: the strategy returns enriched copies.
"""

from __future__ import annotations

import math
import time
import warnings
from collections import Counter
from contextlib import contextmanager
from typing import Dict, List, Optional

from .enrichment_strategy import EnrichGraphSemanticsStrategy, EnrichmentContext


_DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
_NODE_SUPERNODE_PREFIX = "__semantic_node_cluster_"
_EDGE_SUPERNODE_PREFIX = "__semantic_edge_cluster_"
_SUPERNODE_LABEL_TEMPLATE = "semantic_cluster_{cluster_id}"
_BELONGS_TO_SEMANTIC_NODE = "BELONGS_TO_SEMANTIC_NODE"
_PARTICIPATES_IN_SEMANTIC_EDGE = "PARTICIPATES_IN_SEMANTIC_EDGE"


@contextmanager
def _silence_numpy_warnings():
    """Suppress harmless RuntimeWarnings emitted by sklearn's matmul-based
    silhouette internals on degenerate inputs (zero-norm rows, duplicated
    embeddings, etc.). Numerically the silhouette score is still meaningful,
    so we silence the noise locally."""
    import numpy as np

    with warnings.catch_warnings(), np.errstate(
        over="ignore", invalid="ignore", divide="ignore"
    ):
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        yield


class SemanticLabelClusteringEnrichmentStrategy(EnrichGraphSemanticsStrategy):
    """Cluster node and edge labels via SentenceTransformer + K-Means.

    Strategy parameters (``context.strategy_params``):

    ``model_name``       SentenceTransformer model id (default
                         ``all-MiniLM-L6-v2``).
    ``random_state``     Seed for K-Means (default ``42``).
    ``max_k``            Override for the upper bound of the grid search.
                         If unset, ``ceil(sqrt(n_unique_labels))`` is used.
    ``verbose``          Bool, prints progress logs (default ``False``).
    """

    @property
    def name(self) -> str:
        return "semantic_label_clustering"

    @property
    def description(self) -> str:
        return (
            "Embeds node/edge labels with a SentenceTransformer model "
            "(default: all-MiniLM-L6-v2), groups them via K-Means (k selected "
            "automatically by silhouette score), then adds one synthetic super-node "
            "per cluster to each graph. Original nodes are connected to their "
            "label's super-node via a BELONGS_TO_SEMANTIC_NODE edge; edge endpoints "
            "are connected to the edge-label super-node via a "
            "PARTICIPATES_IN_SEMANTIC_EDGE edge."
        )

    # ------------------------------------------------------------------ main

    def enrich(self, context: EnrichmentContext) -> List:
        verbose = bool(context.strategy_params.get("verbose", False))
        model_name = context.strategy_params.get("model_name", _DEFAULT_MODEL_NAME)
        random_state = int(context.strategy_params.get("random_state", 42))
        max_k_override = context.strategy_params.get("max_k", None)

        graphs = list(context.db_graphs)
        if not graphs:
            return graphs

        unique_node_labels = self._collect_unique_node_labels(graphs)
        unique_edge_labels = self._collect_unique_edge_labels(graphs)

        if verbose:
            print(
                "[semantic_label_clustering] "
                f"unique node labels: {len(unique_node_labels)} | "
                f"unique edge labels: {len(unique_edge_labels)}"
            )

        node_label_to_cluster = self._cluster_labels(
            unique_node_labels,
            family="node",
            model_name=model_name,
            random_state=random_state,
            max_k_override=max_k_override,
            verbose=verbose,
        )
        edge_label_to_cluster = self._cluster_labels(
            unique_edge_labels,
            family="edge",
            model_name=model_name,
            random_state=random_state,
            max_k_override=max_k_override,
            verbose=verbose,
        )

        if verbose:
            print(
                "[semantic_label_clustering] "
                f"augmenting {len(graphs)} graphs with semantic super-nodes..."
            )

        enriched = []
        t_start = time.time()
        last_log = t_start
        for index, graph in enumerate(graphs, start=1):
            enriched_graph = self._enrich_graph(
                graph, node_label_to_cluster, edge_label_to_cluster
            )
            enriched.append(enriched_graph)
            if verbose and (
                index == len(graphs)
                or index % 10 == 0
                or (time.time() - last_log) >= 5.0
            ):
                graph_name = (
                    graph.get_name() if hasattr(graph, "get_name") else f"#{index}"
                )
                added_nodes = (
                    enriched_graph.number_of_nodes() - graph.number_of_nodes()
                )
                added_edges = (
                    enriched_graph.number_of_edges() - graph.number_of_edges()
                )
                print(
                    f"[semantic_label_clustering]   {index}/{len(graphs)} "
                    f"({graph_name}): +{added_nodes} super-nodes, "
                    f"+{added_edges} edges",
                    flush=True,
                )
                last_log = time.time()

        if verbose:
            elapsed = time.time() - t_start
            print(
                "[semantic_label_clustering] "
                f"{len(enriched)} graphs enriched in {elapsed:.1f}s."
            )

        return enriched

    # ---------------------------------------------------- label collection

    @staticmethod
    def _collect_unique_node_labels(graphs) -> List[str]:
        seen = set()
        for graph in graphs:
            for _, node_data in graph.nodes(data=True):
                for label in node_data.get("labels", []) or []:
                    if label is None:
                        continue
                    text = str(label).strip()
                    if text:
                        seen.add(text)
        return sorted(seen)

    @staticmethod
    def _collect_unique_edge_labels(graphs) -> List[str]:
        seen = set()
        for graph in graphs:
            for _, _, _, edge_data in graph.edges(keys=True, data=True):
                edge_type = edge_data.get("type", None)
                if edge_type is None:
                    continue
                text = str(edge_type).strip()
                if text:
                    seen.add(text)
        return sorted(seen)

    # -------------------------------------------------------- clustering core

    def _cluster_labels(
        self,
        labels: List[str],
        family: str,
        model_name: str,
        random_state: int,
        max_k_override: Optional[int],
        verbose: bool,
    ) -> Dict[str, int]:
        if len(labels) == 0:
            return {}
        if len(labels) == 1:
            if verbose:
                print(
                    f"[semantic_label_clustering] {family}: only 1 unique label, "
                    "assigning it to cluster 0."
                )
            return {labels[0]: 0}
        if len(labels) == 2:
            # K-Means with k=2 on 2 points is trivial; assign one per cluster
            # without bothering the embedding model.
            if verbose:
                print(
                    f"[semantic_label_clustering] {family}: only 2 unique labels, "
                    "assigning one per cluster."
                )
            return {labels[0]: 0, labels[1]: 1}

        embeddings = self._compute_embeddings(labels, model_name, verbose, family)
        best_k, hard_labels = self._grid_search_kmeans(
            embeddings,
            max_k_override=max_k_override,
            random_state=random_state,
            family=family,
            verbose=verbose,
        )

        # Per-sample silhouette → outlier handling.
        try:
            import numpy as np
            from sklearn.metrics import silhouette_samples

            with _silence_numpy_warnings():
                sample_scores = silhouette_samples(embeddings, hard_labels)
            sample_scores = np.nan_to_num(
                sample_scores, nan=0.0, posinf=0.0, neginf=0.0
            )
        except (ValueError, ImportError):
            sample_scores = None

        result: Dict[str, int] = {}
        for idx, label in enumerate(labels):
            cluster_id = int(hard_labels[idx])
            if sample_scores is not None and sample_scores[idx] < 0:
                cluster_id = -1
            result[label] = cluster_id

        if verbose:
            counts = Counter(result.values())
            print(
                f"[semantic_label_clustering] {family}: best_k={best_k}, "
                f"cluster_sizes={dict(sorted(counts.items()))}"
            )

        return result

    @staticmethod
    def _compute_embeddings(labels, model_name, verbose, family):
        # Lazy import: ``sentence_transformers`` is only required when this
        # strategy is actually selected.
        import numpy as np

        from sentence_transformers import SentenceTransformer
        from sklearn.preprocessing import normalize as l2_normalize

        if verbose:
            print(
                f"[semantic_label_clustering] {family}: loading "
                f"SentenceTransformer('{model_name}')..."
            )
        model = SentenceTransformer(model_name)
        embeddings = model.encode(list(labels), show_progress_bar=False)
        embeddings = np.asarray(embeddings, dtype=np.float64)
        # Defensive sanitization: replace any NaN/Inf produced by the encoder
        # with 0 so downstream sklearn routines (silhouette, K-Means) don't
        # warn on overflows / divisions by zero.
        embeddings = np.nan_to_num(
            embeddings, nan=0.0, posinf=0.0, neginf=0.0
        )
        embeddings = l2_normalize(embeddings)
        return embeddings

    @staticmethod
    def _grid_search_kmeans(
        embeddings, max_k_override, random_state, family, verbose
    ):
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        n_samples = len(embeddings)
        if max_k_override is not None:
            k_max = int(max_k_override)
        else:
            k_max = max(2, int(math.ceil(math.sqrt(n_samples))))
        # silhouette requires 2 <= k <= n_samples - 1
        k_max = min(k_max, n_samples - 1)
        k_min = 2

        if k_max < k_min:
            # Degenerate case: not enough samples for a non-trivial grid.
            km = KMeans(n_clusters=k_min, random_state=random_state, n_init=10)
            assignments = km.fit_predict(embeddings)
            return k_min, assignments

        if verbose:
            print(
                f"[semantic_label_clustering] {family}: searching k in "
                f"[{k_min}, {k_max}] on {n_samples} samples"
            )

        best_score = float("-inf")
        best_k = k_min
        best_labels = None

        with _silence_numpy_warnings():
            for k in range(k_min, k_max + 1):
                km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
                assignments = km.fit_predict(embeddings)
                try:
                    score = silhouette_score(embeddings, assignments)
                except ValueError:
                    continue
                if verbose:
                    print(
                        f"[semantic_label_clustering]   k={k:2d} silhouette={score:.4f}"
                    )
                if score > best_score:
                    best_score = score
                    best_k = k
                    best_labels = assignments

            if best_labels is None:
                km = KMeans(n_clusters=k_min, random_state=random_state, n_init=10)
                best_labels = km.fit_predict(embeddings)
                best_k = k_min

        if verbose:
            print(
                f"[semantic_label_clustering] {family}: selected k={best_k} "
                f"(silhouette={best_score:.4f})"
            )

        return best_k, best_labels

    # ----------------------------------------------------- per-graph augment

    def _enrich_graph(
        self,
        graph,
        node_label_to_cluster: Dict[str, int],
        edge_label_to_cluster: Dict[str, int],
    ):
        """Return an augmented copy of ``graph`` (the original is untouched)."""
        enriched = self._clone_graph(graph)

        # 1. Decide which clusters are actually used inside this graph.
        node_to_cluster: Dict[object, int] = {}
        node_clusters_present = set()
        for node_id, node_data in list(enriched.nodes(data=True)):
            cluster_id = self._first_known_cluster(
                node_data.get("labels", []) or [], node_label_to_cluster
            )
            if cluster_id is None:
                continue
            node_to_cluster[node_id] = cluster_id
            node_clusters_present.add(cluster_id)

        edge_to_cluster: Dict[tuple, int] = {}
        edge_clusters_present = set()
        for src, dst, key, edge_data in list(enriched.edges(keys=True, data=True)):
            edge_type = edge_data.get("type", None)
            if edge_type is None:
                continue
            text = str(edge_type).strip()
            if not text:
                continue
            cluster_id = edge_label_to_cluster.get(text)
            if cluster_id is None:
                continue
            edge_to_cluster[(src, dst, key)] = cluster_id
            edge_clusters_present.add(cluster_id)

        # 2. Materialize the super-nodes (one per family per cluster).
        # Super-node ids must be type-compatible with existing node ids,
        # otherwise downstream code that calls `sorted(self.nodes())`
        # (e.g. canonical_code) blows up with a TypeError when comparing
        # str against int. When the graph already uses int ids we allocate
        # large negative integers; otherwise we fall back to string ids.
        node_super = self._allocate_super_ids(
            enriched, node_clusters_present, kind="node"
        )
        for cid, super_id in node_super.items():
            enriched.add_node(
                super_id,
                labels=[_SUPERNODE_LABEL_TEMPLATE.format(cluster_id=cid)],
            )

        edge_super = self._allocate_super_ids(
            enriched, edge_clusters_present, kind="edge"
        )
        for cid, super_id in edge_super.items():
            enriched.add_node(
                super_id,
                labels=[_SUPERNODE_LABEL_TEMPLATE.format(cluster_id=cid)],
            )

        # 3. Connect original nodes to their cluster super-node.
        for node_id, cid in node_to_cluster.items():
            enriched.add_edge(
                node_id,
                node_super[cid],
                type=_BELONGS_TO_SEMANTIC_NODE,
            )

        # 4. Connect endpoints of each clustered edge to the edge super-node.
        for (src, dst, _key), cid in edge_to_cluster.items():
            super_id = edge_super[cid]
            enriched.add_edge(src, super_id, type=_PARTICIPATES_IN_SEMANTIC_EDGE)
            enriched.add_edge(dst, super_id, type=_PARTICIPATES_IN_SEMANTIC_EDGE)

        return enriched

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def _first_known_cluster(node_labels, node_label_to_cluster):
        for label in node_labels:
            if label is None:
                continue
            text = str(label).strip()
            if text in node_label_to_cluster:
                return node_label_to_cluster[text]
        return None

    @staticmethod
    def _allocate_super_ids(graph, cluster_ids, kind: str):
        """Pick super-node ids type-compatible with the existing node ids.

        ``sorted(self.nodes())`` is called by ``canonical_code`` and other
        graph routines; in Python 3 mixing ``str`` and ``int`` ids in the
        same graph raises ``TypeError``. This helper inspects the existing
        ids and:

        * if they are all integers (the common case for graphs loaded from
          ``.data`` files), returns large negative integers chosen far
          enough from the natural id range to avoid collisions;
        * otherwise falls back to the string ``__semantic_<kind>_cluster_<cid>__``
          form.

        The two families (``kind="node"`` and ``kind="edge"``) get disjoint
        offsets so node-cluster and edge-cluster super-nodes can coexist.
        """
        cluster_ids = sorted(cluster_ids)
        if not cluster_ids:
            return {}

        existing_ids = list(graph.nodes())
        existing_types = {type(n) for n in existing_ids}

        use_int = (not existing_types) or existing_types == {int}

        if not use_int:
            prefix = (
                _NODE_SUPERNODE_PREFIX if kind == "node" else _EDGE_SUPERNODE_PREFIX
            )
            return {cid: f"{prefix}{cid}__" for cid in cluster_ids}

        # Allocate ints far below any natural id, with disjoint base per
        # family so collisions across families are impossible.
        base = -10 ** 12 if kind == "node" else -2 * 10 ** 12
        allocated = {}
        candidate = base
        for cid in cluster_ids:
            while candidate in allocated.values() or graph.has_node(candidate):
                candidate -= 1
            allocated[cid] = candidate
            candidate -= 1
        return allocated

    @staticmethod
    def _clone_graph(graph):
        """Build a fresh graph instance with the same nodes and edges.

        Mirrors the constructor-handling logic used by the flexible_subgraph
        embedding strategy so the helper works for both base graph classes
        and the DB graph wrappers, which require ``(graph, name)``.
        """
        graph_cls = type(graph)
        try:
            new_graph = graph_cls()
        except TypeError:
            new_graph = graph_cls(None, getattr(graph, "name", None))

        # Preserve the original name so downstream code (graph_names alignment,
        # output emission) keeps working transparently.
        if hasattr(graph, "get_name"):
            try:
                new_graph.name = graph.get_name()
            except AttributeError:
                pass

        for node_id, node_data in graph.nodes(data=True):
            new_graph.add_node(node_id, **dict(node_data))
        for src, dst, key, edge_data in graph.edges(keys=True, data=True):
            new_graph.add_edge(src, dst, key=key, **dict(edge_data))

        return new_graph
