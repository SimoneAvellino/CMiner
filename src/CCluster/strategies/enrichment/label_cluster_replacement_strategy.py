"""Label-cluster replacement enrichment strategy.

Performs the same semantic label clustering as
:class:`SemanticLabelClusteringEnrichmentStrategy` (SentenceTransformer
embeddings + K-Means grid search + silhouette-based outlier handling), but
instead of adding synthetic super-nodes and super-edges it *replaces* each
node label and each edge type in-place with the name of its cluster
(``semantic_cluster_<id>``).

This produces graphs that have the same structure as the originals but whose
labels are collapsed into their semantic groups — useful when you want the
embedding step to see coarser, cluster-level labels rather than extra
super-node connectivity.

Strategy parameters (``context.strategy_params``) are identical to the parent:

``model_name``       SentenceTransformer model id (default ``all-MiniLM-L6-v2``).
``random_state``     Seed for K-Means (default ``42``).
``max_k``            Override for the upper bound of the grid search.
``verbose``          Bool, prints progress logs (default ``False``).
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

from .enrichment_strategy import EnrichmentContext
from .semantic_label_clustering_strategy import (
    SemanticLabelClusteringEnrichmentStrategy,
    _SUPERNODE_LABEL_TEMPLATE,
)

_TAG = "[label_cluster_replacement]"


class LabelClusterReplacementEnrichmentStrategy(
    SemanticLabelClusteringEnrichmentStrategy
):
    """Replace node/edge labels with their semantic cluster name.

    Inherits all clustering helpers from
    :class:`SemanticLabelClusteringEnrichmentStrategy`; only the
    per-graph augmentation step differs.
    """

    @property
    def name(self) -> str:
        return "label_cluster_replacement"

    # ------------------------------------------------------------------ main

    def enrich(self, context: EnrichmentContext) -> List:
        verbose = bool(context.strategy_params.get("verbose", False))
        model_name = context.strategy_params.get("model_name", "all-MiniLM-L6-v2")
        random_state = int(context.strategy_params.get("random_state", 42))
        max_k_override = context.strategy_params.get("max_k", None)

        graphs = list(context.db_graphs)
        if not graphs:
            return graphs

        unique_node_labels = self._collect_unique_node_labels(graphs)
        unique_edge_labels = self._collect_unique_edge_labels(graphs)

        if verbose:
            print(
                f"{_TAG} unique node labels: {len(unique_node_labels)} | "
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
            print(f"{_TAG} replacing labels in {len(graphs)} graphs...")

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
                print(
                    f"{_TAG}   {index}/{len(graphs)} ({graph_name}): labels replaced",
                    flush=True,
                )
                last_log = time.time()

        if verbose:
            elapsed = time.time() - t_start
            print(f"{_TAG} {len(enriched)} graphs enriched in {elapsed:.1f}s.")

        return enriched

    # ----------------------------------------------------- per-graph replace

    def _enrich_graph(
        self,
        graph,
        node_label_to_cluster: Dict[str, int],
        edge_label_to_cluster: Dict[str, int],
    ):
        """Return a copy of *graph* with labels replaced by cluster names.

        Each node label that has a known cluster mapping is replaced with
        ``semantic_cluster_<id>``; labels that are not in the mapping (e.g.
        from the empty / null-label pool) are kept as-is.  Duplicate cluster
        names produced by collapsing multiple labels of the same node are
        deduplicated while preserving order.

        Edge types follow the same rule: a known type is replaced by its
        cluster name; unknown types are left unchanged.
        """
        enriched = self._clone_graph(graph)

        # --- replace node labels -------------------------------------------
        for node_id, node_data in list(enriched.nodes(data=True)):
            old_labels = node_data.get("labels", []) or []
            new_labels: List[str] = []
            seen: set = set()
            for label in old_labels:
                if label is None:
                    continue
                text = str(label).strip()
                cluster_id = node_label_to_cluster.get(text)
                if cluster_id is not None:
                    replacement = _SUPERNODE_LABEL_TEMPLATE.format(
                        cluster_id=cluster_id
                    )
                else:
                    # Keep labels that have no semantic mapping.
                    replacement = text if text else label
                if replacement not in seen:
                    new_labels.append(replacement)
                    seen.add(replacement)
            enriched.nodes[node_id]["labels"] = new_labels

        # --- replace edge types --------------------------------------------
        for src, dst, key, edge_data in list(enriched.edges(keys=True, data=True)):
            edge_type = edge_data.get("type", None)
            if edge_type is None:
                continue
            text = str(edge_type).strip()
            if not text:
                continue
            cluster_id = edge_label_to_cluster.get(text)
            if cluster_id is not None:
                enriched[src][dst][key]["type"] = _SUPERNODE_LABEL_TEMPLATE.format(
                    cluster_id=cluster_id
                )

        return enriched
