import os
import re

from sklearn.metrics import silhouette_score

from .strategies.clustering import (
    AgglomerativeClusteringStrategy,
    ClusteringContext,
    ClusteringStrategy,
    KMedoidsClusteringStrategy,
    SpectralClusteringStrategy,
)
from .strategies.embedding import (
    DistanceMatrixStrategyContext,
    FlexibleSubgraphDistanceStrategy,
    GraphDistanceStrategy,
    MCSDistanceStrategy,
    SimpleStructuralDistanceStrategy,
)
from .strategies.enrichment import (
    EnrichGraphSemanticsStrategy,
    EnrichmentContext,
    LabelClusterReplacementEnrichmentStrategy,
    NoOpEnrichmentStrategy,
    SemanticLabelClusteringEnrichmentStrategy,
)


class CCluster:
    """Pipeline orchestrator: read graphs -> enrich -> embed -> cluster -> emit.

    The three pluggable steps are:

    * ``enrichment_strategy``: an :class:`EnrichGraphSemanticsStrategy` that
      augments the semantic content of the graphs read from the database.
      Defaults to a no-op identity strategy.
    * ``embedding_strategy``: a :class:`GraphDistanceStrategy` that turns the
      (possibly enriched) graphs into a pairwise distance matrix.
    * ``clustering_strategy``: a :class:`ClusteringStrategy` that partitions
      the graphs based on the distance matrix. Defaults to k-medoids.
    """

    def __init__(
        self,
        db_file,
        num_clusters,
        directed_graph=0,
        output_path=None,
        embedding_strategy="simple_structural",
        clustering_strategy="kmedoids",
        enrichment_strategy="noop",
        init_method="random",
        max_iter=100,
        tolerance=1e-4,
        auto_k_max=None,
        embedding_params=None,
        clustering_params=None,
        enrichment_params=None,
    ):
        self.db_file = db_file
        self.num_clusters = num_clusters
        self.directed_graph = directed_graph
        self.output_path = output_path
        self.embedding_strategy = embedding_strategy
        self.clustering_strategy = clustering_strategy
        self.enrichment_strategy = enrichment_strategy
        self.init_method = init_method
        self.max_iter = max_iter
        self.tolerance = tolerance
        # Upper bound for the auto-K silhouette grid search.
        # ``None`` (default) -> ceil(sqrt(n_items)) heuristic.
        self.auto_k_max = auto_k_max
        self.embedding_params = embedding_params or {}
        self.clustering_params = clustering_params or {}
        self.enrichment_params = enrichment_params or {}
        self.db = []

    # --------------------------------------------------------------- helpers

    @staticmethod
    def _silhouette_score(distance_matrix, assignments, n_items):
        labels = [-1] * n_items
        for cluster_idx, points in enumerate(assignments):
            for point in points:
                labels[point] = cluster_idx

        # Ignore failed candidate clusterings (for example when empty clusters
        # reduce the number of effective labels below 2).
        try:
            return float(
                silhouette_score(distance_matrix, labels, metric="precomputed")
            )
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

    # ------------------------------------------------------ strategy resolvers

    def _resolve_enrichment_strategy(self) -> EnrichGraphSemanticsStrategy:
        if isinstance(self.enrichment_strategy, EnrichGraphSemanticsStrategy):
            return self.enrichment_strategy

        registry = {
            "noop": NoOpEnrichmentStrategy(),
            "semantic_label_clustering": SemanticLabelClusteringEnrichmentStrategy(),
            "label_cluster_replacement": LabelClusterReplacementEnrichmentStrategy(),
        }
        if self.enrichment_strategy not in registry:
            available = ", ".join(sorted(registry))
            raise ValueError(
                f"Unknown enrichment strategy '{self.enrichment_strategy}'. "
                f"Available strategies: {available}"
            )
        return registry[self.enrichment_strategy]

    def _resolve_embedding_strategy(self) -> GraphDistanceStrategy:
        if isinstance(self.embedding_strategy, GraphDistanceStrategy):
            return self.embedding_strategy

        registry = {
            "simple_structural": SimpleStructuralDistanceStrategy(),
            "flexible_subgraph": FlexibleSubgraphDistanceStrategy(),
            "mcs": MCSDistanceStrategy(),
        }
        if self.embedding_strategy not in registry:
            available = ", ".join(sorted(registry))
            raise ValueError(
                f"Unknown embedding strategy '{self.embedding_strategy}'. "
                f"Available strategies: {available}"
            )
        return registry[self.embedding_strategy]

    def _resolve_clustering_strategy(self) -> ClusteringStrategy:
        if isinstance(self.clustering_strategy, ClusteringStrategy):
            return self.clustering_strategy

        registry = {
            "kmedoids": KMedoidsClusteringStrategy(),
            "agglomerative": AgglomerativeClusteringStrategy(),
            "spectral": SpectralClusteringStrategy(),
        }
        if self.clustering_strategy not in registry:
            available = ", ".join(sorted(registry))
            raise ValueError(
                f"Unknown clustering strategy '{self.clustering_strategy}'. "
                f"Available strategies: {available}"
            )
        return registry[self.clustering_strategy]

    # ------------------------------------------------------------ auto-tuning

    def _select_num_clusters_auto(
        self, clustering_strategy, distance_matrix, graph_names
    ):
        import math

        n_items = len(graph_names)
        if n_items == 0:
            raise ValueError("Graph database is empty.")

        if n_items <= 2:
            print(
                "Auto mode fallback: silhouette needs at least 3 graphs; "
                f"using num_clusters={n_items}."
            )
            return n_items

        # Verbose flag is consistent with the verbose flag of the clustering
        # strategy itself; when on we print one line per candidate k so the
        # user can see auto-mode making progress on long runs.
        verbose = bool(self.clustering_params.get("verbose", False))

        # Upper bound for the silhouette grid search.
        # Default: ceil(sqrt(n_items)) — enough to find a sensible k for
        # most real DBs without paying the O(n) cost of testing every k up
        # to n_items - 1. Override with ``auto_k_max=N`` (constructor) or
        # ``--auto_k_max N`` (CLI).
        if self.auto_k_max is None:
            k_max = max(2, int(math.ceil(math.sqrt(n_items))))
        else:
            k_max = max(2, int(self.auto_k_max))
        # silhouette requires k <= n_items - 1
        k_max = min(k_max, n_items - 1)

        best_k = 2
        best_score = float("-inf")

        if verbose:
            tag = "default sqrt-heuristic" if self.auto_k_max is None else "user override"
            print(
                f"[auto_k] testing candidate_k in [2, {k_max}] on {n_items} graphs "
                f"({tag})",
                flush=True,
            )

        # Pass-through clustering params for auto-K: silence the inner kmedoids
        # logging during the grid search to avoid swamping the output with
        # iteration logs from O(n) candidate fits.
        inner_clustering_params = dict(self.clustering_params)
        inner_clustering_params["verbose"] = False

        for candidate_k in range(2, k_max + 1):
            context = ClusteringContext(
                num_clusters=candidate_k,
                init_method=self.init_method,
                max_iter=self.max_iter,
                tolerance=self.tolerance,
                strategy_params=inner_clustering_params,
            )
            _, assignments = clustering_strategy.cluster(
                distance_matrix, graph_names, context
            )

            if any(len(points) == 0 for points in assignments):
                if verbose:
                    print(
                        f"[auto_k]   k={candidate_k}: empty cluster, skipped",
                        flush=True,
                    )
                continue

            score = self._silhouette_score(distance_matrix, assignments, n_items)

            if verbose:
                marker = " <-- best" if score > best_score else ""
                print(
                    f"[auto_k]   k={candidate_k}: silhouette={score:.4f}{marker}",
                    flush=True,
                )

            if score > best_score:
                best_score = score
                best_k = candidate_k

        print(
            "Auto mode selected "
            f"num_clusters={best_k} using silhouette score={best_score:.6f}."
        )
        return best_k

    # ------------------------------------------------------------------ I/O

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
                        labels_part = " ".join(
                            str(label) for label in labels if str(label)
                        )
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

    # --------------------------------------------------------------- pipeline

    def cluster(self):
        # 1. Load graphs from the database file.
        print("Reading graphs from file...", end=" ")
        self._read_graphs_from_file()
        print("done.")

        # 2. Optionally enrich the semantic content of the graphs.
        # The enriched graphs are fed only to the embedding step; ``self.db``
        # keeps the original graphs so the output emitted on disk does NOT
        # include any synthetic semantic super-node / super-edge.
        enrichment_strategy = self._resolve_enrichment_strategy()
        enrichment_context = EnrichmentContext(
            db_graphs=self.db,
            db_file=self.db_file,
            directed_graph=bool(self.directed_graph),
            strategy_params=self.enrichment_params,
        )
        enriched_graphs = list(enrichment_strategy.enrich(enrichment_context))

        # 3. Compute the pairwise distance matrix via the embedding strategy.
        embedding_strategy = self._resolve_embedding_strategy()
        context_num_clusters = (
            2 if isinstance(self.num_clusters, str) else self.num_clusters
        )
        embedding_context = DistanceMatrixStrategyContext(
            db_graphs=enriched_graphs,
            db_file=self.db_file,
            directed_graph=bool(self.directed_graph),
            num_clusters=context_num_clusters,
            init_method=self.init_method,
            max_iter=self.max_iter,
            tolerance=self.tolerance,
            strategy_params=self.embedding_params,
        )
        distance_matrix, graph_names = embedding_strategy.compute_distance_matrix(
            embedding_context
        )

        # 4. Run the clustering algorithm on the distance matrix.
        clustering_strategy = self._resolve_clustering_strategy()

        if isinstance(self.num_clusters, str):
            if self.num_clusters != "auto":
                raise ValueError(
                    "num_clusters string value is invalid. Supported value: 'auto'."
                )
            self.num_clusters = self._select_num_clusters_auto(
                clustering_strategy, distance_matrix, graph_names
            )

        clustering_context = ClusteringContext(
            num_clusters=self.num_clusters,
            init_method=self.init_method,
            max_iter=self.max_iter,
            tolerance=self.tolerance,
            strategy_params=self.clustering_params,
        )
        clusters, _ = clustering_strategy.cluster(
            distance_matrix, graph_names, clustering_context
        )

        # 5. Emit results.
        self._emit_results(clusters)
