from collections import defaultdict
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from numpy import delete

# from networkx.algorithms import isomorphism as iso

from Graph.DBGraph import DirectedDBGraph, UndirectedDBGraph
from MultiGraphMatch.MultiGraphMatch import (
    Mapping,
    MultiGraphMatch,
    BitMatrixStrategy2,
    TargetBitMatrixOptimized,
)
from NetworkLoader.NetworkConfigurator import NetworkConfigurator
from NetworkLoader.NetworksLoading import NetworksLoading

from .Pattern import DirectedPattern, Pattern, PatternMappings, UndirectedPattern
from .Stack import DFSStack


class CMiner:

    def __init__(
        self,
        db_file,  # path to the file containing the graphs
        support: float,  # minimum support
        min_nodes=1,  # minimum number of nodes in the pattern
        max_nodes=float("inf"),  # maximum number of nodes in the pattern
        templates_file=None,  # path to the file containing the templates
        show_mappings=False,  # whether to show mappings or not in the output
        output_path=None,  # path to the output file
        directed_graph=False,  # whether the graph is directed or not
        with_frequencies=False,  # whether to show frequencies or not in the output
        pattern_type="all",  # type of pattern to mine: 'all' or 'maximum'
        workers: int = 1,  # number of parallel workers
    ):
        self.db_file = db_file
        self.stack = DFSStack(
            min_nodes,
            max_nodes,
            {
                "directed_graph": directed_graph,
                "show_mappings": show_mappings,
                "with_frequencies": with_frequencies,
                "pattern_type": pattern_type,
                "output_path": output_path,
            },
        )
        self.min_support = support
        self.templates_file = templates_file
        self.db = []
        self.show_mappings = show_mappings
        self.output_path = output_path
        self.directed_graph = directed_graph
        self.with_frequencies = with_frequencies
        self.pattern_type = pattern_type
        self.workers = max(1, int(workers))

    def show_info(self):
        """
        Show the information about the CMiner instance.
        """
        directed_graph = "Directed" if self.directed_graph else "Undirected"
        print(f"DB file ({directed_graph}): {self.db_file}")
        print(f"Minimum support: {self.min_support}")
        if self.stack.min_nodes > 1:
            print(f"Minimum nodes: {self.stack.min_nodes}")
        if self.stack.max_nodes < float("inf"):
            print(f"Maximum nodes: {self.stack.max_nodes}")
        if self.templates_file:
            print(f"Start patterns: {self.templates_file}")
        if self.output_path:
            print(f"Output file: {self.output_path}")
        print()

    def mine(self):
        self.show_info()
        print("Reading graphs from file...", end=" ")
        self._read_graphs_from_file()
        print("done.")
        self._parse_support()

        start = time.time()
        self._find_start_patterns()
        if self.pattern_type == "all":
            self.mine_all_patterns()
        elif self.pattern_type == "maximum":
            self.mine_maximum_patterns()
        else:
            raise ValueError(
                "Unknown pattern type: "
                f"{self.pattern_type}. Expected 'all' or 'maximum'."
            )
        end = time.time()
        elapsed_time = end - start
        hours, seconds = divmod(elapsed_time, 3600)
        print(f"Mining completed in {int(hours)} hours and {seconds:.2f} seconds.")
        self.stack.close()

    def mine_all_patterns(self):
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = set()
            while True:
                self._schedule_patterns(executor, futures, self._process_pattern_all)
                if not futures:
                    break
                done, futures = wait(futures, return_when=FIRST_COMPLETED)

    def mine_maximum_patterns(self):
        if not self.db:
            self._read_graphs_from_file()
            self._find_start_patterns()
            self._parse_support()

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = set()
            while True:
                self._schedule_patterns(
                    executor, futures, self._process_pattern_maximum
                )
                if not futures:
                    break
                done, futures = wait(futures, return_when=FIRST_COMPLETED)

    def _schedule_patterns(self, executor, futures, worker_fn):
        while len(futures) < self.workers:
            pattern = self.stack.try_pop()
            if pattern is None:
                break
            futures.add(executor.submit(worker_fn, pattern))

    def _process_pattern_all(self, pattern_to_extend: Pattern):
        if len(pattern_to_extend.nodes()) >= self.stack.max_nodes:
            return

        node_extensions = pattern_to_extend.find_node_extensions(self.min_support)

        if len(node_extensions) == 0:
            return

        for node_ext in node_extensions:

            node_extended_pattern = pattern_to_extend.apply_node_extension(node_ext)

            if self.stack.was_stacked(node_extended_pattern):
                continue

            node_extended_pattern.update_node_mappings(node_ext)

            self.stack.push(node_extended_pattern)

            edge_extensions = node_extended_pattern.find_edge_extensions(
                self.min_support
            )

            for edge_ext in edge_extensions:

                edge_extended_pattern = node_extended_pattern.apply_edge_extension(
                    edge_ext
                )

                if self.stack.was_stacked(edge_extended_pattern):
                    continue

                edge_extended_pattern.update_edge_mappings(edge_ext)

                self.stack.push(edge_extended_pattern)

    def _process_pattern_maximum(self, pattern_to_extend: Pattern):
        if len(pattern_to_extend.nodes()) >= self.stack.max_nodes:
            return

        node_extensions = pattern_to_extend.find_node_extensions(self.min_support)

        if len(node_extensions) == 0:
            # In maximum pattern mining, patterns that cannot be extended are outputs.
            self.stack.output(pattern_to_extend)
            return

        for node_ext in node_extensions:

            node_extended_pattern = pattern_to_extend.apply_node_extension(node_ext)

            # Ensure no duplicate patterns are processed
            if self.stack.was_stacked(node_extended_pattern):
                continue

            node_extended_pattern.update_node_mappings(node_ext)

            tree_pattern_added = False

            edge_extensions = node_extended_pattern.find_edge_extensions(
                self.min_support
            )

            # If no edge extensions are found, add the pattern
            # to the stack, it could be extended adding a node
            if len(edge_extensions) == 0:
                self.stack.push(node_extended_pattern)
                continue

            graphs_covered_by_edge_extensions = {
                g for edge_ext in edge_extensions for g in edge_ext[0].graphs()
            }

            for edge_ext in edge_extensions:

                edge_extended_pattern = node_extended_pattern.apply_edge_extension(
                    edge_ext
                )

                if self.stack.was_stacked(edge_extended_pattern):
                    continue

                edge_extended_pattern.update_edge_mappings(edge_ext)

                # If the support of the tree pattern is greater than the cycle pattern
                # it means that the tree cannot be closed in a cycle for all of his
                # occurrence in each graph, so it's considered the tree pattern and added to the stack.
                # Also check if the pattern is not already in the stack, because the same tree can be
                # considered with more than one edge extension.
                if (
                    (not tree_pattern_added)
                    and (
                        node_extended_pattern.support()
                        > len(graphs_covered_by_edge_extensions)
                    )
                    and (
                        node_extended_pattern.support()
                        > edge_extended_pattern.support()
                    )
                ):
                    self.stack.push(node_extended_pattern)
                    tree_pattern_added = True

                self.stack.push(edge_extended_pattern)

    def _pattern_factory(self, graph) -> Pattern:
        p = (
            DirectedPattern(PatternMappings())
            if self.directed_graph
            else UndirectedPattern(PatternMappings())
        )

        for node, node_data in graph.nodes(data=True):
            p.add_node(node, **node_data)
        for u, v, key, edge_data in graph.edges(keys=True, data=True):
            p.add_edge(u, v, key=key, **edge_data)
        return p

    def _find_start_patterns(self):

        # If no templates file is provided, mine 1-node patterns
        if self.templates_file is None:
            for p in self._mine_1node_patterns():
                self.stack.push(p)
            return
        # Otherwise, load templates from the provided file
        pattern_dict = {}
        templates = self._templates()
        for g in self.db:
            matcher = MultiGraphMatch(g)
            for t in templates:
                mappings = matcher.match(t)
                if len(mappings) == 0:
                    continue
                if t.miss_some_labels():
                    # if the template miss some labels, when constructing the pattern
                    # we must consider the canonical code of each mapped graph because
                    # it can be different
                    for m in mappings:
                        mapped_g = m.get_mapped_graph(g).zero_index_graph()
                        mapped_g_code = mapped_g.canonical_code()
                        if not (mapped_g_code in pattern_dict):
                            pattern_dict[mapped_g_code] = self._pattern_factory(
                                mapped_g
                            )

                        p = pattern_dict[mapped_g_code]
                        p.pattern_mappings.add_mapping(g, m)
                else:
                    t_code = t.canonical_code()
                    if not (t_code in pattern_dict):
                        pattern_dict[t_code] = self._pattern_factory(t)

                    p = pattern_dict[t_code]
                    p.pattern_mappings.set_mapping(g, mappings)

            del matcher
        for p in pattern_dict.values():
            if p.support() >= self.min_support:
                self.stack.push(p)

    def _mine_1node_patterns(self) -> list[Pattern]:
        # Costruisci un dizionario che mappa un pattern di etichette (tuple)
        # ad un dizionario: {graph: [lista dei nodi che hanno questo pattern]}
        counter = defaultdict(lambda: defaultdict(list))

        pattern_constructor = (
            DirectedPattern if self.directed_graph else UndirectedPattern
        )

        for g in self.db:
            for node in g.nodes():
                # Recupera le etichette del nodo una sola volta
                node_labels = g.get_node_labels(node)
                # Se le etichette sono già ordinate, convertili in tuple per usarle come chiavi
                labels_tuple = tuple(node_labels)
                counter[labels_tuple][g].append(node)

        patterns = []
        # Itera sui pattern raccolti
        for labels_tuple, graph_nodes in counter.items():
            # Se il pattern appare in un numero di grafi maggiore o uguale a self.min_support
            if len(graph_nodes) >= self.min_support:
                pattern_mappings = PatternMappings()
                p = pattern_constructor(pattern_mappings)
                # Aggiungi il nodo modello, usando la lista delle etichette (convertita da tuple a list)
                p.add_node(0, labels=list(labels_tuple))
                # Per ogni grafo, aggiungi le mapping per tutti i nodi che hanno questo pattern
                for g, nodes in graph_nodes.items():
                    mappings = [Mapping(node_mapping={0: node}) for node in nodes]
                    p.pattern_mappings.set_mapping(g, mappings)
                patterns.append(p)

        return patterns

    def _read_graphs_from_file(self):
        constructor_db_graph = (
            DirectedDBGraph if self.directed_graph else UndirectedDBGraph
        )
        type_file = self.db_file.split(".")[-1]
        configurator = NetworkConfigurator(self.db_file, type_file)
        for name, network in NetworksLoading(
            type_file, configurator.config, self.directed_graph
        ).Networks.items():
            self.db.append(constructor_db_graph(network, name))

    def _templates(self) -> list[Pattern]:
        """
        Retrieve the starting templates.
        """
        templates = []
        constructor_template = (
            DirectedPattern if self.directed_graph else UndirectedPattern
        )
        type_file = self.db_file.split(".")[-1]
        configurator = NetworkConfigurator(self.templates_file, type_file)
        for _, network in NetworksLoading(
            type_file, configurator.config, self.directed_graph
        ).Networks.items():
            template = constructor_template(pattern_mappings=PatternMappings())
            # Copy nodes
            for node, node_data in network.nodes(data=True):
                template.add_node(node, **node_data)
            # Copy edges
            for u, v, key, edge_data in network.edges(keys=True, data=True):
                template.add_edge(u, v, key=key, **edge_data)
            templates.append(template)
        return templates

    def _parse_support(self):
        """
        If the support is > 1, then the user want common
        graphs that are present in a certain amount of
        db graphs.
        If the support is <= 1 then the user want common
        graphs that are present in a certain percentage of
        df graphs.
        """
        if self.min_support > 1:
            self.min_support = int(self.min_support)
        if self.min_support <= 1:
            db_len = len(self.db)
            self.min_support = int(self.min_support * db_len)

    def close(self):
        """
        Close the solution saver.
        """
        self.stack.close()
