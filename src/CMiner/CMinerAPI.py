from asyncio import ALL_COMPLETED
from collections import defaultdict
from threading import Thread
import os
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import threading


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
from .SolutionSaver import QueueSolutionSaver
from .SpillManager import close_spill_manager, configure_spill
from .Stack import DFSStack


class CMinerAPI:

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
        string_output: bool = False,  # whether to return results as a string instead of printing
        streaming_output: bool = False,  # whether to yield patterns one-by-one instead of accumulating them
        lazy_dfs: bool = False,  # whether to traverse with lazy DFS (O(depth) live patterns) instead of the stack frontier (O(depth x siblings))
        spill_mappings: bool = True,  # whether to spill idle occurrence mappings to disk (LRU, budget-based) to bound RAM
        spill_max_loaded: int = 4_000_000,  # budget of live mapping links (occurrences x pattern depth) before eviction
        spill_dir: str | None = None,  # directory for spill files (default: fresh temp dir)
        two_pass_discovery: bool = True,  # count-first extension discovery (cuts the candidate-tuple accumulator peak RAM; ~2x candidate-scan CPU)
    ):
        # Two-pass extension discovery toggle. Env override: CMINER_TWO_PASS=0/1.
        from .EdgeExtension import set_two_pass

        env_tp = os.environ.get("CMINER_TWO_PASS")
        set_two_pass(
            two_pass_discovery
            if env_tp is None
            else env_tp.strip().lower() not in ("0", "false", "no", "off")
        )
        # Configure the process-wide spill manager for occurrence mappings.
        # Env overrides: CMINER_SPILL, CMINER_SPILL_MAX_LINKS, CMINER_SPILL_DIR.
        # When disabled (or never configured, e.g. legacy CMiner class),
        # PatternMappings stays purely in-RAM.
        configure_spill(
            enabled=spill_mappings,
            max_loaded=spill_max_loaded,
            directory=spill_dir,
        )
        self.db_file = db_file
        self.string_output = string_output
        self.streaming_output = streaming_output
        self.lazy_dfs = lazy_dfs
        self.stack = DFSStack(
            min_nodes,
            max_nodes,
            {
                "directed_graph": directed_graph,
                "show_mappings": show_mappings,
                "with_frequencies": with_frequencies,
                "pattern_type": pattern_type,
                "output_path": output_path,
                "string_output": string_output,
                "streaming_output": streaming_output,
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
        self._active_tasks = 0
        self._active_lock = threading.Lock()

    def set_db_file(self, db_file: str):
        self.db_file = db_file

    def set_min_support(self, support: float):
        self.min_support = support

    def set_min_nodes(self, min_nodes: int):
        if min_nodes < 0:
            raise ValueError("Minimum nodes must be at least 0.")
        elif min_nodes > self.stack.max_nodes:
            raise ValueError(
                "Minimum nodes cannot be greater than maximum nodes."
            )
        max_nodes = self.stack.max_nodes
        self.stack = DFSStack(
            min_nodes,
            max_nodes,
            {
                "directed_graph": self.directed_graph,
                "show_mappings": self.show_mappings,
                "with_frequencies": self.with_frequencies,
                "pattern_type": self.pattern_type,
                "output_path": self.output_path,
                "string_output": self.string_output,
                "streaming_output": self.streaming_output,
            },
        )

    def set_max_nodes(self, max_nodes: int):
        if max_nodes < self.stack.min_nodes:
            raise ValueError(
                "Maximum nodes cannot be less than minimum nodes."
            )
        min_nodes = self.stack.min_nodes
        self.stack = DFSStack(
            min_nodes,
            max_nodes,
            {
                "directed_graph": self.directed_graph,
                "show_mappings": self.show_mappings,
                "with_frequencies": self.with_frequencies,
                "pattern_type": self.pattern_type,
                "output_path": self.output_path,
                "string_output": self.string_output,
                "streaming_output": self.streaming_output,
            },
        )

    def _worker_wrapper(self, worker_fn, pattern):
        with self._active_lock:
            self._active_tasks += 1
        try:
            worker_fn(pattern)
        except Exception as e:
            print(f"EXCEPTION in worker: {e}")
            import traceback

            traceback.print_exc()
        finally:
            with self._active_lock:
                self._active_tasks -= 1

    def get_info(self) -> str:
        """
        Show the information about the CMiner instance.
        """
        directed_graph = "Directed" if self.directed_graph else "Undirected"
        info_str = f"CMiner Instance Info:\n"
        info_str += f"DB file ({directed_graph}): {self.db_file}\n"
        info_str += f"Minimum support: {self.min_support}\n"
        if self.stack.min_nodes > 1:
            info_str += f"Minimum nodes: {self.stack.min_nodes}\n"
        if self.stack.max_nodes < float("inf"):
            info_str += f"Maximum nodes: {self.stack.max_nodes}\n"
        if self.templates_file:
            info_str += f"Start patterns: {self.templates_file}\n"
        if self.output_path:
            info_str += f"Output file: {self.output_path}\n"
        return info_str

    def mine(self) -> str | None:
        if self.streaming_output:
            raise RuntimeError(
                "streaming_output=True is set. Use mine_stream() instead of mine()."
            )
        print(self.get_info())
        print("Reading graphs from file...", end=" ")
        self._read_graphs_from_file()
        print("done.")
        self._parse_support()

        start = time.time()
        self._find_start_patterns()

        if self.pattern_type == "all":
            self._all_patterns_target()()
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

        if self.string_output:
            result = self.stack.get_string_results()
            self.stack.close()
            return result
        self.stack.close()
        return None

    def mine_stream(
        self,
        pattern_type=None,
        start_patterns_done: bool = False,
        yield_strings: bool = True,
    ):
        """
        Generator that yields patterns one at a time while mining runs in the
        background. This avoids building the full result string in memory.

        Parameters
        ----------
        pattern_type : str, optional
            Override the instance pattern type. Defaults to self.pattern_type.
        start_patterns_done : bool
            If False, this method reads the graphs, parses support, and finds
            the start patterns before streaming. If True, it assumes the caller
            has already done that (e.g. called _read_graphs_from_file,
            _parse_support and _find_start_patterns).
        yield_strings : bool
            If True (default), yields formatted strings. If False, yields the
            raw Pattern objects so the caller can format or store them itself.

        Yields
        ------
        str or Pattern
            The next mined pattern, either as a formatted string or as a
            Pattern object.

        Raises
        ------
        TypeError
            If streaming_output was not set to True when constructing the
            CMinerAPI instance.
        """
        if not isinstance(self.stack.solution_saver, QueueSolutionSaver):
            raise TypeError(
                "mine_stream() requires streaming_output=True when constructing "
                "the CMinerAPI instance."
            )

        if not start_patterns_done:
            print(self.get_info())
            print("Reading graphs from file...", end=" ")
            self._read_graphs_from_file()
            print("done.")
            self._parse_support()
            self._find_start_patterns()

        pt = pattern_type or self.pattern_type
        if pt == "all":
            target = self._all_patterns_target()
        elif pt == "maximum":
            target = self.mine_maximum_patterns
        else:
            raise ValueError(
                f"Unknown pattern type: {pt}. Expected 'all' or 'maximum'."
            )

        def run():
            try:
                target()
            finally:
                self.stack.close()

        thread = Thread(target=run, daemon=True)
        thread.start()

        try:
            yield from self.stack.iter_results(yield_strings=yield_strings)
        finally:
            self.stack.close()
            thread.join(timeout=1)

    def _all_patterns_target(self):
        """Pick the 'all' mining strategy: lazy DFS when enabled, stack otherwise."""
        if self.lazy_dfs:
            return self.mine_all_patterns_lazy
        return self.mine_all_patterns

    def mine_all_patterns(self):
        """
        Depth-First Search (DFS) for 'all' pattern mining.
        Pushes EVERY child of a popped pattern
        onto the DFS stack before any of them is processed, so the number of
        live patterns is O(depth x siblings-per-level) and each one pins its
        full occurrence mappings in RAM.
        """
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = set()
            while True:
                self._schedule_patterns(executor, futures, self._process_pattern_all)

                if not futures and self.stack.is_empty():
                    break
                done, futures = wait(futures, return_when=FIRST_COMPLETED)

    def mine_all_patterns_lazy(self):
        """
        Lazy Depth-First Search (DFS) for 'all' pattern mining.
        Visits each child immediately after creating it, so only
        the current root-to-leaf path (O(depth) patterns) is alive at any
        moment.
        Consume less memory than mine_all_patterns(),
        but runs serially (the workers setting is ignored).
        """
        while True:
            pattern = self.stack.try_pop()
            if pattern is None:
                break
            self._visit_lazy(pattern)

    def _visit_lazy(self, pattern: Pattern):
        # Mirror of _process_pattern_all, but recurses immediately instead of
        # pushing children onto the shared stack.
        if len(pattern.nodes()) >= self.stack.max_nodes:
            return

        # Pin the occurrence mappings for the whole visit: extensions key
        # their locations on Mapping object identities, so the set must not
        # be spilled/reloaded (new identities) until the visit is over.
        pattern.pattern_mappings.pin()
        try:
            node_extensions = pattern.find_node_extensions(self.min_support)

            if len(node_extensions) == 0:
                return

            for node_ext in node_extensions:

                node_extended_pattern = pattern.apply_node_extension(node_ext)

                if self.stack.was_stacked(node_extended_pattern):
                    del node_extended_pattern
                    continue

                node_extended_pattern.update_node_mappings(node_ext)

                # register() fingerprints + outputs the pattern and reports whether
                # it was accepted (new, within bounds, has occurrences).
                if not self.stack.register(node_extended_pattern):
                    continue

                node_extended_pattern.pattern_mappings.pin()
                try:
                    edge_extensions = node_extended_pattern.find_edge_extensions(
                        self.min_support
                    )

                    for edge_ext in edge_extensions:

                        edge_extended_pattern = node_extended_pattern.apply_edge_extension(
                            edge_ext
                        )

                        if self.stack.was_stacked(edge_extended_pattern):
                            del edge_extended_pattern
                            continue

                        edge_extended_pattern.update_edge_mappings(edge_ext)

                        if not self.stack.register(edge_extended_pattern):
                            continue

                        self._visit_lazy(edge_extended_pattern)
                finally:
                    node_extended_pattern.pattern_mappings.unpin()

                self._visit_lazy(node_extended_pattern)
        finally:
            pattern.pattern_mappings.unpin()

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
                with self._active_lock:
                    active = self._active_tasks
                if not futures and active == 0 and self.stack.is_empty():
                    break
                done, futures = wait(futures, return_when=FIRST_COMPLETED)

    def _schedule_patterns(self, executor: ThreadPoolExecutor, futures, worker_fn):
        while len(futures) < self.workers:
            pattern = self.stack.try_pop()
            if pattern is None:
                break
            futures.add(executor.submit(self._worker_wrapper, worker_fn, pattern))

    def _process_pattern_all(self, pattern_to_extend: Pattern):

        if len(pattern_to_extend.nodes()) >= self.stack.max_nodes:
            return

        # Pin the occurrence mappings for the whole processing step: the
        # extension objects found here key their locations on the Mapping
        # object identities, so the set must not be spilled (and reloaded
        # with different identities) until the step is over.
        pattern_to_extend.pattern_mappings.pin()
        try:
            node_extensions = pattern_to_extend.find_node_extensions(self.min_support)

            if len(node_extensions) == 0:
                return

            for node_ext in node_extensions:

                node_extended_pattern = pattern_to_extend.apply_node_extension(node_ext)

                if self.stack.was_stacked(node_extended_pattern):
                    continue

                node_extended_pattern.update_node_mappings(node_ext)

                self.stack.push(node_extended_pattern)

                # Same pinning for the edge-extension phase: update_edge_mappings
                # membership-tests Mapping identities collected by
                # find_edge_extensions on this pattern's mappings.
                node_extended_pattern.pattern_mappings.pin()
                try:
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
                finally:
                    node_extended_pattern.pattern_mappings.unpin()
        finally:
            pattern_to_extend.pattern_mappings.unpin()

    def _process_pattern_maximum(self, pattern_to_extend: Pattern):
        if len(pattern_to_extend.nodes()) >= self.stack.max_nodes:
            return

        # Pin the occurrence mappings for the whole step (see _process_pattern_all).
        pattern_to_extend.pattern_mappings.pin()
        try:
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

                node_extended_pattern.pattern_mappings.pin()
                try:
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
                finally:
                    node_extended_pattern.pattern_mappings.unpin()
        finally:
            pattern_to_extend.pattern_mappings.unpin()

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
            # Start patterns' occurrence sets are complete: seal them
            # (immutable, evictable by the SpillManager when not pinned).
            p.pattern_mappings.seal(depth=len(p.nodes()))
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
                # Seal the 1-node pattern occurrence set (see _find_start_patterns).
                pattern_mappings.seal(depth=1)
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
        Close the solution saver and release spill resources (temp files).
        The spill configuration is kept: a new manager is created lazily on
        the next mining run.
        """
        self.stack.close()
        close_spill_manager()

    def get_structured_patterns(self) -> list[dict]:
        """
        Return structured pattern data including per-occurrence node/edge mappings.
        Each dict contains: nodes, edges, support, frequency, mappings.
        The 'mappings' key is a dict: {graph_name: [occurrence, ...]} where each
        occurrence has 'nodes' {pattern_node: target_node} and 'edges' [target_edge].
        """
        results = []
        for pattern in self.stack.output_structured_patterns:
            nodes = []
            for node_id, data in pattern.nodes(data=True):
                labels = list(data.get("labels", []))
                nodes.append((node_id, labels))

            edges = []
            for u, v, key, data in pattern.edges(keys=True, data=True):
                edge_type = data.get("type", "")
                edges.append((u, v, edge_type))

            mappings = {}
            pm = pattern.pattern_mappings
            for graph in pm.graphs():
                graph_name = graph.get_name()
                occurrences = []
                for mapping in pm.mappings(graph):
                    node_map = mapping.nodes_mapping()
                    edge_map = mapping._retrieve_edge_mapping()
                    occurrences.append({
                        "nodes": dict(node_map),
                        "edges": [
                            {"src": t_src, "dst": t_dst, "key": t_key}
                            for (p_src, p_dst, p_key), (t_src, t_dst, t_key) in edge_map.items()
                        ],
                    })
                mappings[graph_name] = occurrences

            results.append({
                "nodes": nodes,
                "edges": edges,
                "support": pattern.support(),
                "frequency": pattern.frequency(),
                "mappings": mappings,
            })
        return results