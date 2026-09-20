from .EdgeExtension import EdgeExtension
from .NodeExtension import (
    DirectedNodeExtensionManager,
    NodeExtension,
    NodeExtensionManager,
    UndirectedNodeExtensionManager,
)
from Graph.DBGraph import DBGraph
from Graph.DirectedMultiGraph import DirectedMultiGraph
from Graph.UndirectedMultiGraph import UndirectedMultiGraph
from MultiGraphMatch.MultiGraphMatch import Mapping

from array import array
from collections import Counter, defaultdict
from collections.abc import Sequence
import os
import errno
import threading
import zlib

from .EdgeExtension import (
    DirectedEdgeExtensionManager,
    EdgeExtensionManager,
    UndirectedEdgeExtensionManager,
)
from .Extension import DirectedExtension, Extension, UndirectedExtension
from .OccurrenceCodec import decode_node, encode_node
from .SpillManager import get_spill_manager


_ROW_TYPE = "q"


class _MappingSequence(Sequence):

    def __init__(self, pattern_mappings, graph):
        self._pattern_mappings = pattern_mappings
        self._graph = graph

    def __len__(self):
        return self._pattern_mappings.count(self._graph)

    def __iter__(self):
        for row in self._pattern_mappings.iter_rows(self._graph):
            yield self._pattern_mappings._mapping_from_row(self._graph, row)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return list(self)[index]
        if index < 0:
            index += len(self)
        return self._pattern_mappings._mapping_from_row(
            self._graph, self._pattern_mappings.row(self._graph, index)
        )


class PatternMappings:

    def __init__(self):
        """
        Keep track of each mapping for each graph of a specific pattern.

        Spill-aware container: while the mappings are being built it behaves
        like the classic {DBGraph: [Mapping]} dict. After ``seal()`` (called
        by the mining pipeline once the mappings of a pattern are complete)
        the set is immutable and becomes evictable: under memory budget
        pressure the SpillManager serializes it to a temp file and it is
        transparently reloaded on the next read access. Counts stay in RAM
        at all times, so support()/frequency() never trigger I/O.
        """
        self._mappings: dict = {}  # graph -> flat array('q') or None when spilled
        self._counts: dict = {}  # graph -> occurrence count (always in RAM)
        self._graphs_order: list = []  # graph keys aligned with the spill file
        self._state = "building"  # building | loaded | spilled
        self._pinned = 0  # >0: currently used by a mining step, never spill
        self._weight = 0  # estimated live Mapping links (occurrences x depth)
        self._spilled: dict = {}  # graph -> (offset, compressed byte count)
        self._lock = threading.RLock()
        self._manager = get_spill_manager()  # None: spill disabled
        self._uid = self._manager.next_uid() if self._manager is not None else None
        self.pattern = None

    def bind(self, pattern):
        """Associate rows with their pattern without rebinding shared parent rows."""
        if self.pattern is None:
            self.pattern = pattern

    def _node_order(self):
        return list(self.pattern.nodes())

    def _edge_order(self):
        return list(self.pattern.edges(keys=True, data=True))

    def _row_width(self):
        return len(self._node_order()) + len(self._edge_order())

    def _loaded_slots(self):
        return sum(
            len(block) for block in self._mappings.values() if block is not None
        )

    def _path_for(self):
        return self._manager.file_for(self._uid)

    def _write_graph_block(self, graph, block):
        path = self._path_for()
        payload = zlib.compress(memoryview(block), level=1)
        offset = os.path.getsize(path) if os.path.exists(path) else 0
        try:
            with open(path, "ab") as fh:
                fh.write(payload)
        except OSError as exc:
            try:
                with open(path, "r+b") as fh:
                    fh.truncate(offset)
            except OSError:
                pass
            if exc.errno in (errno.EDQUOT, errno.ENOSPC):
                raise OSError(
                    exc.errno,
                    "CMiner spill storage is full. Set CMINER_SPILL_DIR to a "
                    "filesystem with more quota or increase "
                    "CMINER_SPILL_MAX_LINKS if additional RAM is available",
                    path,
                ) from exc
            raise
        self._spilled[graph] = (offset, len(payload))
        self._manager.stats["bytes_written"] += len(payload)

    def _edge_rank(self, graph, pattern_edge, target_edge):
        src_p, dst_p, _key_p, data = pattern_edge
        src_t, dst_t, key_t = target_edge
        if not self.pattern.is_directed() and src_t > dst_t:
            src_t, dst_t = dst_t, src_t
        keys = sorted(graph.edge_keys_by_type(src_t, dst_t, data.get("type")))
        return keys.index(key_t)

    def _mapping_row(self, graph, mapping):
        node_mapping = mapping._retrieve_node_mapping()
        edge_mapping = mapping._retrieve_edge_mapping()
        row = [encode_node(graph, node_mapping[node]) for node in self._node_order()]
        for pattern_edge in self._edge_order():
            src_p, dst_p, key_p, _data = pattern_edge
            target_edge = edge_mapping.get((src_p, dst_p, key_p))
            if target_edge is None and not self.pattern.is_directed():
                target_edge = edge_mapping.get((dst_p, src_p, key_p))
            if target_edge is None:
                raise ValueError(
                    f"Matcher returned no mapping for edge {(src_p, dst_p, key_p)!r}"
                )
            row.append(self._edge_rank(graph, pattern_edge, target_edge))
        return row

    def _mapping_from_row(self, graph, row):
        node_order = self._node_order()
        node_mapping = {
            pattern_node: decode_node(graph, code)
            for pattern_node, code in zip(node_order, row[: len(node_order)])
        }
        edge_mapping = {}
        for position, (src_p, dst_p, key_p, data) in enumerate(self._edge_order()):
            src_t = node_mapping[src_p]
            dst_t = node_mapping[dst_p]
            if not self.pattern.is_directed() and src_t > dst_t:
                src_t, dst_t = dst_t, src_t
            keys = sorted(graph.edge_keys_by_type(src_t, dst_t, data.get("type")))
            rank = row[len(node_order) + position]
            if rank < 0 or rank >= len(keys):
                raise ValueError(f"Stored edge rank {rank} is invalid")
            edge_mapping[(src_p, dst_p, key_p)] = (src_t, dst_t, keys[rank])
        return Mapping(node_mapping=node_mapping, edge_mapping=edge_mapping)

    def iter_rows(self, graph):
        values = self.rows(graph)
        width = self._row_width()
        for start in range(0, len(values), width):
            yield list(values[start : start + width])

    def rows(self, graph):
        with self._lock:
            self._load_graph(graph)
            return self._mappings[graph]

    def row(self, graph, index):
        width = self._row_width()
        values = self.rows(graph)
        start = index * width
        if start < 0 or start + width > len(values):
            raise IndexError(index)
        return list(values[start : start + width])

    # ---- spill/pin plumbing ----

    @property
    def patterns_mappings(self) -> dict:
        """
        Backward-compatible accessor (loads from disk if spilled).
        """
        with self._lock:
            for graph in self.graphs():
                self._load_graph(graph)
            return self._mappings

    def seal(self, depth: int = 1):
        """
        Mark the mapping set complete and immutable, and register it with
        the SpillManager (which may evict other, non-pinned sets).
        ``depth`` is the pattern size in nodes (chain length per occurrence),
        used to estimate the live-link weight of this set.
        """
        if self._state == "spilled":
            raise RuntimeError("cannot seal a spilled PatternMappings")
        self._state = "loaded"
        if self._manager is None:
            return
        self._manager.register(self, self._loaded_slots())

    def pin(self):
        """Prevent spilling. Must be balanced with unpin() (see mining loops)."""
        with self._lock:
            self._pinned += 1

    def unpin(self):
        with self._lock:
            self._pinned = max(0, self._pinned - 1)

    def _spill_now(self) -> bool:
        """
        Serialize the occurrence lists to the spill file and drop them from
        RAM. Called by the SpillManager (never while pinned). Since sealed
        sets are immutable, an already-written file stays valid and later
        spills only drop the in-RAM copy.
        """
        with self._lock:
            if self._state not in ("loaded", "spilled") or self._pinned > 0:
                return False
            spilled_any = False
            for graph in self.graphs():
                block = self._mappings.get(graph)
                if block is None:
                    continue
                if graph not in self._spilled:
                    self._write_graph_block(graph, block)
                self._mappings[graph] = None
                spilled_any = True
            if not spilled_any:
                return False
            self._state = "spilled"
            self._manager.note_spilled(self)
            return True

    def _load_graph(self, graph):
        """Load one graph block, leaving all unrelated graph blocks spilled."""
        if self._mappings.get(graph) is not None:
            return
        block_location = self._spilled.get(graph)
        if block_location is None:
            raise RuntimeError(f"No occurrence data available for graph {graph!r}")
        offset, compressed_size = block_location
        path = self._path_for()
        values = array(_ROW_TYPE)
        with open(path, "rb") as fh:
            fh.seek(offset)
            payload = fh.read(compressed_size)
        values.frombytes(zlib.decompress(payload))
        expected_slots = self._counts[graph] * self._row_width()
        if len(values) != expected_slots:
            raise RuntimeError(
                f"Corrupt occurrence block: expected {expected_slots} slots, "
                f"found {len(values)}"
            )
        self._mappings[graph] = values
        self._state = "loaded"
        self._manager.stats["loads"] += 1
        self._manager.stats["bytes_read"] += compressed_size
        self._manager.note_loaded(self)

    def release(self, graph):
        """Release one sealed graph block after graph-local processing."""
        if self._manager is None or self._state == "building":
            return
        with self._lock:
            block = self._mappings.get(graph)
            if block is None:
                return
            if graph not in self._spilled:
                self._write_graph_block(graph, block)
            self._mappings[graph] = None
            if not any(block is not None for block in self._mappings.values()):
                self._state = "spilled"
            self._manager.note_spilled(self)

    # ---- container API ----

    def __str__(self) -> str:
        """
        String representation of the pattern mappings.
        """
        output = ""
        # id of the projected nodes
        for g in self.graphs():
            output += f"{g.get_name()}\n"
            for m in self.mappings(g):
                output += ",".join(
                    str(v) for _, v in sorted(m._retrieve_node_mapping().items())
                )
                output += "\n"
        return output

    def graphs(self) -> list[DBGraph]:
        """
        Return the graphs that contains the pattern
        """
        return list(self._counts.keys())

    def mappings(self, graph) -> Sequence[Mapping]:
        """
        Return the mappings of the pattern in the graph.
        The dict access is under the set lock so a concurrent eviction
        (which sets _mappings=None) cannot race with it; the returned list
        stays alive in the caller's hands regardless of later evictions.
        """
        return _MappingSequence(self, graph)

    def count(self, graph) -> int:
        """
        Number of occurrences in the graph, without loading them from disk.
        """
        return self._counts.get(graph, 0)

    def set_mapping(self, graph, mappings: list[Mapping]):
        """
        Set the mappings of the pattern in the graph.
        """
        if self._state != "building":
            raise RuntimeError("PatternMappings is sealed (immutable)")
        block = array(_ROW_TYPE)
        for mapping in mappings:
            block.extend(self._mapping_row(graph, mapping))
        self._mappings[graph] = block
        self._counts[graph] = len(mappings)
        if graph not in self._graphs_order:
            self._graphs_order.append(graph)

    def set_rows(self, graph, rows):
        """Store already-encoded occurrence rows without creating Mapping objects."""
        if self._state != "building":
            raise RuntimeError("PatternMappings is sealed (immutable)")
        width = self._row_width()
        block = array(_ROW_TYPE)
        count = 0
        for row in rows:
            if len(row) != width:
                raise ValueError(f"Expected row width {width}, received {len(row)}")
            block.extend(encode_node(graph, node) for node in row)
            count += 1
        if count == 0:
            return
        self._mappings[graph] = block
        self._counts[graph] = count
        if graph not in self._graphs_order:
            self._graphs_order.append(graph)

    def add_mapping(self, graph, mapping: Mapping):
        """
        Add a mapping of the pattern in the graph.
        """
        if self._state != "building":
            raise RuntimeError("PatternMappings is sealed (immutable)")
        if graph not in self._mappings:
            self._mappings[graph] = array(_ROW_TYPE)
            self._counts[graph] = 0
            self._graphs_order.append(graph)
        self._mappings[graph].extend(self._mapping_row(graph, mapping))
        self._counts[graph] += 1

    # ---- helper methods ----

    @staticmethod
    def mapping_code(mapped_node_ids: list[int]) -> str:
        """
        Return the mapping code of the pattern.

        Parameters:
            mapping (Mapping): The mapping of the pattern.
        """
        return "".join(sorted([str(x) for x in mapped_node_ids]))

    @staticmethod
    def create_edge_mapping_dict(
        src_p, dst_p, src_t, dst_t, target, labels
    ) -> dict[tuple[int, int, int], tuple[int, int, int]]:
        """
        Create a dictionary that maps the edges of the pattern to the edges of the target graph.

        Args:
            src_p (_type_): src node of the pattern
            dst_p (_type_): dst node of the pattern
            src_t (_type_): src node of the target graph
            dst_t (_type_): dst node of the target graph
            target (_type_): target graph
            labels (_type_): labels of the edges

        Returns:
            dict[tuple[int, int, int], tuple[int, int, int]]: mapping of the edges
        """
        edge_mapping = {}
        new_key = 0
        prev_lab = None
        prev_keys = []
        for lab in sorted(labels):
            new_pattern_edge = (src_p, dst_p, new_key)
            if prev_lab != lab:
                try:
                    prev_keys = target.edge_keys_by_type(src_t, dst_t, lab)
                except KeyError:
                    prev_keys = []
                prev_lab = lab

            if len(prev_keys) == 0:
                raise ValueError(
                    "Not enough target edges to map pattern edge labels: "
                    f"({src_t}->{dst_t}, type='{lab}')"
                )

            target_edge = (src_t, dst_t, prev_keys.pop(0))
            edge_mapping[new_pattern_edge] = target_edge
            new_key += 1
        return edge_mapping

    @staticmethod
    def update_edge_mapping(src_p, dst_p, src_t, dst_t, mapping, target, labels):
        new_key = 0
        prev_lab = None
        prev_keys = []
        for lab in sorted(labels):
            if prev_lab != lab:
                prev_keys = target.edge_keys_by_type(src_t, dst_t, lab)
                prev_lab = lab
            if len(prev_keys) == 0:
                continue
            target_edge = (src_t, dst_t, prev_keys.pop(0))
            mapping.set_edge((dst_p, src_p, new_key), target_edge)
            new_key += 1


class Pattern:

    def __init__(
        self,
        pattern_mappings: PatternMappings,
        extended_pattern: "Pattern" = None,
        **attr,
    ):
        """
        Represents a pattern in the database.

        Parameters:
            pattern_mappings (PatternMappings): The mappings of the pattern.
            extended_pattern (Pattern): The extended pattern, if any.
            **attr: Additional attributes for the pattern.
        """
        self.pattern_mappings = pattern_mappings
        self.pattern_mappings.bind(self)
        # Do NOT retain the parent pattern, nothing reads this attribute
        self.extended_pattern = None

    # ---- basic methods ----

    def __str__(self) -> str:
        """
        String representation of the pattern.
        """
        graph_str = ""
        # graph_str = "code: " + self.canonical_code() + "\n"
        for node in self.nodes(data=True):
            graph_str += f"v {node[0]} {' '.join(node[1]['labels'])}\n"
        for edge in self.edges(data=True):
            graph_str += f"e {edge[0]} {edge[1]} {edge[2]['type']}\n"

        # for maps in self.pattern_mappings.patterns_mappings.values():
        #     for m in maps:
        #         graph_str += f"nodi {m._retrieve_node_mapping()}\n"
        #         graph_str += f"archi {m._retrieve_edge_mapping()}\n"

        return graph_str

    def graphs(self) -> list[DBGraph]:
        """
        Return the graphs that contains the pattern
        """
        return self.pattern_mappings.graphs()

    def frequency(self):
        """
        Return the frequency of the pattern.
        Counts are kept in RAM: this never triggers spill I/O.
        """
        pm = self.pattern_mappings
        return sum(pm.count(g) for g in pm.graphs())

    def support(self):
        """
        Return the support of the pattern
        """
        return len(self.graphs())

    def mappings_str(self, mapping_info: bool = False):
        """
        Return the mappings of the pattern.
        """
        output = ""
        pm = self.pattern_mappings
        for g in self.graphs():
            output += g.get_name() + " " + str(pm.count(g)) + "\n"
            if mapping_info:
                for _map in pm.mappings(g):
                    output += "    " + str(_map) + "\n"
        return output

    # ---- subclass methods ----

    def create_node_extension_manager(self, min_support) -> NodeExtensionManager:
        """
        Create a node extension manager to keep track of the candidate extensions.

        It is created in the pattern subclass to allow for different implementations
        for directed and undirected patterns.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def create_edge_extension_manager(self, min_support) -> EdgeExtensionManager:
        """
        Create an edge extension manager to keep track of the candidate extensions.

        It is created in the pattern subclass to allow for different implementations
        for directed and undirected patterns.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def create_pattern(
        self,
        pattern_mappings: PatternMappings,
        extended_pattern: "Pattern" = None,
        **attr,
    ) -> "Pattern":
        """
        Create a new pattern that work with directed or undirected graphs.

        It is created in the pattern subclass to allow for different implementations
        for directed and undirected patterns.

        Parameters:
            pattern_mappings (PatternMappings): The mappings of the new pattern.
            extended_pattern (Pattern): The extended pattern, if any.
            **attr: Additional attributes for the new pattern.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def add_edges_to_pattern(self, pattern: "Pattern", src, dst, extension: Extension):
        """
        Add edges to the pattern extended.

        It is created in the pattern subclass to allow for different implementations
        for directed and undirected patterns.

        Parameters:
            pattern (Pattern): The pattern to extend.
            src: The source node of the edge.
            dst: The destination node of the edge.
            extension (Extension): The extension to apply.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def create_edge_mapping_dict(
        self, src_p, dst_p, src_t, dst_t, target, extension_strategy: Extension
    ) -> dict[tuple[int, int, int], tuple[int, int, int]]:
        """
        Create a dictionary that maps the edges of the pattern to the edges of the target graph.

        It is created in the pattern subclass to allow for different implementations
        for directed and undirected patterns.

        Parameters:
            src_p (int): Source node of the pattern.
            dst_p (int): Destination node of the pattern.
            src_t (int): Source node of the target graph.
            dst_t (int): Destination node of the target graph.
            target (DBGraph): Target graph.
            labels (list[str]): Labels of the edges.

        Returns:
            dict[tuple[int, int, int], tuple[int, int, int]]: Mapping of the edges.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def _complete_mapping(self, target, node_mapping):
        """Create a deterministic edge-injective mapping for a node mapping."""
        edge_mapping = {}
        used_keys = defaultdict(set)
        for src_p, dst_p, key_p, data in self.edges(keys=True, data=True):
            src_t = node_mapping[src_p]
            dst_t = node_mapping[dst_p]
            if not self.is_directed() and src_t > dst_t:
                src_t, dst_t = dst_t, src_t
            label = data.get("type")
            signature = (src_t, dst_t, label)
            keys = sorted(target.edge_keys_by_type(src_t, dst_t, label))
            key_t = next(
                (key for key in keys if key not in used_keys[signature]), None
            )
            if key_t is None:
                raise ValueError(
                    f"Not enough target edges for ({src_t}, {dst_t}, {label!r})"
                )
            used_keys[signature].add(key_t)
            edge_mapping[(src_p, dst_p, key_p)] = (src_t, dst_t, key_t)
        return Mapping(node_mapping=node_mapping, edge_mapping=edge_mapping)

    # ---- node extension methods ----

    def find_node_extensions(self, min_support) -> list[NodeExtension]:
        """
        Find all possible node extensions for the pattern.
        """
        # Create a node extension manager to keep track of the candidate extensions
        extension_manager = self.create_node_extension_manager(min_support)
        node_order = list(self.nodes())
        node_count = len(node_order)
        # for all graph in the database that contains the current extension
        for g in self.graphs():
            # For each map we know one place where the extension is located in the graph.
            # We search all nodes that are neighbors of the current pattern and create a new extension.
            for occurrence_index, row in enumerate(
                self.pattern_mappings.iter_rows(g)
            ):
                decoded_nodes = [decode_node(g, code) for code in row[:node_count]]
                mapped_target_nodes = set(decoded_nodes)
                # node_p  := node pattern
                # node_db := node in the DB graph mapped to node_p
                for node_p, node_db in zip(node_order, decoded_nodes):
                    # for each node of the pattern search a possible extension
                    for neigh in g.all_neighbors(node_db).difference(
                        mapped_target_nodes
                    ):
                        extension_manager.add(
                            node_p, node_db, neigh, g, occurrence_index
                        )
            self.pattern_mappings.release(g)

        extensions = extension_manager.frequent_extensions()
        del extension_manager
        return extensions

    def apply_node_extension(self, node_extension: NodeExtension):
        """
        Apply the node extension to the pattern.

        Parameters:
            node_extension (NodeExtension): The node extension to apply.
        """

        # The id of the previous pattern node that is extended
        pattern_node_id = node_extension.pattern_node_id

        # Apply extension to the pattern (add node and edges)
        new_pattern = self.create_pattern(
            extended_pattern=self, pattern_mappings=self.pattern_mappings
        )
        integer_nodes = [node for node in new_pattern.nodes() if isinstance(node, int)]
        new_pattern_new_node_id = max(integer_nodes, default=-1) + 1
        while new_pattern_new_node_id in new_pattern:
            new_pattern_new_node_id += 1
        new_pattern.add_node(new_pattern_new_node_id, labels=node_extension.node_labels)

        self.add_edges_to_pattern(
            new_pattern,
            new_pattern_new_node_id,
            pattern_node_id,
            node_extension.get_strategy(),
        )

        return new_pattern

    def update_node_mappings(self, node_extension: NodeExtension):
        """
        Update the node mappings based on the applied node extension.

        Parameters:
            node_extension (NodeExtension): The node extension that was applied.
        """
        new_pattern_new_node_id = len(self.nodes()) - 1
        pattern_node_id = node_extension.pattern_node_id
        # Object to keep track of the new pattern mappings
        new_pattern_mappings = PatternMappings()
        new_pattern_mappings.bind(self)
        # Update the pattern mappings
        for target in node_extension.graphs():
            for occurrence_index, target_map in enumerate(
                self.pattern_mappings.mappings(target)
            ):  # old pattern mapping of the extended graph
                # set to store the code of the mappings to avoid unnecessary duplicates mirrored mappings
                mappings_codes = set()
                target_node_ids = node_extension.target_node_ids(
                    target, occurrence_index
                )
                # when trying to extend the pattern Pn (pattern with n nodes), there can be some mappings of Pn
                # that are not extended because the extension is not applicable.
                if len(target_node_ids) == 0:
                    continue

                # mapped node ids of the pattern (without the new node)
                base_node_ids = list(target_map.nodes_mapping().values())

                for target_node_code in target_node_ids:
                    target_node_id = decode_node(target, target_node_code)

                    # ---- START CHECK THE MAPPING IS REDUNDANT ----

                    # complete the array with all mapped node ids (including
                    # the new node). build a fresh list per candidate
                    # appending to a shared list leaks the previous candidates'
                    # ids into the code, which breaks duplicate detection.
                    mapped_node_ids = base_node_ids + [target_node_id]

                    mapping_code = PatternMappings.mapping_code(mapped_node_ids)
                    # check if the mapping code is already in the set
                    if mapping_code in mappings_codes:
                        continue
                    # add the mapping code to the set
                    mappings_codes.add(mapping_code)
                    # ---- END CHECK THE MAPPING IS REDUNDANT ----

                    # ---- START CREATE THE NEW MAPPING ----
                    # there is no need to reconstruct again the mapping of the nodes
                    # in the old pattern mapping because it is already done in the previous step

                    # we just create the new mapping for the new node and for the edges

                    try:
                        node_mapping = target_map._retrieve_node_mapping()
                        node_mapping[new_pattern_new_node_id] = target_node_id
                        new_mapping = self._complete_mapping(
                            target, node_mapping
                        )
                    except (ValueError, KeyError, IndexError):
                        # The extension may be frequent overall but not applicable for this
                        # specific mapping/target-node occurrence.
                        continue

                    # ---- END CREATE THE NEW MAPPING ----

                    new_pattern_mappings.add_mapping(target, new_mapping)
            self.pattern_mappings.release(target)

        self.pattern_mappings = new_pattern_mappings
        new_pattern_mappings.bind(self)
        # The occurrence set of this pattern is complete: seal it (immutable
        # from now on, evictable by the SpillManager when not pinned).
        new_pattern_mappings.seal(depth=len(self.nodes()))

    # ---- edge extension methods ----

    def find_edge_extensions(self, min_support) -> list[list[EdgeExtension]]:
        """
        Find all possible edge extensions for the pattern.
        """
        if len(self.nodes()) < 3:
            # if the pattern has less than 3 nodes,
            # it is not possible to find edge extensions
            return []

        extension_manager = self.create_edge_extension_manager(min_support)

        pattern_edge_counts = Counter()
        for src, dst, _key, data in self.edges(keys=True, data=True):
            if not self.is_directed() and src > dst:
                src, dst = dst, src
            pattern_edge_counts[(src, dst, data.get("type"))] += 1

        node_order = list(self.nodes())
        node_count = len(node_order)

        for g in self.graphs():
            for occurrence_index, row in enumerate(
                self.pattern_mappings.iter_rows(g)
            ):
                node_values = [decode_node(g, code) for code in row[:node_count]]
                target_to_pattern = dict(zip(node_values, node_order))
                mapped_pattern_complete_graph_edges = g.all_edges_of_subgraph(
                    node_values
                )
                target_edge_counts = Counter()
                for src, dst, key in mapped_pattern_complete_graph_edges:
                    if src not in target_to_pattern or dst not in target_to_pattern:
                        continue
                    pattern_node_src = target_to_pattern[src]
                    pattern_node_dest = target_to_pattern[dst]
                    if not self.is_directed() and pattern_node_src > pattern_node_dest:
                        pattern_node_src, pattern_node_dest = (
                            pattern_node_dest,
                            pattern_node_src,
                        )
                    label = g.get_edge_label((src, dst, key))
                    target_edge_counts[
                        (pattern_node_src, pattern_node_dest, label)
                    ] += 1

                groups = defaultdict(list)
                for (src, dst, label), target_count in target_edge_counts.items():
                    remaining = target_count - pattern_edge_counts.get(
                        (src, dst, label), 0
                    )
                    if remaining > 0:
                        groups[(src, dst)].extend([label] * remaining)

                for (src, dst), labels in groups.items():
                    extension_manager.add(src, dst, labels, g, occurrence_index)
            self.pattern_mappings.release(g)

        extensions = extension_manager.frequent_extensions()

        if len(extensions) == 0:
            return []

        extension_graph_sets = [frozenset(ext.graphs()) for ext in extensions]

        groups = []
        for graph_set in dict.fromkeys(extension_graph_sets):
            group = []
            for ext, candidate_graphs in zip(extensions, extension_graph_sets):
                skip = False
                if graph_set.issubset(candidate_graphs):
                    for e in group:
                        if (
                            ext.pattern_node_src == e.pattern_node_src
                            and ext.pattern_node_dst == e.pattern_node_dst
                        ):
                            skip = True
                            break
                    if skip:
                        continue
                    ext_copy = ext.__copy__()
                    new_location = {
                        v: k
                        for v, k in ext.extension_strategy.location.items()
                        if v in graph_set
                    }
                    ext_copy.location = new_location
                    group.append(ext_copy)
            groups.append(group)

        return groups

    def apply_edge_extension(self, edge_extensions: list[EdgeExtension]) -> "Pattern":
        """
        Apply the edge extension to the pattern.

        Parameters:
            edge_extensions (list[EdgeExtension]): The edge extensions to apply.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def update_edge_mapping_template(self, mapping):
        """
        Update the edge mapping of the pattern.
        Implemented in the pattern subclass.

        Parameters:
            mapping (Mapping): The mapping to update.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def update_edge_mappings(self, edge_extensions: list[EdgeExtension]):
        """
        Update the edge mappings based on the applied edge extension.

        Parameters:
            edge_extension (EdgeExtension): The edge extension that was applied.
        """
        db_graphs = edge_extensions[0].graphs()
        new_pattern_mappings = PatternMappings()
        new_pattern_mappings.bind(self)
        # Update the pattern mappings
        for target in db_graphs:
            for occurrence_index, target_map in enumerate(
                self.pattern_mappings.mappings(target)
            ):

                try:
                    if any(
                        occurrence_index not in ext.extension_strategy.mapping(target)
                        for ext in edge_extensions
                    ):
                        continue
                except KeyError:
                    # target could not be associated to any mapping in the extension
                    continue

                try:
                    new_mapping = self._complete_mapping(
                        target, target_map._retrieve_node_mapping()
                    )
                except (ValueError, KeyError, IndexError):
                    continue
                new_pattern_mappings.add_mapping(target, new_mapping)
            self.pattern_mappings.release(target)

        self.pattern_mappings = new_pattern_mappings
        new_pattern_mappings.bind(self)
        # The occurrence set of this pattern is complete: seal it (immutable
        # from now on, evictable by the SpillManager when not pinned).
        new_pattern_mappings.seal(depth=len(self.nodes()))


class DirectedPattern(Pattern, DirectedMultiGraph):

    def __init__(self, pattern_mappings, extended_pattern: "Pattern" = None, **attr):
        """
        Represents a pattern in the database.
        """
        if extended_pattern is not None:
            DirectedMultiGraph.__init__(self, extended_pattern, **attr)
        else:
            DirectedMultiGraph.__init__(self, **attr)
        Pattern.__init__(self, pattern_mappings, extended_pattern, **attr)

    def create_pattern(
        self,
        pattern_mappings: PatternMappings,
        extended_pattern: "Pattern" = None,
        **attr,
    ) -> "DirectedPattern":
        """
        Create a new directed pattern.

        Parameters:
            pattern_mappings (PatternMappings): The mappings of the new pattern.
            extended_pattern (Pattern): The extended pattern, if any.
            **attr: Additional attributes for the new pattern.
        """
        return DirectedPattern(pattern_mappings, extended_pattern, **attr)

    def create_node_extension_manager(self, min_support):
        return DirectedNodeExtensionManager(min_support)

    def create_edge_extension_manager(self, min_support):
        return DirectedEdgeExtensionManager(min_support)

    def add_edges_to_pattern(
        self, pattern: "Pattern", src, dst, extension: DirectedExtension
    ):
        """
        Add edges to the pattern extended.

        Parameters:
            pattern (Pattern): The pattern to extend.
            extension (DirectedExtension): The directed extension to apply.
        """
        # in_edge_labels: labels that enter the node from which the extension start
        for lab in extension.in_edge_labels:
            pattern.add_edge(src, dst, type=lab)
        # out_edge_labels: labels that exit the node from which the extension start
        for lab in extension.out_edge_labels:
            pattern.add_edge(dst, src, type=lab)

    def create_edge_mapping_dict(
        self, src_p, dst_p, src_t, dst_t, target, extension_strategy: DirectedExtension
    ) -> dict[tuple[int, int, int], tuple[int, int, int]]:
        """
        Create a dictionary that maps the edges of the pattern to the edges of the target graph.

        Parameters:
            src_p (int): Source node of the pattern.
            dst_p (int): Destination node of the pattern.
            src_t (int): Source node of the target graph.
            dst_t (int): Destination node of the target graph.
            target (DBGraph): Target graph.
            extension_strategy (DirectedExtension): The directed extension strategy.

        Returns:
            dict[tuple[int, int, int], tuple[int, int, int]]: Mapping of the edges.
        """
        # edge mapping
        edge_mapping_in = PatternMappings.create_edge_mapping_dict(
            src_p, dst_p, src_t, dst_t, target, extension_strategy.out_edge_labels
        )
        edge_mapping_out = PatternMappings.create_edge_mapping_dict(
            dst_p, src_p, dst_t, src_t, target, extension_strategy.in_edge_labels
        )
        # merge the two edge mappings
        return {**edge_mapping_in, **edge_mapping_out}

    def update_edge_mapping_template(
        self,
        src_p,
        dst_p,
        src_t,
        dst_t,
        mapping,
        target,
        extension_strategy: DirectedExtension,
    ):
        """
        Update the edge mapping of the pattern.

        Parameters:
            src_p (int): Source node of the pattern.
            dst_p (int): Destination node of the pattern.
            src_t (int): Source node of the target graph.
            dst_t (int): Destination node of the target graph.
            mapping (Mapping): The mapping to update.
            target (DBGraph): Target graph.
            extension_strategy (DirectedExtension): The directed extension strategy.
        """
        # update the edge mapping
        PatternMappings.update_edge_mapping(
            src_p,
            dst_p,
            src_t,
            dst_t,
            mapping,
            target,
            extension_strategy.out_edge_labels,
        )
        PatternMappings.update_edge_mapping(
            dst_p,
            src_p,
            dst_t,
            src_t,
            mapping,
            target,
            extension_strategy.in_edge_labels,
        )

    def apply_edge_extension(
        self, edge_extensions: list[EdgeExtension]
    ) -> "DirectedPattern":
        """
        Apply the edge extension to the pattern.

        Parameters:
            edge_extensions (EdgeExtension): The edge extension to apply.
        """
        # Apply extension to the pattern (add edges)
        new_pattern = DirectedPattern(
            extended_pattern=self, pattern_mappings=self.pattern_mappings
        )

        for ext in edge_extensions:
            for lab in ext.extension_strategy.in_edge_labels:
                new_pattern.add_edge(
                    ext.pattern_node_dst, ext.pattern_node_src, type=lab
                )
            for lab in ext.extension_strategy.out_edge_labels:
                new_pattern.add_edge(
                    ext.pattern_node_src, ext.pattern_node_dst, type=lab
                )

        return new_pattern


class UndirectedPattern(Pattern, UndirectedMultiGraph):

    def __init__(self, pattern_mappings, extended_pattern: "Pattern" = None, **attr):
        """
        Represents a pattern in the database.
        """
        if extended_pattern is not None:
            UndirectedMultiGraph.__init__(self, extended_pattern, **attr)
        else:
            UndirectedMultiGraph.__init__(self, **attr)
        Pattern.__init__(self, pattern_mappings, extended_pattern, **attr)

    def create_pattern(
        self,
        pattern_mappings: PatternMappings,
        extended_pattern: "Pattern" = None,
        **attr,
    ) -> "DirectedPattern":
        """
        Create a new directed pattern.

        Parameters:
            pattern_mappings (PatternMappings): The mappings of the new pattern.
            extended_pattern (Pattern): The extended pattern, if any.
            **attr: Additional attributes for the new pattern.
        """
        return UndirectedPattern(pattern_mappings, extended_pattern, **attr)

    def create_node_extension_manager(self, min_support):
        return UndirectedNodeExtensionManager(min_support)

    def create_edge_extension_manager(self, min_support):
        return UndirectedEdgeExtensionManager(min_support)

    def add_edges_to_pattern(self, pattern: "Pattern", src, dst, extension: Extension):
        """
        Add edges to the pattern extended.

        Parameters:
            pattern (Pattern): The pattern to extend.
            src: The source node of the edge.
            dst: The destination node of the edge.
            extension (Extension): The extension to apply.
        """
        for lab in extension.edge_labels:
            pattern.add_edge(src, dst, type=lab)

    def create_edge_mapping_dict(
        self,
        src_p,
        dst_p,
        src_t,
        dst_t,
        target,
        extension_strategy: UndirectedExtension,
    ) -> dict[tuple[int, int, int], tuple[int, int, int]]:
        """
        Create a dictionary that maps the edges of the pattern to the edges of the target graph.

        Parameters:
            src_p (int): Source node of the pattern.
            dst_p (int): Destination node of the pattern.
            src_t (int): Source node of the target graph.
            dst_t (int): Destination node of the target graph.
            target (DBGraph): Target graph.
            labels (list[str]): Labels of the edges.

        Returns:
            dict[tuple[int, int, int], tuple[int, int, int]]: Mapping of the edges.
        """
        # convention for the algorithm, when dealing with undirected graphs: src < dst
        if src_t > dst_t:
            src_t, dst_t = dst_t, src_t
            src_p, dst_p = dst_p, src_p
        # edge mapping
        return PatternMappings.create_edge_mapping_dict(
            src_p, dst_p, src_t, dst_t, target, extension_strategy.edge_labels
        )

    def update_edge_mapping_template(
        self,
        src_p,
        dst_p,
        src_t,
        dst_t,
        mapping,
        target,
        extension_strategy: DirectedExtension,
    ):
        """
        Update the edge mapping of the pattern.

        Parameters:
            src_p (int): Source node of the pattern.
            dst_p (int): Destination node of the pattern.
            src_t (int): Source node of the target graph.
            dst_t (int): Destination node of the target graph.
            mapping (Mapping): The mapping to update.
            target (DBGraph): Target graph.
            extension_strategy (DirectedExtension): The directed extension strategy.
        """
        if src_t > dst_t:
            src_t, dst_t = dst_t, src_t
            src_p, dst_p = dst_p, src_p
        # update the edge mapping
        PatternMappings.update_edge_mapping(
            src_p, dst_p, src_t, dst_t, mapping, target, extension_strategy.edge_labels
        )

    def apply_edge_extension(
        self, edge_extensions: list[EdgeExtension]
    ) -> "UndirectedPattern":
        """
        Apply the edge extension to the pattern.

        Parameters:
            edge_extension (EdgeExtension): The edge extension to apply.
        """
        # Apply extension to the pattern (add edges)
        new_pattern = UndirectedPattern(
            extended_pattern=self, pattern_mappings=self.pattern_mappings
        )

        for ext in edge_extensions:
            for lab in ext.extension_strategy.edge_labels:
                new_pattern.add_edge(
                    ext.pattern_node_src, ext.pattern_node_dst, type=lab
                )
        return new_pattern
