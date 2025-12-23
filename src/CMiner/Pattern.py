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

from .EdgeExtension import (
    DirectedEdgeExtensionManager,
    EdgeExtensionManager,
    UndirectedEdgeExtensionManager,
)
from .Extension import DirectedExtension, Extension, UndirectedExtension


class PatternMappings:

    def __init__(self):
        """
        Keep track of each mapping for each graph of a specific pattern.
        """
        self.patterns_mappings = {}

    def __str__(self) -> str:
        """
        String representation of the pattern mappings.
        """
        output = ""
        # id of the projected nodes
        for g, maps in self.patterns_mappings.items():
            output += f"{g.get_name()}\n"
            for m in maps:
                output += ",".join(
                    str(v) for _, v in sorted(m._retrieve_node_mapping().items())
                )
                output += "\n"
        return output

    def graphs(self) -> list[DBGraph]:
        """
        Return the graphs that contains the pattern
        """
        return list(self.patterns_mappings.keys())

    def mappings(self, graph) -> list[Mapping]:
        """
        Return the mappings of the pattern in the graph.
        """
        return self.patterns_mappings[graph]

    def set_mapping(self, graph, mappings: list[Mapping]):
        """
        Set the mappings of the pattern in the graph.
        """
        self.patterns_mappings[graph] = mappings

    def add_mapping(self, graph, mapping: Mapping):
        """
        Add a mapping of the pattern in the graph.
        """
        if graph not in self.patterns_mappings:
            self.patterns_mappings[graph] = []
        self.patterns_mappings[graph].append(mapping)

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
                prev_keys = target.edge_keys_by_type(src_t, dst_t, lab)
                prev_lab = lab
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
        self.extended_pattern = extended_pattern

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
        """
        return sum(len(self.pattern_mappings.mappings(g)) for g in self.graphs())

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
        for g in self.graphs():
            output += (
                g.get_name() + " " + str(len(self.pattern_mappings.mappings(g))) + "\n"
            )
            if mapping_info:
                for _map in self.pattern_mappings.mappings(g):
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

    # ---- node extension methods ----

    def find_node_extensions(self, min_support) -> list[NodeExtension]:
        """
        Find all possible node extensions for the pattern.
        """
        # Create a node extension manager to keep track of the candidate extensions
        extension_manager = self.create_node_extension_manager(min_support)
        # for all graph in the database that contains the current extension
        for g in self.graphs():
            # obtain where the current extension is located in the graph
            mappings = self.pattern_mappings.mappings(g)
            # For each map we know one place where the extension is located in the graph.
            # We search all nodes that are neighbors of the current pattern and create a new extension.
            for _map in mappings:
                # retrieve nodes mapped in the DB graph
                mapped_target_nodes = _map.nodes()
                # node_p  := node pattern
                # node_db := node in the DB graph mapped to node_p
                for node_p, node_db in _map.node_pairs():
                    # for each node of the pattern search a possible extension
                    for neigh in g.all_neighbors(node_db).difference(
                        mapped_target_nodes
                    ):
                        extension_manager.add(node_p, node_db, neigh, g, _map)

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
        new_pattern_new_node_id = len(new_pattern.nodes())
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
        # Update the pattern mappings
        for target in node_extension.graphs():
            new_mappings = []
            for target_map in self.pattern_mappings.mappings(
                target
            ):  # old pattern mapping of the extended graph
                # set to store the code of the mappings to avoid unnecessary duplicates mirrored mappings
                mappings_codes = set()
                target_node_ids = node_extension.target_node_ids(target, target_map)
                # when trying to extend the pattern Pn (pattern with n nodes), there can be some mappings of Pn
                # that are not extended because the extension is not applicable.
                if len(target_node_ids) == 0:
                    continue

                # mapped node ids of the pattern (without the new node)
                mapped_node_ids = list(target_map.nodes_mapping().values())  #####

                for target_node_id in target_node_ids:

                    # ---- START CHECK THE MAPPING IS REDUNDANT ----

                    # complete the array with all mapped node ids (including the new node)
                    mapped_node_ids.append(target_node_id)

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

                    # node mapping
                    node_mapping = {new_pattern_new_node_id: target_node_id}

                    # edge mapping
                    edge_mapping = self.create_edge_mapping_dict(
                        pattern_node_id,
                        new_pattern_new_node_id,
                        target_map.nodes_mapping()[pattern_node_id],
                        target_node_id,
                        target,
                        node_extension.get_strategy(),
                    )

                    # ---- END CREATE THE NEW MAPPING ----

                    # set the new mapping
                    new_mapping = Mapping(
                        node_mapping=node_mapping,
                        edge_mapping=edge_mapping,
                        extended_mapping=target_map,
                    )
                    new_mappings.append(new_mapping)
            if len(new_mappings) > 0:
                new_pattern_mappings.set_mapping(target, new_mappings)

        self.pattern_mappings = new_pattern_mappings

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

        for g in self.graphs():
            for _map in self.pattern_mappings.mappings(g):
                mapped_pattern_complete_graph_edges = g.all_edges_of_subgraph(
                    _map.nodes()
                )
                mapped_pattern_edges = set(_map.get_target_edges())
                candidate_edges = set()

                for src, dst, key in mapped_pattern_complete_graph_edges:
                    skip = False
                    for s, d, k in mapped_pattern_edges:
                        if src == s and dst == d:
                            # remove i-th element from the list
                            mapped_pattern_edges.remove((s, d, k))
                            skip = True
                            break
                        # if src == d and dst == s:
                        #     # remove i-th element from the list
                        #     mapped_pattern_edges.remove((d, s, k))
                        #     skip = True
                        #     break
                    if skip:
                        continue
                    candidate_edges.add(
                        (src, dst, key, g.get_edge_label((src, dst, key)))
                    )

                groups = {}

                inverse_map = _map.inverse()
                for src, dst, key, lab in candidate_edges:
                    pattern_node_src = inverse_map.node(src)
                    pattern_node_dest = inverse_map.node(dst)
                    code = (pattern_node_src, pattern_node_dest)
                    if code not in groups:
                        groups[code] = []
                    groups[code].append(lab)

                for (src, dst), labels in groups.items():
                    extension_manager.add(src, dst, labels, g, _map)

        extensions = extension_manager.frequent_extensions()

        if len(extensions) == 0:
            return []

        graphs = sorted(self.graphs(), key=lambda x: x.get_name())
        extension_matrix = [
            [0 for _ in range(len(graphs))] for _ in range(len(extensions))
        ]
        for i, ext in enumerate(extensions):
            for j, g in enumerate(graphs):
                if g in ext.graphs():
                    extension_matrix[i][j] = 1

        # group row by row
        matrix_indices_grouped = {}
        for i, row in enumerate(extension_matrix):
            row_code = "".join(map(str, row))
            if row_code not in matrix_indices_grouped:
                matrix_indices_grouped[row_code] = []
            matrix_indices_grouped[row_code].append(i)

        groups = []
        for row_code, indices in matrix_indices_grouped.items():
            columns_to_select = [i for i, v in enumerate(row_code) if v == "1"]
            group = []
            for i, ext in enumerate(extensions):
                skip = False
                if all(extension_matrix[i][j] == 1 for j in columns_to_select):
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
                        if any(v == graphs[j] for j in columns_to_select)
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
        # Update the pattern mappings
        for target in db_graphs:
            new_mappings = []

            for target_map in self.pattern_mappings.mappings(target):

                try:
                    if any(
                        target_map not in ext.extension_strategy.mapping(target)
                        for ext in edge_extensions
                    ):
                        continue
                except:
                    # target could not be associated to any mapping in the extension
                    continue

                new_mapping = Mapping(extended_mapping=target_map)

                for extension in edge_extensions:

                    target_edge_src = target_map.nodes_mapping()[
                        extension.pattern_node_src
                    ]
                    target_edge_dest = target_map.nodes_mapping()[
                        extension.pattern_node_dst
                    ]

                    self.update_edge_mapping_template(
                        extension.pattern_node_src,
                        extension.pattern_node_dst,
                        target_edge_src,
                        target_edge_dest,
                        new_mapping,
                        target,
                        extension.extension_strategy,
                    )

                new_mappings.append(new_mapping)
            new_pattern_mappings.set_mapping(target, new_mappings)

        self.pattern_mappings = new_pattern_mappings


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
