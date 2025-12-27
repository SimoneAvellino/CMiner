from .Extension import DirectedExtension, Extension, UndirectedExtension
from .EdgeExtension import EdgeGroupsFinder


class NodeExtension:

    def __init__(self, pattern_node_id, node_labels, extension_strategy: Extension):
        """
        Initialize the NodeExtension with a specific extension strategy.

        Parameters:
            extension_strategy (Extension): The strategy to use for extending the node.
        """
        self.pattern_node_id = pattern_node_id
        self.node_labels = node_labels
        self.extension_strategy = extension_strategy

    def __str__(self):
        """
        Return a string representation of the NodeExtension.
        """
        return f"NodeExtension(pattern_node_id={self.pattern_node_id}, node_labels={self.node_labels}, extension_strategy={self.extension_strategy})"

    def get_strategy(self):
        """
        Return the extension strategy.
        """
        return self.extension_strategy

    def graphs(self):
        """
        Return the graphs where the extension is found.
        """
        return self.extension_strategy.graphs()

    def target_node_ids(self, graph, _map):
        """
        Return the target node ids from which the extension is found.

        Parameters:
            graph (DBGraph): The graph where the extension is found.
            _map (Mapping): The mapping of the pattern in the db_graph.
        """
        return self.extension_strategy.target_node_ids(graph, _map)


class NodeExtensionManager:

    def __init__(self, min_support: int):
        """
        Manages the extensions of nodes in a directed graph.
        It keeps track of the extensions and their locations in the database graphs.
        """
        self.min_support = min_support
        self.extensions = {}
        self.memoization = {}

    @staticmethod
    def _memo_key(db_graph, src_node_id: int, dst_node_id: int):
        # Include the graph identity to avoid collisions between different DB graphs
        # that may reuse the same node ids.
        return (id(db_graph), src_node_id, dst_node_id)

    def add(
        self, pattern_node_id, target_node_id, neigh_target_node_id, db_graph, _map
    ):
        """
        Add an extension to the manager

        Parameters:
            pattern_node_id (int): the id of the node in the pattern that is extended
            target_node_id (int): the id of the node mapped with pattern_node_id
            neigh_target_node_id (int): neighbor of target_node_id
            db_graph (DBGraph): the graph where the extension is found
            _map (Mapping): the mapping of the pattern in the db_graph
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def frequent_extensions(self) -> list[NodeExtension]:
        """
        Return a list of NodeExtensions that if applied to the pattern, it still remains frequent.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")


class DirectedNodeExtensionManager(NodeExtensionManager):

    def __init__(self, min_support: int):
        """
        Initialize the DirectedNodeExtensionManager with a minimum support.

        Parameters:
            min_support (int): The minimum support for the extensions.
        """
        super().__init__(min_support)

    def add(
        self, pattern_node_id, target_node_id, neigh_target_node_id, db_graph, _map
    ):
        """
        Add an extension to the manager

        Parameters:
            pattern_node_id (int): the id of the node in the pattern that is extended
            target_node_id (int): the id of the node mapped with pattern_node_id
            neigh_target_node_id (int): neighbor of target_node_id
            db_graph (DBGraph): the graph where the extension is found
            _map (Mapping): the mapping of the pattern in the db_graph
        """
        edge_labels = []

        forward_key = self._memo_key(db_graph, target_node_id, neigh_target_node_id)
        reverse_key = self._memo_key(db_graph, neigh_target_node_id, target_node_id)

        if forward_key not in self.memoization:
            self.memoization[forward_key] = db_graph.get_edge_labels_with_duplicate(
                target_node_id, neigh_target_node_id
            )
        if reverse_key not in self.memoization:
            self.memoization[reverse_key] = db_graph.get_edge_labels_with_duplicate(
                neigh_target_node_id, target_node_id
            )

        for label in self.memoization[forward_key]:
            edge_labels.append(
                DirectedNodeExtensionManager.orientation_code(label, True)
            )
        for label in self.memoization[reverse_key]:
            edge_labels.append(
                DirectedNodeExtensionManager.orientation_code(label, False)
            )

        neigh_target_node_labels = db_graph.get_node_labels(neigh_target_node_id)
        edge_labels = sorted(edge_labels)

        neigh_target_node_labels_code = " ".join(neigh_target_node_labels)
        target_edge_labels_code = " ".join(edge_labels)

        ext_code = (
            pattern_node_id,
            neigh_target_node_labels_code,
            target_edge_labels_code,
        )

        if ext_code not in self.extensions:
            self.extensions[ext_code] = {}
        if db_graph not in self.extensions[ext_code]:
            self.extensions[ext_code][db_graph] = []
        self.extensions[ext_code][db_graph].append((_map, neigh_target_node_id))

    def frequent_extensions(self) -> list["NodeExtension"]:
        """
        Return a list of NodeExtensions that if applied to the pattern, it still remains frequent.
        """

        frequent_extensions = []

        edge_group_finders = {}

        for (
            pattern_node_id,
            node_labels_code,
            target_edge_labels_code,
        ), db_graphs in self.extensions.items():

            # use the finder code to identify the finder
            finder_code = (pattern_node_id, node_labels_code)
            # instantiate the finder if it is not present
            if finder_code not in edge_group_finders:
                edge_group_finders[finder_code] = EdgeGroupsFinder(self.min_support)

            # create the location dictionary
            location = {}
            for g in db_graphs:
                aa = {}
                for mapping, node_id in db_graphs[g]:
                    if mapping not in aa:
                        aa[mapping] = []
                    aa[mapping].append(node_id)
                location[g] = aa

            # select the correct finder and add the edge extension
            edge_group_finder = edge_group_finders[finder_code]
            edge_group_finder.add(target_edge_labels_code.split(" "), location)

        # save all frequent extensions
        for (
            pattern_node_id,
            node_labels_code,
        ), edge_group_finder in edge_group_finders.items():
            ext = edge_group_finder.find()
            for e in ext:
                frequent_extensions.append(
                    NodeExtension(
                        pattern_node_id,
                        node_labels_code.split(" "),
                        DirectedExtension(
                            e.out_edge_labels, e.in_edge_labels, e.location
                        ),
                    )
                )
        return frequent_extensions

    @staticmethod
    def orientation_code(label, outgoing):
        """
        Return the code of the orientation of the edge.

        E.g.
            src_node -- edge_label --> dst_node

            edge_label became 'out_edge_label' if outgoing is True
            edge_label became 'in_edge_label' if outgoing is False
        """
        return ("out_" if outgoing else "in_") + label


class UndirectedNodeExtensionManager(NodeExtensionManager):

    def __init__(self, min_support: int):
        """
        Initialize the UndirectedNodeExtensionManager with a minimum support.

        Parameters:
            min_support (int): The minimum support for the extensions.
        """
        super().__init__(min_support)

    def add(
        self, pattern_node_id, target_node_id, neigh_target_node_id, db_graph, _map
    ):
        edge_labels = []

        forward_key = self._memo_key(db_graph, target_node_id, neigh_target_node_id)
        if forward_key not in self.memoization:
            self.memoization[forward_key] = db_graph.get_edge_labels_with_duplicate(
                target_node_id, neigh_target_node_id
            )

        for label in self.memoization[forward_key]:
            edge_labels.append(
                DirectedNodeExtensionManager.orientation_code(label, True)
            )

        neigh_target_node_labels = db_graph.get_node_labels(neigh_target_node_id)
        edge_labels = sorted(edge_labels)

        neigh_target_node_labels_code = " ".join(neigh_target_node_labels)
        target_edge_labels_code = " ".join(edge_labels)

        ext_code = (
            pattern_node_id,
            neigh_target_node_labels_code,
            target_edge_labels_code,
        )

        if ext_code not in self.extensions:
            self.extensions[ext_code] = {}
        if db_graph not in self.extensions[ext_code]:
            self.extensions[ext_code][db_graph] = []
        self.extensions[ext_code][db_graph].append((_map, neigh_target_node_id))

    def frequent_extensions(self) -> list["NodeExtension"]:
        """
        Return a list of NodeExtensions that if applied to the pattern, it still remains frequent.
        """

        frequent_extensions = []

        edge_group_finders = {}

        for (
            pattern_node_id,
            node_labels_code,
            target_edge_labels_code,
        ), db_graphs in self.extensions.items():

            # use the finder code to identify the finder
            finder_code = (pattern_node_id, node_labels_code)
            # instantiate the finder if it is not present
            if finder_code not in edge_group_finders:
                edge_group_finders[finder_code] = EdgeGroupsFinder(self.min_support)

            # create the location dictionary
            location = {}
            for g in db_graphs:
                aa = {}
                for mapping, node_id in db_graphs[g]:
                    if mapping not in aa:
                        aa[mapping] = []
                    aa[mapping].append(node_id)
                location[g] = aa

            # select the correct finder and add the edge extension
            edge_group_finder = edge_group_finders[finder_code]
            edge_group_finder.add(target_edge_labels_code.split(" "), location)

        # save all frequent extensions
        for (
            pattern_node_id,
            node_labels_code,
        ), edge_group_finder in edge_group_finders.items():
            ext = edge_group_finder.find()
            for e in ext:
                frequent_extensions.append(
                    NodeExtension(
                        pattern_node_id,
                        node_labels_code.split(" "),
                        UndirectedExtension(e.out_edge_labels, e.location),
                    )  # NOTE: i use only out_edge_labels because the in_edge_labels are not used in undirected graphs
                )
        return frequent_extensions
