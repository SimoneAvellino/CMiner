import networkx as nx
from .constants import DUMMY_LABEL, NULL_LABEL
from igraph import Graph as igraphGraph
from pynauty import Graph as nautyGraph, autgrp


def flat_map(list_of_lists):
    array = []
    for l in list_of_lists:
        array.extend(l)
    return array


class AbstractGraph(nx.Graph):
    """
    Abstract class for a graph
    """

    def __init__(self, incoming_graph_data=None, **attr):
        super().__init__(**attr)
        self.node_labels = None
        self.edge_labels = None
        if incoming_graph_data:
            self._initialize_from_graph_data(incoming_graph_data)

    def compute_orbits_nodes_helper(self, directed: bool):

        # COMPUTING THE ORBITS BASED ON THE NODES
        adjacency_list = self.adjacency_list()
        vertex_coloring = {}
        nodes = self.nodes()
        for n in nodes:
            color = "".join(sorted(self.get_node_labels(n)))
            if color not in vertex_coloring:
                vertex_coloring[color] = set()
            vertex_coloring[color].add(n)

        g = nautyGraph(
            number_of_vertices=self.number_of_nodes(),
            directed=directed,
            adjacency_dict=adjacency_list,
            vertex_coloring=vertex_coloring.values(),
        )
        orbits_array = autgrp(g)[3]
        del g

        n_orbits = max(orbits_array) + 1
        orbits = [[] for _ in range(n_orbits)]
        for n, index in enumerate(orbits_array):
            orbits[index].append(n)

        # remove possible empty orbits
        # sometimes autgrp return placeholder for orbits skipping values
        # ex. [0, 1, 1, 3]
        orbits = [orbit for orbit in orbits if orbit]
        # ADAPT THE ORBITS BASED ON THE EDGES
        # For each orbits, we need to check if the nodes are equivalent also considering the edges
        for i, original_orbit in enumerate(orbits):
            orbit = original_orbit.copy()
            sub_orbits = []
            while orbit:
                start_node = orbit.pop()

                sub_orbit = {start_node}

                nodes_to_check = orbit.copy()

                for node in nodes_to_check:
                    if self.are_equivalent(start_node, node):
                        sub_orbit.add(node)
                        orbit.remove(node)
                sub_orbits.append(sub_orbit)
            orbits[i] = sub_orbits

        # flatten the list
        orbits = flat_map(orbits)

        return orbits

    def are_equivalent_edges(self, edge1, edge2):

        source1, target1, key1 = edge1
        source2, target2, key2 = edge2

        # Verifica se i nodi sorgente e destinazione hanno le stesse etichette
        if (
            set(self.nodes[source1]["labels"]) == set(self.nodes[source2]["labels"])
        ) and (
            set(self.nodes[target1]["labels"]) == set(self.nodes[target2]["labels"])
        ):
            # Verifica se gli archi hanno lo stesso tipo
            if self.edges[edge1]["type"] == self.edges[edge2]["type"]:
                return True

        return False

    def _initialize_from_graph_data(self, incoming_graph_data):
        for node, node_data in incoming_graph_data.nodes(data=True):
            self.add_node(node, **node_data)
        for u, v, edge_data in incoming_graph_data.edges(data=True):
            self.add_edge(u, v, **edge_data)

    def zero_index_graph(self):
        """
        Return a copy of the graph with nodes reindexed starting from 0.
        """
        mapping = {node: i for i, node in enumerate(self.nodes())}
        new_graph = type(self)()
        for node in self.nodes(data=True):
            new_graph.add_node(mapping[node[0]], **node[1])
        for edge in self.edges(data=True, keys=True):
            new_graph.add_edge(
                mapping[edge[0]], mapping[edge[1]], key=edge[2], **edge[3]
            )
        return new_graph

    def reset_memoization(self):
        self.node_labels = None
        self.edge_labels = None

    def add_node(self, node_for_adding, **attr):
        self.node_labels = None
        if "labels" not in attr:
            attr["labels"] = []
        # Delegate to actual graph implementation (MultiGraph or MultiDiGraph)
        return super().add_node(node_for_adding, **attr)

    def add_edge(self, src, dst, key=None, **attr):

        self.edge_labels = None  # Clear edge labels (if needed)

        # Auto-generate unique edge key if not provided (for multigraphs)
        if key is None:
            key = 0
            while self.has_edge(src, dst, key):
                key += 1

        # Default 'type' attribute
        if "type" not in attr:
            attr["type"] = NULL_LABEL

        # Delegate to actual graph implementation (MultiGraph or MultiDiGraph)
        return super().add_edge(src, dst, key, **attr)

    def remove_nodes(self, nodes):
        """Remove multiple nodes and clear node_labels cache."""
        self.node_labels = None
        for node in nodes:
            if self.has_node(node):
                self.remove_node(node)

    def remove_edges(self, edges):
        """Remove multiple edges and clear edge_labels cache."""
        self.edge_labels = None
        for src, dst, key in edges:
            if self.has_edge(src, dst, key):
                self.remove_edge(src, dst, key)

    def edge_has_label(self, edge):
        """Check if a given edge has a non-null label."""
        src, dst, key = edge
        try:
            return self[src][dst][key].get("type") != NULL_LABEL
        except KeyError:
            return False  # Edge missing or no type attribute

    def get_edge_labels(self, source, destination):
        labels = []
        if self.has_edge(source, destination):
            labels.extend(
                [
                    edge_data.get("type")
                    for edge_data in self[source][destination].values()
                    if edge_data.get("type") != NULL_LABEL
                ]
            )
        return sorted(set(labels))

    def get_edge_label(self, edge):
        source, destination, key = edge
        return self[source][destination][key]["type"]

    def get_node_labels(self, _id):
        if "labels" not in self.nodes[_id]:
            return []
        return sorted(list(self.nodes[_id]["labels"]))

    def get_all_node_labels(self):
        if self.node_labels is None:
            self.node_labels = sorted(
                set(flat_map([self.get_node_labels(node) for node in self.nodes]))
            )
        return self.node_labels

    def get_all_edge_labels(self):
        if self.edge_labels is None:
            self.edge_labels = sorted(
                set(
                    [
                        self.get_edge_data(edge[0], edge[1], edge[2])["type"]
                        for edge in self.edges
                        if self.get_edge_data(edge[0], edge[1], edge[2])["type"]
                        != NULL_LABEL
                    ]
                )
            )
        return self.edge_labels

    def edge_keys(self, src, dest):
        """
        Each edge is a triple (src, dest, key).

        In a multigraph it is possible to have multiple edges between the same pair of nodes.
        This method returns the keys of all edges between the same pair of nodes.
        """
        return [key for key in self[src][dest]]

    def edge_keys_by_type(self, src, dest, type):
        """
        Each edge is a triple (src, dest, key).

        In a multi graph it is possible to have multiple edges between the same pair of nodes.
        This method returns the keys of all edges between the same pair of nodes.
        """
        return [key for key in self[src][dest] if self[src][dest][key]["type"] == type]

    def miss_some_labels(self):
        """Returns true if the graph miss some labels on nodes or edges"""
        for node in self.nodes():
            if len(self.get_node_labels(node)) == 0:
                return True
        for src, dst in self.edges():
            if len(self.get_edge_labels(src, dst)) == 0:
                return True
        return False

    def edges_keys(self, edge):
        """
        Returns the edge keys between source and destination nodes.

        :param edge: tuple (source, destination)
        :return list of keys
        """
        if not self.has_edge(edge[0], edge[1]):
            return []
        return list(self[edge[0]][edge[1]].keys())

    def all_neighbors(self, node_id):
        """
        Returns all neighbors of the node with id node_id.

        :param node_id: The ID of the node.
        :return: A list of neighbors of the node.
        """
        return set(self.neighbors(node_id)) if self.has_node(node_id) else set()

    def jaccard_similarity(self, node_id_1, node_id_2):
        """
        Compute the Jaccard similarity between two nodes considering all neighbors.
        The Jaccard similarity is defined as the size of the
        intersection of the neighbors of the two nodes divided
        by the size of the union of the neighbors of the two
        nodes.
        :param node_id_1:
        :param node_id_2:
        :return:
        """
        neighbors_1 = self.all_neighbors(node_id_1)
        neighbors_2 = self.all_neighbors(node_id_2)
        intersection = neighbors_1.intersection(neighbors_2)
        union = neighbors_1.union(neighbors_2)
        return len(intersection) / len(union)

    def get_all_edges(self):
        """
        Returns a list of tuples representing all edges in the graph,
        including their keys.
        Each tuple contains (source, target, key).
        """
        all_edges = []
        for u, v, key in self.edges(keys=True):
            if u == DUMMY_LABEL or v == DUMMY_LABEL:
                continue
            all_edges.append((u, v, key))
        return all_edges

    def get_node_attributes(self, node_id):
        """
        Returns the attributes of a node, excluding the 'labels' attribute.
        :param node_id: The ID of the node.
        :return: A dictionary of attributes for the node.
        """
        if not node_id in self.nodes:
            return {}
        # delete labels from attributes
        return {k: v for k, v in self.nodes[node_id].items() if k != "labels"}

    def get_edge_attributes(self, edge):
        """
        Returns the attributes of an edge, excluding the 'type' attribute.
        :param edge: A tuple (source, target, key).
        :return: A dictionary of attributes for the edge.
        """
        if not edge in self.edges:
            return {}
        # delete type from attributes
        source, target, key = edge
        return {k: v for k, v in self[source][target][key].items() if k != "type"}

    def set_node_attributes(self, node_attributes, attribute_name):
        for node, attributes in node_attributes.items():
            if node not in self.nodes:
                self.add_node(node)  # Aggiungi il nodo se non esiste
            self.nodes[node][attribute_name] = attributes

    def set_edge_attributes(self, edge_attributes, attribute_name):
        for edge, attribute in edge_attributes.items():
            u, v = edge  # Estrai i nodi sorgente e destinazione dall'arco
            if not self.has_edge(u, v):
                self.add_edge(u, v)  # Aggiungi l'arco se non esiste
            self[u][v][0][attribute_name] = attribute  # Imposta l'attributo per l'arco

    def node_contains_attributes(self, node_id, attributes):
        """
        Checks if a node in the graph contains the specified attributes.

        Args:
            node_id: The ID of the node to check.
            attributes: A dictionary containing the attributes to check for,
                        where keys are attribute names and values are attribute values.

        Returns:
            True if the node contains all the specified attributes, False otherwise.
        """
        return node_id in self.nodes and all(
            attr in self.nodes[node_id].items() for attr in attributes.items()
        )

    def edge_contains_attributes(self, edge, attributes):
        """
        Checks if an edge in the graph contains the specified attributes.

        Args:
            source: The source node of the edge.
            target: The target node of the edge.
            attributes: A dictionary containing the attributes to check for,
                        where keys are attribute names and values are attribute values.

        Returns:
            True if the edge contains all the specified attributes, False otherwise.
        """
        source, target, key = edge
        return self.has_edge(source, target, key) and all(
            attr in self[source][target][key].items() for attr in attributes.items()
        )

    def subgraph(self, nodes):
        """
        Returns a subgraph induced by the given nodes.

        Args:
            nodes: A list of node IDs.

        Returns:
            A MultiDiGraph object representing the subgraph induced by the given nodes.
        """
        subgraph = type(self)()
        for node in nodes:
            subgraph.add_node(node, **self.nodes[node])
        for u, v, key, data in self.edges(keys=True, data=True):
            if u in nodes and v in nodes:
                subgraph.add_edge(u, v, key, **data)
        return subgraph

    def get_edge_labels_with_duplicate(self, source, destination):
        if not self.has_edge(source, destination):
            return []

        return [
            edge_data["type"]
            for edge_data in self[source][destination].values()
            if edge_data.get("type") != ""
        ]

    def canonical_code_template(self, node) -> str:  # FIXME: find a better name
        """
        Piece of code to be used in the canonical code function.
        It is implemented in the subclasses.

        It return a string with the following format:
        <node_labels><edge_labels>
        where:
            - node_labels: labels of the node
            - edge_labels: labels of the edges
        """
        # This method should be implemented in subclasses
        raise NotImplementedError("Subclasses must implement this method.")

    def permutated_adjacency_matrix(self, permutation, reindex):
        """
        Returns the adjacency matrix of the graph.
        It is implemented in the subclasses.

        Parameters:
            permutation: A permutation of the nodes of the graph.

        Returns:
            A list of lists representing the adjacency matrix of the graph.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def create_igraph_object(self, edges):
        """
        Create an igraph object from the edges of the graph.
        It is implemented in the subclasses.

        Parameters:
            edges: A list of edges of the graph.

        Returns:
            An igraph object representing the graph.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def canonical_code(self):

        # ig the graph has nodes id different from 0 to n-1, we need to reindex the nodes
        reindex = {node: i for i, node in enumerate(sorted(self.nodes()))}

        edges = [(reindex[src], reindex[dst]) for src, dst in self.edges(keys=False)]

        if len(edges) == 0:
            # Se il grafo non ha archi, restituisci solo le etichette dei nodi ordinate
            return "".join(sorted(self.get_node_labels(0)))

        # Creazione del grafo per igraph
        igraph = self.create_igraph_object(edges)
        nodes = sorted(self.nodes())
        node_colors = [self.canonical_code_template(node) for node in nodes]

        # Mappa dei colori a interi
        node_colors_map_to_int = {
            color: i for i, color in enumerate(sorted(set(node_colors)))
        }
        node_colors_int = [node_colors_map_to_int[color] for color in node_colors]

        # Calcolo della permutazione canonica
        canonical_permutation = igraph.canonical_permutation(color=node_colors_int)

        del igraph

        # # Permuta la matrice di adiacenza
        adjacency_matrix = self.permutated_adjacency_matrix(
            canonical_permutation, reindex
        )

        inverse_canonical_permutation = {
            canonical_permutation[i]: i for i in range(len(canonical_permutation))
        }

        inverse_reindex = {v: k for k, v in reindex.items()}
        # Costruisci il codice canonico
        code = ""
        for i, row in enumerate(adjacency_matrix):
            src = inverse_canonical_permutation[i]
            # Aggiungi le etichette del nodo di origine
            src_labels = "".join(sorted(self.get_node_labels(inverse_reindex[src])))
            code += src_labels
            for j, bit in enumerate(row):
                if bit == 1:  # C'è un arco
                    dst = inverse_canonical_permutation[j]

                    # Aggiungi le etichette del nodo di destinazione e dell'arco
                    dst_labels = "".join(
                        sorted(self.get_node_labels(inverse_reindex[dst]))
                    )
                    edge_labels = "".join(
                        sorted(
                            self.get_edge_labels_with_duplicate(
                                inverse_reindex[src], inverse_reindex[dst]
                            )
                        )
                    )
                    code += (
                        "1" + dst_labels + edge_labels
                    )  # if the graph has no labels, the code will be 1
                else:
                    # Aggiungi uno 0 per indicare assenza di arco
                    code += "0"

        return code
