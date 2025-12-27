import networkx as nx
import random
from igraph import Graph as igraphGraph
from pynauty import Graph as nautyGraph, autgrp
from more_itertools.more import adjacent
from sympy import false
from collections import defaultdict
from .constants import NULL_LABEL
from .AbstractGraph import AbstractGraph


def flat_map(list_of_lists):
    array = []
    for l in list_of_lists:
        array.extend(l)
    return array


class DirectedMultiGraph(AbstractGraph, nx.MultiDiGraph):

    def __init__(self, incoming_graph_data=None, **attr):
        super().__init__(incoming_graph_data, **attr)

    def get_edges_consider_no_direction(self, edge):
        """
        Returns the edges between source and destination nodes, not considering the direction of the edges.
        EXAMPLE:
            G = MultiDiGraph()
            G.add_edge(1, 2, type="A")
            G.add_edge(2, 1, type="B")
            G.get_edges_consider_no_direction(1, 2) -> [(1, 2, 0), (2, 1, 0)]
        :param edge: tuple (source, destination)
        :return list of edges between source and destination nodes
        """
        node_id_1, node_id_2 = edge
        edges = []
        if self.has_edge(node_id_1, node_id_2):
            edges.extend(
                (node_id_1, node_id_2, key) for key in self[node_id_1][node_id_2]
            )
        if self.has_edge(node_id_2, node_id_1):
            edges.extend(
                (node_id_2, node_id_1, key) for key in self[node_id_2][node_id_1]
            )
        return edges

    def all_neighbors(self, node_id):
        """
        Returns all neighbors of the node with id node_id.
        It does not consider the direction of the edges.
        :param node_id: The ID of the node.
        :return: A list of neighbors of the node.
        """
        return set(self.successors(node_id)) | set(self.predecessors(node_id))

    def tot_deg(self, node_id):
        """
        Returns the total degree of the node with id node_id.

        :param node_id: The ID of the node.
        :return: The total degree of the node.
        """
        return self.in_deg(node_id) + self.out_deg(node_id)

    def in_deg(self, node_id):
        """
        Returns the in-degree of the node with id node_id.

        :param node_id: The ID of the node.
        :return: The in-degree of the node.
        """
        return len(self.in_edges(node_id))

    def out_deg(self, node_id):
        """
        Returns the out-degree of the node with id node_id.

        :param node_id: The ID of the node.
        :return: The out-degree of the node.
        """
        return len(self.out_edges(node_id))

    def t_out_deg(self, node_id, t):
        """
        Returns the number of edges that exit from the node with id node_id and have label t.

        Parameters:
        - node_id: The ID of the node.
        - t: The label of the edge.

        Returns:
        - The number of edges that exit from the node with label t.
        """
        # Get all outgoing edges from the node with node_id
        out_edges = self.out_edges(node_id, data=True)

        # Count the number of edges with label t
        count = sum(1 for _, _, attrs in out_edges if attrs.get("type") == t)

        return count

    def t_in_deg(self, node_id, t):
        """
        Returns the number of edges that enter the node with id node_id and have label t.

        Parameters:
        - node_id: The ID of the node.
        - t: The label of the edge.

        Returns:
        - The number of edges that enter the node with label t.
        """
        # Get all incoming edges to the node with node_id
        in_edges = self.in_edges(node_id, data=True)

        # Count the number of edges with label t
        count = sum(1 for _, _, attrs in in_edges if attrs.get("type") == t)

        return count

    def adjacency_list(self):
        adj_list = {}
        for u, v in self.edges():
            if u not in adj_list:
                adj_list[u] = set()
            adj_list[u].add(v)

        for node in adj_list:
            adj_list[node] = list(adj_list[node])
        return adj_list

    def canonical_code_template(self, node):
        return (
            "".join(sorted(self.get_node_labels(node)))
            + "".join(
                sorted(
                    [
                        self.edges[edge]["type"]
                        for edge in self.out_edges(node, keys=True)
                    ]
                )
            )
            + "".join(
                sorted(
                    [
                        self.edges[edge]["type"]
                        for edge in self.in_edges(node, keys=True)
                    ]
                )
            )
        )

    def are_equivalent(self, node1, node2):
        # Verifica se le etichette di node1 sono un sottoinsieme o uguali alle etichette di node2
        if set(self.nodes[node1]["labels"]) == set(self.nodes[node2]["labels"]):
            # Verifica se gli attributi degli archi in uscita sono gli stessi
            out_edges_node1 = sorted(
                [self.edges[edge]["type"] for edge in self.out_edges(node1, keys=True)]
            )
            out_edges_node2 = sorted(
                [self.edges[edge]["type"] for edge in self.out_edges(node2, keys=True)]
            )
            if out_edges_node1 != out_edges_node2:
                return False

            # Verifica se gli attributi degli archi in entrata sono gli stessi
            in_edges_node1 = sorted(
                [self.edges[edge]["type"] for edge in self.in_edges(node1, keys=True)]
            )
            in_edges_node2 = sorted(
                [self.edges[edge]["type"] for edge in self.in_edges(node2, keys=True)]
            )
            if in_edges_node1 != in_edges_node2:
                return False
            # Se tutte le condizioni sono soddisfatte, i nodi sono equivalenti
            return True
        else:
            return False

    def permutated_adjacency_matrix(self, permutation, reindex):
        """
        Returns the adjacency matrix of the graph.

        Parameters:
            permutation: A permutation of the nodes of the graph.

        Returns:
            A list of lists representing the adjacency matrix of the graph.
        """
        # Initialize the adjacency matrix with zeros
        matrix = [[0] * self.number_of_nodes() for _ in range(self.number_of_nodes())]

        # Fill the adjacency matrix with the edge labels
        for u, v in self.edges(keys=False):
            matrix[permutation[reindex[u]]][permutation[reindex[v]]] = 1

        return matrix

    def create_igraph_object(self, edges):
        return igraphGraph(edges=edges, directed=True)

    def add_reverse_edges(self):
        """
        Add reverse edges to the graph.
        """
        edge_to_add = defaultdict(list)
        for src, dst, key in self.edges(keys=True):
            type = self.get_edge_label((src, dst, key))
            if (dst, src, type) in edge_to_add:
                edge_to_add[(dst, src, type)].append(max(self.edge_keys(dst, src)) + 1)
            else:
                keys = (
                    self.edge_keys(dst, src)
                    if self.has_edge(dst, src, key=key)
                    else [-1]
                )
            edge_to_add[(dst, src, type)].append(max(keys) + 1)

        for (src, dst, type), keys in edge_to_add.items():
            for key in keys:
                self.add_edge(src, dst, key=key, type=type, dummy=True)

    def remove_reverse_edges(self):
        """
        Remove reverse edges from the graph.
        """
        edges_to_remove = [
            (src, dst, key)
            for src, dst, key, data in self.edges(keys=True, data=True)
            if "dummy" in data
        ]
        for src, dst, key in edges_to_remove:
            self.remove_edge(src, dst, key=key)

    # ---- FIXME: choose the correct class ----
    def compute_orbits_nodes(self):
        return self.compute_orbits_nodes_helper(directed=True)

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

    def compute_orbits_edges(self):

        orbits = []
        unvisited_edges = set(self.edges(keys=True))

        while unvisited_edges:
            start_edge = unvisited_edges.pop()
            orbit = {start_edge}
            edges_to_check = unvisited_edges.copy()

            for edge in edges_to_check:
                if self.are_equivalent_edges(start_edge, edge):
                    orbit.add(edge)
                    unvisited_edges.remove(edge)

            orbits.append(orbit)
        return orbits

    def compute_orbits_edges_id(self):
        orbits = []
        unvisited_edges = set(self.edges(keys=True))

        while unvisited_edges:
            start_edge = unvisited_edges.pop()
            orbit = {start_edge}
            edges_to_check = unvisited_edges.copy()

            for edge in edges_to_check:
                if self.are_equivalent_edges(start_edge, edge):
                    orbit.add(edge)
                    unvisited_edges.remove(edge)

            orbits.append(list(orbit))

        return orbits

    def compute_orbits_nodes_id(self):
        orbits = []
        unvisited_nodes = set(self.nodes())

        while unvisited_nodes:
            start_node = unvisited_nodes.pop()
            orbit = {start_node}
            nodes_to_check = unvisited_nodes.copy()

            for node in nodes_to_check:
                if self.are_equivalent(start_node, node):
                    orbit.add(node)
                    unvisited_nodes.remove(node)

            orbits.append(list(orbit))

        return orbits
