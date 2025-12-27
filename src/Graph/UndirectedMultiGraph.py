import networkx as nx
import random
from igraph import Graph as igraphGraph
from pynauty import Graph as nautyGraph, autgrp
from more_itertools.more import adjacent
from sympy import false
from collections import defaultdict
from .AbstractGraph import AbstractGraph


def flat_map(list_of_lists):
    array = []
    for l in list_of_lists:
        array.extend(l)
    return array


NULL_LABEL = ""


class UndirectedMultiGraph(AbstractGraph, nx.MultiGraph):

    def __init__(self, incoming_graph_data=None, **attr):
        super().__init__(incoming_graph_data, **attr)

    def are_equivalent(self, node1, node2):
        # Verifica se le etichette di node1 sono un sottoinsieme o uguali alle etichette di node2
        if set(self.nodes[node1]["labels"]) == set(self.nodes[node2]["labels"]):
            # Verifica se gli attributi degli archi in uscita sono gli stessi
            edges_node1 = sorted(
                [self.edges[edge]["type"] for edge in self.edges(node1, keys=True)]
            )
            edges_node2 = sorted(
                [self.edges[edge]["type"] for edge in self.edges(node2, keys=True)]
            )
            if edges_node1 != edges_node2:
                return False
        else:
            return False

    def t_out_deg(self, node_id, label):
        """
        Returns the number of outgoing edges with a specific label from the node with id node_id.

        :param node_id: The ID of the node.
        :param label: The label of the edges to consider.
        :return: The number of outgoing edges with the specified label.
        """
        count = 0
        for _, _, edge_data in self.edges(node_id, data=True):
            if edge_data["type"] == label:
                count += 1
        return count

    def t_in_deg(self, node_id, label):
        """
        Returns the number of incoming edges with a specific label to the node with id node_id.

        :param node_id: The ID of the node.
        :param label: The label of the edges to consider.
        :return: The number of incoming edges with the specified label.
        """
        count = 0
        for _, _, edge_data in self.edges(node_id, data=True):
            if edge_data["type"] == label:
                count += 1
        return count

    def tot_deg(self, node_id):
        """
        Returns the total degree of the node with id node_id.

        :param node_id: The ID of the node.
        :return: The total degree of the node.
        """
        return self.degree(node_id)

    def compute_orbits_nodes(self):
        return self.compute_orbits_nodes_helper(directed=False)

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
        u, v = edge
        if u > v:
            u, v = v, u
        return [(u, v, k) for k in self[u][v]] if self.has_edge(u, v) else []

    def all_neighbors(self, node_id):
        """
        Returns all neighbors of the node with id node_id.

        :param node_id: The ID of the node.
        :return: A list of neighbors of the node.
        """
        return set(self.neighbors(node_id)) if self.has_node(node_id) else set()

    def adjacency_list(self):
        adj_list = {}
        for u, v in self.edges():
            if u not in adj_list:
                adj_list[u] = set()
            if v not in adj_list:
                adj_list[v] = set()
            adj_list[u].add(v)
            adj_list[v].add(u)

        # remove duplicates
        for node in adj_list:
            adj_list[node] = list(adj_list[node])
        return adj_list

    def canonical_code_template(self, node):
        return "".join(sorted(self.get_node_labels(node))) + "".join(
            sorted([self.edges[edge]["type"] for edge in self.edges(node, keys=True)])
        )

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
            matrix[permutation[reindex[v]]][permutation[reindex[u]]] = 1

        return matrix

    def create_igraph_object(self, edges):
        return igraphGraph(edges=edges, directed=False)
