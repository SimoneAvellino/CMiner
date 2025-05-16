import networkx as nx
import random
from igraph import Graph as igraphGraph
from pynauty import Graph as nautyGraph, autgrp
from more_itertools.more import adjacent
from sympy import false
from collections import defaultdict


# TO DO: improve random query generation
# TO DO: come è strutturato l'arco esempio di tupla (source, target, key, id ) ?
# To DO: valutare assieme i metodi set_edge_attributes, are_equivalent_edge, compute_orbits_edge,edge_contain_attributes
# breaking_condition, e edge_id


def flat_map(list_of_lists):
    array = []
    for l in list_of_lists:
        array.extend(l)
    return array


NULL_LABEL = ""


# TO DO: valutare se ordinare le etichette o no
class MultiDiGraph(nx.MultiDiGraph):

    def __init__(self, incoming_graph_data=None, multigraph_input=None, **attr):
        super().__init__(**attr)
        self.node_labels = None
        self.edge_labels = None
        if incoming_graph_data:
            self._initialize_from_graph_data(incoming_graph_data)
            

    def _initialize_from_graph_data(self, incoming_graph_data):
        for node, node_data in incoming_graph_data.nodes(data=True):
            self.add_node(node, **node_data)
        edge_key = 0
        for u, v, edge_data in incoming_graph_data.edges(data=True):
            self.add_edge(u, v, key=edge_key, **edge_data)
            edge_key += 1
            
    def zero_index_graph(self):
        """
        Return a copy of the graph with nodes reindexed starting from 0.
        """
        mapping = {node: i for i, node in enumerate(self.nodes())}
        new_graph = MultiDiGraph()
        for node in self.nodes(data=True):
            new_graph.add_node(mapping[node[0]], **node[1])
        for edge in self.edges(data=True, keys=True):
            new_graph.add_edge(mapping[edge[0]], mapping[edge[1]], key=edge[2], **edge[3])
        return new_graph


    def reset_memoization(self):
        self.node_labels = None
        self.edge_labels = None

    def add_edge(self, u_of_edge, v_of_edge, key=None, **attr):
        self.edge_labels = None
        if key is None:
            key = 0
            while self.has_edge(u_of_edge, v_of_edge, key):
                key += 1
        # check if the user pass the type attribute
        if 'type' not in attr:
            # set the type attribute to the default value
            attr['type'] = NULL_LABEL
        super().add_edge(u_of_edge, v_of_edge, key, **attr)

    def add_node(self, node_for_adding, **attr):
        self.node_labels = None
        if 'labels' not in attr:
            attr['labels'] = []
        super().add_node(node_for_adding, **attr)

    def remove_nodes(self, n):
        self.node_labels = None
        for node in n:
            self.remove_node(node)

    def remove_edges(self, edges):
        self.edge_labels = None
        for edge in edges:
            self.remove_edge(edge[0], edge[1], edge[2])


    def edge_has_label(self, edge):
        source, target, key = edge
        return self[source][target][key]['type'] != NULL_LABEL

    def get_edge_labels(self, source, destination):
        labels = []
        if self.has_edge(source, destination):
            labels.extend([edge_data.get('type') for edge_data in self[source][destination].values() if edge_data.get('type') != NULL_LABEL])
        return sorted(set(labels))

    def get_edge_label(self, edge):
        source, destination, key = edge
        return self[source][destination][key]['type']

    def get_node_labels(self, _id):
        if 'labels' not in self.nodes[_id]:
            return []
        return sorted(list(self.nodes[_id]["labels"]))

    def get_all_node_labels(self):
        if self.node_labels is None:
            self.node_labels = sorted(set(flat_map([self.get_node_labels(node) for node in self.nodes])))
            # self.node_labels = sorted(set(flat_map([self.nodes[node]['labels'] for node in self.nodes])))
        return self.node_labels

    def get_all_edge_labels(self):
        if self.edge_labels is None:
            self.edge_labels = sorted(
                set([self.get_edge_data(edge[0], edge[1], edge[2])['type'] for edge in self.edges if self.get_edge_data(edge[0], edge[1], edge[2])['type'] != NULL_LABEL]))
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
        return [key for key in self[src][dest] if self[src][dest][key]['type'] == type]

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
            edges.extend((node_id_1, node_id_2, key) for key in self[node_id_1][node_id_2])
        if self.has_edge(node_id_2, node_id_1):
            edges.extend((node_id_2, node_id_1, key) for key in self[node_id_2][node_id_1])
        return edges
    
    def miss_some_labels(self):
        """ Returns true if the pattern miss some labels on nodes or edges """
        for node in self.nodes():
            if len(self.get_node_labels(node)) == 0:
                return True
        for src, dst in self.edges():
            if len(self.get_edge_labels(src, dst)) == 0:
                return True

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
        It does not consider the direction of the edges.
        :param node_id: The ID of the node.
        :return: A list of neighbors of the node.
        """
        return set(self.successors(node_id)) | set(self.predecessors(node_id))

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

    def generate_random_query(self, num_nodes, num_edges):
        """Generate a random query graph with num_nodes nodes and num_edges edges."""
        # Create a graph
        G = MultiDiGraph()

        # Add nodes with random labels
        for i in range(num_nodes):
            label = random.choice(self.get_all_node_labels())
            G.add_node(node_for_adding=i, labels=[label])

        # Add edges with random labels
        for _ in range(num_edges):
            u, v = random.sample(range(num_nodes), 2)
            label = random.choice(self.get_all_edge_labels())
            G.add_edge(u, v, type=label)

        return G

    def get_connected_subgraph_with_n_nodes(self, n):
        # Initialize the set of visited nodes
        visited = set()

        # Choose a random starting node
        start_node = random.choice(list(self.nodes()))

        # Initialize a queue for BFS
        queue = [start_node]

        # Explore the graph using BFS until finding a connected subgraph with n nodes
        while queue:
            node = queue.pop(0)
            visited.add(node)

            # If the connected subgraph has n nodes, return it
            if len(visited) == n:
                return self.subgraph(visited).copy()

            # Add neighboring nodes to the queue
            neighbors = list(self.neighbors(node))
            for neighbor in neighbors:
                if neighbor not in visited:
                    queue.append(neighbor)

        # If no connected subgraph with n nodes is found, return None
        return None

    def get_all_edges(self):
        """
        Returns a list of tuples representing all edges in the graph,
        including their keys.
        Each tuple contains (source, target, key).
        """
        all_edges = []
        for u, v, key in self.edges(keys=True):
            if u == "dummy" or v == "dummy":
                continue
            all_edges.append((u, v, key))
        return all_edges

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
        count = sum(1 for _, _, attrs in out_edges if attrs.get('type') == t)

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
        count = sum(1 for _, _, attrs in in_edges if attrs.get('type') == t)

        return count

    def are_equivalent(self, node1, node2):
        # Verifica se le etichette di node1 sono un sottoinsieme o uguali alle etichette di node2
        if set(self.nodes[node1]['labels']) == set(self.nodes[node2]['labels']):
            # Verifica se gli attributi degli archi in uscita sono gli stessi
            out_edges_node1 = sorted([self.edges[edge]['type'] for edge in self.out_edges(node1, keys=True)])
            out_edges_node2 = sorted([self.edges[edge]['type'] for edge in self.out_edges(node2, keys=True)])
            if out_edges_node1 != out_edges_node2:
                return False

            # Verifica se gli attributi degli archi in entrata sono gli stessi
            in_edges_node1 = sorted([self.edges[edge]['type'] for edge in self.in_edges(node1, keys=True)])
            in_edges_node2 = sorted([self.edges[edge]['type'] for edge in self.in_edges(node2, keys=True)])
            if in_edges_node1 != in_edges_node2:
                return False
            # Se tutte le condizioni sono soddisfatte, i nodi sono equivalenti
            return True
        else:
            return False

    def adjacency_list(self):
        adj_list = {}
        for u, v in self.edges():
            if u not in adj_list:
                adj_list[u] = []
            adj_list[u].append(v)
        return adj_list

    def compute_orbits_nodes(self):

        # COMPUTING THE ORBITS BASED ON THE NODES
        adjacency_list = self.adjacency_list()
        vertex_coloring = {}
        nodes = self.nodes()
        for n in nodes:
            color = "".join(sorted(self.get_node_labels(n)))
            if color not in vertex_coloring:
                vertex_coloring[color] = set()
            vertex_coloring[color].add(n)

        g = nautyGraph(number_of_vertices=self.number_of_nodes(), directed=True, adjacency_dict=adjacency_list, vertex_coloring=vertex_coloring.values())
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
        if (set(self.nodes[source1]['labels']) == set(self.nodes[source2]['labels'])) and \
                (set(self.nodes[target1]['labels']) == set(self.nodes[target2]['labels'])):
            # Verifica se gli archi hanno lo stesso tipo
            if self.edges[edge1]['type'] == self.edges[edge2]['type']:
                return True

        return False

    def get_node_attributes(self, node_id):
        # delete labels from attributes
        return {k: v for k, v in self.nodes[node_id].items() if k != 'labels'}

    def get_edge_attributes(self, edge):
        # delete type from attributes
        source, target, key = edge
        return {k: v for k, v in self[source][target][key].items() if k != 'type'}

    def set_node_attributes(self, node_attributes, attribute_name):
        for node, attributes in node_attributes.items():
            if node not in self.nodes:
                self.add_node(node)  # Aggiungi il nodo se non esiste
            self.nodes[node][attribute_name] = attributes

    # Metodo per impostare gli attributi degli archi
    def set_edge_attributes(self, edge_attributes, attribute_name):
        for edge, attribute in edge_attributes.items():
            u, v = edge  # Estrai i nodi sorgente e destinazione dall'arco
            if not self.has_edge(u, v):
                self.add_edge(u, v)  # Aggiungi l'arco se non esiste
            self[u][v][0][attribute_name] = attribute  # Imposta l'attributo per l'arco

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
        return node_id in self.nodes and all(attr in self.nodes[node_id].items() for attr in attributes.items())

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
            attr in self[source][target][key].items() for attr in attributes.items())

    # Valutare con Simone --->Funzionamento + Utility
    def compute_symmetry_breaking_conditions(self):
        """
        Computes the symmetry breaking conditions for both nodes and edges in the graph.

        Returns:
            A list containing the symmetry breaking conditions for nodes and edges.
            Each element in the list is a list of conditions for either nodes or edges,
            where each condition is represented as a list of node IDs or edge tuples.
        """
        # Compute orbits for nodes and edges
        node_orbits = self.compute_orbits_nodes()
        edge_orbits = self.compute_orbits_edges()

        # List to store the symmetry breaking conditions for nodes and edges
        breaking_conditions = []

        # Compute symmetry breaking conditions for nodes
        node_breaking_conditions = []
        for orbit in node_orbits:
            if len(orbit) > 1:
                smallest_node = min(orbit)
                # Sort the node IDs within each orbit for consistency
                condition = sorted(orbit, key=lambda node: node)
                node_breaking_conditions.append(condition)

        # Compute symmetry breaking conditions for edges
        edge_breaking_conditions = []
        for orbit in edge_orbits:
            if len(orbit) > 1:
                smallest_edge = min(orbit)
                # Sort the edge tuples within each orbit based on their third element (ID) for consistency
                condition = sorted(orbit, key=lambda edge: edge[2])
                edge_breaking_conditions.append(condition)

        # Append node and edge breaking conditions to the main list
        breaking_conditions.append(node_breaking_conditions)
        breaking_conditions.append(edge_breaking_conditions)

        return breaking_conditions

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

    @staticmethod
    def generate_graph(pattern, times, seed = None):
        """
        Generate a graph by repeating a given pattern multiple times.

        Args:
            pattern: The pattern graph to be repeated.
            times: The number of times the pattern should be repeated.

        Returns:
            A new graph containing 'times' repetitions of the pattern, connected to form a connected graph.
        """
        # Create a new MultiDiGraph
        mapping = {}
        new_graph = MultiDiGraph()

        if seed is not None:
            random.seed(seed)

        # Repeat the pattern 'times' times
        for i in range(times):
            # Add nodes and edges from the pattern to the new graph
            mapping = {}  # Mapping of old node IDs to new node IDs in the new graph
            for node in pattern.nodes(data=True):
                # Add node to the new graph
                new_node_id = new_graph.number_of_nodes()  # Generate unique node ID
                # Add node to the new graph considering all the attributes
                new_graph.add_node(new_node_id, **node[1])
                mapping[node[0]] = new_node_id

            for edge in pattern.edges(data=True, keys=True):
                # Add edge to the new graph
                source = mapping[edge[0]]
                target = mapping[edge[1]]
                new_graph.add_edge(source, target, key=edge[2], **edge[3])

            # Connect the pattern to the new graph
            if i > 0:
                # Create up to num_nodes random edges between pattern and new_graph
                num_new_edges = random.randint(1, len(pattern.nodes()))
                for _ in range(num_new_edges):
                    # take a random node from the pattern and a random node from the new graph
                    node_pattern = random.choice(list(pattern.nodes()))
                    # the new node on the graph cannot be the same as the node in the pattern (no loops)
                    cand = list(set(new_graph.nodes()).difference(set([mapping[n] for n in pattern.nodes()])))
                    node_new_graph = random.choice(cand)
                    # randomize the edge direction
                    if random.random() < 0.5:
                        src = node_new_graph
                        dest = mapping[node_pattern]
                    else:
                        src = mapping[node_pattern]
                        dest = node_new_graph
                    new_graph.add_edge(src, dest, type=random.choice(new_graph.get_all_edge_labels()))

        return new_graph


    def subgraph(self, nodes):
        """
        Returns a subgraph induced by the given nodes.

        Args:
            nodes: A list of node IDs.

        Returns:
            A MultiDiGraph object representing the subgraph induced by the given nodes.
        """
        subgraph = MultiDiGraph()
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
            edge_data['type']
            for edge_data in self[source][destination].values()
            if edge_data.get('type') != ""
        ]

    def code(self):
        # sort nodes by out degree
        nodes = sorted(self.nodes(), key=lambda n: self.out_deg(n))
        # group nodes by out degree
        groups = {}
        for n in nodes:
            deg = self.out_deg(n)
            if deg not in groups:
                groups[deg] = []
            groups[deg].append(n)
        # sort nodes within each group by node labels
        for deg in groups:
            groups[deg] = sorted(groups[deg], key=lambda n: "".join(self.get_node_labels(n)))
        # compute code
        code = ""
        for deg in sorted(groups.keys()):
            edge_codes = {}
            for n in groups[deg]:
                # for each node of the group
                edge_labels = []
                for v in self.successors(n):
                    edge_labels.extend(self.get_edge_labels_with_duplicate(n, v))
                edge_codes[n] = "".join(sorted(edge_labels)) # concatenate all edge labels
            # create the code by grouping by keys and sort each group
            for node, edge_code in sorted(edge_codes.items(), key=lambda x: x[1]):
                code += "".join(self.get_node_labels(node)) + edge_code

        return code

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
                keys = self.edge_keys(dst, src) if self.has_edge(dst, src, key=key) else [-1]
            edge_to_add[(dst, src, type)].append( max(keys) + 1)
        
        for (src, dst, type), keys in edge_to_add.items():
            for key in keys:
                self.add_edge(src, dst, key=key, type=type, dummy=True)
                
    def remove_reverse_edges(self):
        """
        Remove reverse edges from the graph.
        """
        edges_to_remove = [
            (src, dst, key) for src, dst, key, data in self.edges(keys=True, data=True) if 'dummy' in data
        ]
        for src, dst, key in edges_to_remove:
            self.remove_edge(src, dst, key=key)

    def a(self, node):
        return "".join(sorted([self.edges[edge]['type'] for edge in self.edges(node, keys=True)]))

    def b(self, node):
        return "".join(sorted([self.edges[edge]['type'] for edge in self.in_edges(node, keys=True)]))

    def canonical_code(self):
        
        # ig the graph has nodes id different from 0 to n-1, we need to reindex the nodes
        reindex = {node: i for i, node in enumerate(sorted(self.nodes()))}
        
        edges = [(reindex[src], reindex[dst]) for src, dst in self.edges(keys=False)]

        if len(edges) == 0:
            # Se il grafo non ha archi, restituisci solo le etichette dei nodi ordinate
            return "".join(sorted(self.get_node_labels(0)))

        # Creazione del grafo per igraph
        igraph = igraphGraph(edges=edges, directed=True)
        nodes = sorted(self.nodes())
        node_colors = ["".join(sorted(self.get_node_labels(node))) + self.a(node) + self.b(node)  for node in nodes]
        # print(node_colors)
        # Mappa dei colori a interi
        node_colors_map_to_int = {color: i for i, color in enumerate(sorted(set(node_colors)))}
        node_colors_int = [node_colors_map_to_int[color] for color in node_colors]
        


        # Calcolo della permutazione canonica
        canonical_permutation = igraph.canonical_permutation(color=node_colors_int)

        del igraph

        # # Permuta la matrice di adiacenza
        adjacency_matrix = self.permutated_adjacency_matrix(canonical_permutation, reindex)

        inverse_canonical_permutation = {canonical_permutation[i]: i for i in range(len(canonical_permutation))}

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
                    dst_labels = "".join(sorted(self.get_node_labels(inverse_reindex[dst])))
                    edge_labels = "".join(sorted(self.get_edge_labels_with_duplicate(inverse_reindex[src], inverse_reindex[dst])))
                    code += "1" + dst_labels + edge_labels # if the graph has no labels, the code will be 1
                else:
                    # Aggiungi uno 0 per indicare assenza di arco
                    code += "0"

        return code