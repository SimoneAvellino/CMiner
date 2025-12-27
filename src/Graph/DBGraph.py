import itertools
from .DirectedMultiGraph import DirectedMultiGraph
from .UndirectedMultiGraph import UndirectedMultiGraph
from MultiGraphMatch.BitMatrix import BitMatrixStrategy2, TargetBitMatrixOptimized
from MultiGraphMatch.MultiGraphMatch import MultiGraphMatch


class DBGraph:
    """
    Represents a graph in the database.
    """
    def __init__(self, graph, name):
        self.graph = graph
        self.name = name

    def get_graph(self):
        return self.graph

    def get_name(self):
        return self.name

    def __str__(self):
        return self.name

class DirectedDBGraph(DirectedMultiGraph):
    
    def __init__(self, graph, name):
        """
        Represents a directed graph in the database.
        """
        super().__init__(graph)
        self.name = name
        self.matcher = None

    def _init_matcher(self):
        """
        Initialize the matching algorithm.
        """
        bit_matrix = TargetBitMatrixOptimized(self, BitMatrixStrategy2())
        bit_matrix.compute()
        self.matcher = MultiGraphMatch(self, target_bit_matrix=bit_matrix)

    def localize(self, pattern) -> list['Mapping']:
        """
        Find all the mappings of the pattern in the graph.
        """
        if self.matcher is None:
            self._init_matcher()
        
        return self.matcher.match(pattern)

    def get_name(self):
        """
        Return the name of the graph
        """
        return self.name

    def __str__(self):
        return self.name

    def all_edges_of_subgraph(self, nodes):
        """
        Return all edges of the subgraph induced by the given nodes.
        The edges are returned as a set of tuples (u, v, key).
        """
        nodes_set = set(nodes)
        edges = set()

        for src, dst in itertools.combinations(nodes, 2):
            for u, v, key in self.edges([src, dst], keys=True):
                if u in nodes_set and v in nodes_set:
                    edges.add((u, v, key))

        return edges
        
class UndirectedDBGraph(UndirectedMultiGraph):
    
    def __init__(self, graph, name):
        """
        Represents a directed graph in the database.
        """
        super().__init__(graph)
        self.name = name
        self.matcher = None

    def _init_matcher(self):
        """
        Initialize the matching algorithm.
        """
        bit_matrix = TargetBitMatrixOptimized(self, BitMatrixStrategy2())
        bit_matrix.compute()
        self.matcher = MultiGraphMatch(self, target_bit_matrix=bit_matrix)

    def localize(self, pattern) -> list['Mapping']:
        """
        Find all the mappings of the pattern in the graph.
        """
        if self.matcher is None:
            self._init_matcher()
        
        return self.matcher.match(pattern)

    def get_name(self):
        """
        Return the name of the graph
        """
        return self.name

    def __str__(self):
        return self.name

    def all_edges_of_subgraph(self, nodes):
        """
        Return all edges of the subgraph induced by the given nodes.
        The edges are returned as a set of tuples (u, v, key).
        """
        nodes_set = set(nodes)
        edges = set()

        for src, dst in itertools.combinations(nodes, 2):
            if src > dst:
                src, dst = dst, src
            for u, v, key in self.edges([src, dst], keys=True):
                if u < v and u in nodes_set and v in nodes_set:
                    edges.add((u, v, key))

        return edges