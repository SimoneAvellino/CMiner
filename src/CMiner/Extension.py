
class Extension:
    """
    Base class representing an extension of a pattern in the database.
    """
    def __init__(self, location):
        """
        Initialize the extension with a location.
        """
        self.location = location

    def graphs(self):
        """
        Return the graphs where the extension is found.
        """
        return list(self.location.keys())

    def mapping(self, graph):
        """
        Return the mapping of the pattern in the graph.
        """
        return self.location[graph]

    def set_graphs(self, graphs):
        """
        Set the graphs where the extension is found.
        """
        self.location = {g: {} for g in graphs}

    def target_node_ids(self, graph, _map):
        """
        Return the target node ids from which the extension is found.
        """
        if _map not in self.location[graph]:
            return []
        return self.location[graph][_map]


class UndirectedExtension(Extension):
    
    def __init__(self, edge_labels, location):
        """
        Represents an extension of the pattern in the database for undirected graphs.
        """
        super().__init__(location)
        self.edge_labels = edge_labels
        
    def __str__(self):
        return f"UndirectedExtension(edge_labels={self.edge_labels})"
        
    def __copy__(self):
        """
        Create a copy of the UndirectedExtension instance.
        """
        return UndirectedExtension(self.edge_labels, self.location.copy())
    

class DirectedExtension(Extension):

    def __init__(self, out_edge_labels, in_edge_labels, location):
        """
        Represents an extension of the pattern in the database for directed graphs.
        """
        super().__init__(location)
        self.out_edge_labels = out_edge_labels
        self.in_edge_labels = in_edge_labels
        
    def __str__(self):
        return f"DirectedExtension(out_edge_labels={self.out_edge_labels}, in_edge_labels={self.in_edge_labels})"
        
    def __copy__(self):
        """
        Create a copy of the DirectedExtension instance.
        """
        return DirectedExtension(self.out_edge_labels, self.in_edge_labels, self.location.copy())