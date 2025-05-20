import networkx as nx
import random
import matplotlib.pyplot as plt

def generate_graphs(num_graphs, required_subgraphs=[], filename="graph.data"):
    """Generates connected graphs containing required subgraphs and saves them in structured text format."""
    
    if len(required_subgraphs) == 0:
        raise ValueError("At least one required subgraph must be provided.")
    
    number_of_nodes_of_patterns = sum(len(subgraph['nodes']) for subgraph in required_subgraphs)
    
    with open(filename, "w") as f:
        
        for pattern_num, subgraph in enumerate(required_subgraphs):
            
            times_per_graph = subgraph.get("times_per_graph", [1] * num_graphs)
            
            print(f"The pattern {pattern_num + 1} will be added to the graphs {sum(times_per_graph)} times.")
            
            for graph_index in range(num_graphs):
                G = nx.MultiDiGraph()
                f.write(f"t # {graph_index} G{graph_index + 1}\n")
                
                node_reindex = {i: i for i in range(number_of_nodes_of_patterns)}
                
                number_of_nodes = len(subgraph['nodes']) # used to reindex nodes
                
                for _ in range(times_per_graph[graph_index]):
                    
                    for node, label in subgraph['nodes']:
                        G.add_node(node_reindex[node], label=label)
                        labels_str = " ".join(label)
                        f.write(f"v {node_reindex[node]} {labels_str}\n")
                    for u, v, label in subgraph['edges']:
                        G.add_edge(node_reindex[u], node_reindex[v], label=label)
                        f.write(f"e {node_reindex[u]} {node_reindex[v]} {label}\n")
                        
                
                    # increment the node_reindex mapping
                    for node in node_reindex:
                        node_reindex[node] += number_of_nodes

            # Ensure connectivity using a spanning tree
            nodes = list(G.nodes)
            random.shuffle(nodes)
            for i in range(len(nodes) - 1):
                u, v = nodes[i], nodes[i + 1]
                if not G.has_edge(u, v):
                    edge_label = f"from_graph_{graph_index}_{i}" # We use the graph index as a label for the edge so that it's impossible to have new patterns between different graphs
                    G.add_edge(u, v, label=edge_label)
                    f.write(f"e {u} {v} {edge_label}\n")

    print(f"{num_graphs} connected graphs with required subgraphs saved to {filename}")

def load_graphs_from_file(filename="graph.data"):
    """Reads graph.data and reconstructs Networkx graphs."""
    graphs = []
    G = None
    is_multi_directed = False
    with open(filename, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == "t":
                if G:
                    graphs.append(G)
                G = nx.MultiDiGraph() if is_multi_directed else nx.Graph()
            elif parts[0] == "v":
                node_id = int(parts[1])
                label = parts[2] if len(parts) > 2 else "N"
                G.add_node(node_id, label=label)
            elif parts[0] == "e":
                u, v = int(parts[1]), int(parts[2])
                label = parts[3] if len(parts) > 3 else "E"
                G.add_edge(u, v, label=label)
                if G.has_edge(v, u):
                    is_multi_directed = True
        if G:
            graphs.append(G)
    return graphs

def plot_graphs(graphs):
    """Plots the loaded graphs using Matplotlib and Networkx."""
    for i, G in enumerate(graphs):
        plt.figure(figsize=(6, 6))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)
        node_labels = nx.get_node_attributes(G, "label")
        if node_labels:
            nx.draw_networkx_labels(G, pos, labels=node_labels)
        edge_labels = nx.get_edge_attributes(G, "label")
        if edge_labels:
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        plt.title(f"Graph {i + 1}")
        plt.show()

# Example subgraph constraints
required_subgraphs = [
    {
        "nodes": [(0,["Book"]), (1,["Author"]), (2, ["entity","BookItem"]), (3,["entity","Account"]), (4,["Catalog"]), (5,["Library"]), (6, ["interface","Search"]), (7,["interface","Manage"]), (8,["Librarian"])],
        "edges": [(1,0,"wrote"), (1,0, "1*-1*"), (2,0, "Generalization"), (3,2,"borrowed"), (3,2,"reserved"), (4,2,"records"), (2,5,"aggregation"), (4,5,"composition"), (5,4,"composition"), (4,6,"realization"), (4,7, "realization"), (8,7,"dependecy"), (8,7,"users"), (8,6,"dependency"), (8,6,"users")],
        "times_per_graph": [150, 10, 5, 10, 50, 10, 20, 10, 10, 10, 10, 10 , 5, 3, 2, 10, 10, 15, 10, 10]
    }
]

generate_graphs(num_graphs=20, required_subgraphs=required_subgraphs, filename="/path/to/graph.data")