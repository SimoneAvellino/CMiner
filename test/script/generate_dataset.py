import networkx as nx
import random
import matplotlib.pyplot as plt

def generate_graphs(num_graphs, required_subgraphs=[], filename="graph.data"):
    """Generates connected graphs containing required subgraphs and saves them in structured text format."""
    
    if len(required_subgraphs) == 0:
        raise ValueError("At least one required subgraph must be provided.")
    
    number_of_nodes_of_patterns = sum(len(subgraph['nodes']) for subgraph in required_subgraphs)
    
    with open(filename, "w") as f:
        for graph_index in range(num_graphs):
            G = nx.MultiDiGraph()
            f.write(f"t # {graph_index} G{graph_index + 1}\n")
            
            node_reindex = {i: i for i in range(number_of_nodes_of_patterns)}
            
            for subgraph in required_subgraphs:
                
                times_per_graph = subgraph.get("times_per_graph", 1)
                
                number_of_nodes = len(subgraph['nodes']) # used to reindex nodes
                
                for _ in range(times_per_graph):
                    
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
                    edge_label = f"from_graph_{graph_index}" # We use the graph index as a label for the edge so that it's impossible to have new patterns between different graphs
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
        "nodes": [(0, ["p1_A", "p1_B", "p1_E"]), (1, ["p1_B", "p1_C"]), (2, ["p1_C","p1_F"]), (3, ["p1_D", "p1_C"]), (4,["p1_A", "p1_E"]), (5, ["p1_B", "p1_F"]), (6, ["p1_H", "p1_D"]), (7, ["p1_H", "p1_G"]) ], 
        "edges": [(0, 1, "p1_y"), (0, 2,  "p1_z"), (0, 3, "p1_x"), (1,4, "p1_z"), (4,1,  "p1_y"), (2,5, "p1_y"),(5,2,"p1_x"), (3,7, "p1_x"),(7,3, "p1_y"), (5,6,"p1_x"), (6,5,"p1_x")],
        "times_per_graph":4
    },
    {
        "nodes": [(0, ["p2_C","p2_A","p2_B"]), (1,["p2_D", "p2_C"]), (2,["p2_D", "p2_A"]), (3,["p2_B", "p2_C"]), (4,["p2_B", "p2_C"]), (5,["p2_F", "p2_A"])],
        "edges": [(0,1,"p2_x"),(1,0,"p2_z"),(5,0,"p2_x"), (0,3, "p2_x"),(3,0,"p2_x"), (1,2,"p2_y"), (3,2,"p2_y"), (4,5,"p2_x"), (4,3,"p2_z"), (3,4,"p2_z")],
        "times_per_graph": 6
    },
    {
        "nodes": [(0, ["p3_A", "p3_A"]), (1, ["p3_A", "p3_B", "p3_C"]), (2, ["p3_D", "p3_E"]), (3, ["p3_A", "p3_A"]), (4, ["p3_A", "p3_A"]), (5, ["p3_B", "p3_B"]), (6, ["p3_C", "p3_A"]), (7, ["p3_D", "p3_D"]), (8, ["p3_E"])],
        "edges": [(1, 0, "p3_x"), (1, 0, "p3_y"), (0, 2, "p3_x"), (2, 0, "p3_y"), (2, 0, "p3_y"), (1, 2, "p3_z"), (1, 3, "p3_y"), (1, 3, "p3_y"), (2, 4, "p3_z"), (4, 2, "p3_x"), (4, 2, "p3_x"), (4, 3, "p3_z"), (5, 3, "p3_x"), (3, 5, "p3_y"), (5, 6, "p3_x"), (6, 5, "p3_z"), (6, 3, "p3_z"), (6, 7, "p3_x"), (6, 7, "p3_x"), (7, 8, "p3_y"), (7, 8, "p3_z"), (4, 7, "p3_z"), (8, 4, "p3_z")],
        "times_per_graph": 8
    },
    {
        "nodes": [(0, ["p4_A"]), (1, ["p4_B"]), (2, ["p4_C", "p4_C"]), (3, ["p4_A", "p4_A"]), (4, ["p4_D", "p4_A", "p4_C"])],
        "edges": [(0, 1, "p4_x"), (1, 0, "p4_y"), (1, 2, "p4_x"), (1, 2, "p4_x"), (1, 2, "p4_y"), (2, 1, "p4_y"), (2, 1, "p4_y"), (2, 1, "p4_z"), (2, 3, "p4_x")],
        "times_per_graph": 10
    }
]

generate_graphs(num_graphs=50, required_subgraphs=required_subgraphs)