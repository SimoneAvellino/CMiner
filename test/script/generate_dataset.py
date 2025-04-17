import networkx as nx
import random
import matplotlib.pyplot as plt

def generate_graphs(num_graphs, num_nodes, num_edges, required_subgraphs=None, labeled=True, multi_directed=False, filename="graph.data"):
    """Generates connected graphs containing required subgraphs and saves them in structured text format."""
    
    required_subgraphs = required_subgraphs or []
    with open(filename, "w") as f:
        for graph_index in range(num_graphs):
            G = nx.MultiDiGraph() if multi_directed else nx.Graph()
            f.write(f"t # {graph_index}\n")

            # Add nodes and edges from the required subgraphs
            for subgraph in required_subgraphs:
                for node, label in subgraph['nodes']:
                    G.add_node(node, label=label)
                    f.write(f"v {node} {label}\n")
                for u, v, label in subgraph['edges']:
                    G.add_edge(u, v, label=label)
                    f.write(f"e {u} {v} {label}\n")

            # Add remaining nodes
            existing_nodes = set(G.nodes)
            for i in range(1, num_nodes + 1):
                if i not in existing_nodes:
                    label = f"Node{i}" if labeled else "N"
                    G.add_node(i, label=label)
                    f.write(f"v {i} {label}\n")

            # Ensure connectivity using a spanning tree
            nodes = list(G.nodes)
            random.shuffle(nodes)
            for i in range(len(nodes) - 1):
                u, v = nodes[i], nodes[i + 1]
                if not G.has_edge(u, v):
                    edge_label = f"Edge_{u}_{v}" if labeled else "E"
                    G.add_edge(u, v, label=edge_label)
                    f.write(f"e {u} {v} {edge_label}\n")
                    if multi_directed and random.choice([True, False]):
                        G.add_edge(v, u, label=edge_label)
                        f.write(f"e {v} {u} {edge_label}\n")

            # Add extra edges up to the requested number
            edge_count = len(G.edges)
            while edge_count < num_edges:
                u, v = random.sample(list(G.nodes), 2)
                if not G.has_edge(u, v) or multi_directed:
                    edge_label = f"Edge_{u}_{v}" if labeled else "E"
                    G.add_edge(u, v, label=edge_label)
                    f.write(f"e {u} {v} {edge_label}\n")
                    edge_count += 1
                    if multi_directed and random.choice([True, False]):
                        G.add_edge(v, u, label=edge_label)
                        f.write(f"e {v} {u} {edge_label}\n")
                        edge_count += 1
    print(f"{num_graphs} connected graphs with required subgraphs saved to {filename}")

def load_graphs_from_file(filename="graph.data"):
    """Reads graph.data and reconstructs NetworkX graphs."""
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
    """Plots the loaded graphs using Matplotlib and NetworkX."""
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
    {"nodes": [(1, "A"), (2, "B"), (3, "C")], "edges": [(1, 2, "Ciao"), (2, 3, "Bla"), (3, 1, "Z")]},
    {"nodes": [(4, "D"), (5, "E")], "edges": [(4, 5, "W")]}
]

generate_graphs(num_graphs=100, num_nodes=10000, num_edges=10000, required_subgraphs=required_subgraphs, labeled=True, multi_directed=True)
# graphs = load_graphs_from_file()
# plot_graphs(graphs)

