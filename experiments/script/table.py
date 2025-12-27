import numpy as np
import pandas as pd

node_counts = []
edge_counts = []

with open('graph.data', 'r') as f:
    current_nodes = 0
    current_edges = 0
    first_graph = True
    for line in f:
        line = line.strip()
        if line.startswith('t '):
            if not first_graph:
                node_counts.append(current_nodes)
                edge_counts.append(current_edges)
            else:
                first_graph = False
            current_nodes = 0
            current_edges = 0
        elif line.startswith('v '):
            current_nodes += 1
        elif line.startswith('e '):
            current_edges += 1
    # append last graph
    node_counts.append(current_nodes)
    edge_counts.append(current_edges)

# Prepare metrics
metrics = {
    'Metric': [
        'Number of graphs',
        'Average nodes',
        'Average edges',
        'Max nodes',
        'Max edges',
        'Min nodes',
        'Min edges',
        'Std. dev. of nodes',
        'Std. dev. of edges'
    ],
    'Value': [
        len(node_counts),
        np.mean(node_counts),
        np.mean(edge_counts),
        np.max(node_counts),
        np.max(edge_counts),
        np.min(node_counts),
        np.min(edge_counts),
        np.std(node_counts, ddof=1),
        np.std(edge_counts, ddof=1)
    ]
}

df = pd.DataFrame(metrics)
print(df.head(10))
