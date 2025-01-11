import networkx as nx
import random

def graph_string(G):
    out = ""
    for n in G.nodes:
        labels = " ".join(G.nodes()[n]['labels'])
        out += f"v {n} {labels}\n"
    for e in G.edges:
        label = G.edges()[e]['type']
        out += f"e {e[0]} {e[1]} {label}\n"
    return out

def generate_graph(num_nodes):
    G = nx.connected_watts_strogatz_graph(num_nodes, 2, 0.1)
    # set random labels
    for n in G.nodes:
        G.nodes()[n]['labels'] = [str(random.randint(0, 5))]
    for e in G.edges:
        G.edges()[e]['type'] = random.randint(0, 5)
    return nx.DiGraph(G)

def node_match(n1, n2):
    labels1 = n1.get('labels', [])
    labels2 = n2.get('labels', [])
    return set(labels1) == set(labels2)


def edge_match(e1, e2):
    return e1.get('type', None) == e2.get('type', None)