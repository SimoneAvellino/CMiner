# This script is used to transform a database of multigraphs into a format that can be used by gSpan.

# When there are multiple edges (e1, e2, ...) between two nodes (n1, n2), this section of graph is
# represented as:
# n1 -- e1  -- n2
# n1 -- e2  -- n2
# n1 -- ... -- n2
# where -- is a new edge and e1, e2, ... are new nodes.

# NOTE: if the edge e is transformed into a node, then all edges like e are transformed into nodes too.
#       This is done in order to find all patterns in the graph.

from NetworkX.NetworkConfigurator import NetworkConfigurator
from NetworkX.NetworksLoading import NetworksLoading

DB_FILE = "../Datasets/Synthetic/test.data"
LABEL_NEW_EDGE = "_"
file_type = DB_FILE.split(".")[-1]

db = []

configurator = NetworkConfigurator(DB_FILE, file_type)
for name, network in NetworksLoading(file_type, configurator.config).Networks.items():
    db.append((network, name))
    
db_string = ""
    
for i, (network, name) in enumerate(db):
    
    # keep track of the edges labels that must be transformed into nodes
    edge_labels = set()
    
    for src, dst in set(network.edges()):
        if network.number_of_edges(src, dst) == 1:
            continue
        for key, data in network[src][dst].items():
            edge_labels.add(data['type'])
    
    new_node_id = int(max(network.nodes())) + 1
    # transform the edges into nodes
    for src, dst in set(network.edges()):
        # Make a static list of items to avoid modifying the dict during iteration
        for key, data in list(network[src][dst].items()):
            if data['type'] in edge_labels:
                network.add_node(new_node_id, labels=[data['type']])
                network.add_edge(src, new_node_id, type=LABEL_NEW_EDGE)
                network.add_edge(new_node_id, dst, type=LABEL_NEW_EDGE)
                network.remove_edge(src, dst, key)
                new_node_id += 1
                
    db_string += f"t # {i} {name}\n"
    for n, data in network.nodes(data=True):
        node_labels = " ".join(data['labels'])
        db_string += f"v {n} {node_labels}\n"
    for src, dst, data in network.edges(data=True):
        db_string += f"e {src} {dst} {data['type']}\n"
        
# save the transformed database to a file
new_file = DB_FILE[:len(DB_FILE) - len(file_type) - 1] + "_gspan" + "." + file_type
with open(new_file, "w") as f:
    f.write(db_string)