import time
import argparse
from CMiner import CMiner
from Graph.Graph import MultiDiGraph

def parse_graph_str(str, is_directed):
    graph = MultiDiGraph()
    nodes, edges = [], []

    for line in str.strip().split("\n"):
        parts = line.split()
        if line.startswith("v"):
            node_id = parts[1]
            labels = parts[2:]
            nodes.append((int(node_id), labels))
        elif line.startswith("e"):
            src = parts[1]
            tgt = parts[2]
            labels = parts[3:]
            edges.append((int(src), int(tgt), labels))

    for node_id, labels in nodes:
        if len(labels) > 0:
            graph.add_node(node_id, labels=labels)
        else:
            graph.add_node(node_id)
    for src, tgt, labels in edges:
        if len(labels) > 0:
            for label in labels:
                graph.add_edge(src, tgt, type=label)
        else:
            graph.add_edge(src, tgt)
            
    return graph

def main_function():
    parser = argparse.ArgumentParser(description="CMiner algorithm")
    parser.add_argument('db_file', type=str, help="Path to graph db")
    parser.add_argument('support', type=float, help="Support")
    parser.add_argument('-l', '--min_nodes', type=int, help="Minimum number of nodes", default=1)
    parser.add_argument('-u', '--max_nodes', type=int, help="Maximum number of nodes", default=float('inf'))
    parser.add_argument('-n', '--num_nodes', type=int, help="Number of nodes", default=None)
    parser.add_argument('-m', '--show_mappings', type=int, help="Show pattern mappings", default=0)
    parser.add_argument('-o', '--output_path', type=str, help="Output file", default=None)
    parser.add_argument('-t', '--templates_path', type=str, help="Starting patterns file", default=None)
    parser.add_argument('-d', '--is_directed', type=int, help="Specify if the graph is directed", default=0)
    parser.add_argument('-f', '--with_frequencies', type=int, help="Show the relative frequencies of the pattern", default=0)
    parser.add_argument('-x', '--pattern_type', type=str, help="[all | maximum]", default='all')

    args = parser.parse_args()

    start_patterns = None
    if args.templates_path is not None:
        with open(args.templates_path, 'r') as f:
            patterns = f.read().split("----------")
            start_patterns = [parse_graph_str(pattern, args.is_directed) for pattern in patterns]

    if args.num_nodes is not None:
        args.min_nodes = args.num_nodes
        args.max_nodes = args.num_nodes

    miner = CMiner(
        args.db_file,
        support=args.support,
        min_nodes=args.min_nodes,
        max_nodes=args.max_nodes,
        show_mappings=args.show_mappings,
        output_path=args.output_path,
        start_patterns=start_patterns,
        is_directed=args.is_directed,
        with_frequencies=args.with_frequencies,
        pattern_type=args.pattern_type
    )

    start_time = time.time()

    try:
        miner.mine()
    except KeyboardInterrupt:
        print("\n-> Ctrl+C detected. Closing miner...")
        miner.close()
    finally:
        end_time = time.time()
        print(f"\n-> Execution time: {end_time - start_time} seconds")

    
    

