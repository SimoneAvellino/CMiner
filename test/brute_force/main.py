import argparse
import subprocess
import sys
import os

import networkx as nx

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from parser import CMinerParser
from functions import *
# Initialize the parser
parser = argparse.ArgumentParser(description="Brute force testing")

parser.add_argument('-n', '--min_nodes', required=True, type=int, help='Minimum number of nodes')
parser.add_argument('-N', '--max_nodes', required=True, type=int, help='Maximum number of nodes')
parser.add_argument('-i', '--iteration', required=True, type=int, help='Test iteration times')

args = parser.parse_args()

min_nodes = args.min_nodes
max_nodes = args.max_nodes
iteration = args.iteration

for i in range(iteration):
    print(f"Iteration {i + 1}")

    num_nodes = random.randint(min_nodes, max_nodes)
    print(f"Generating graph with {num_nodes} nodes")
    G = generate_graph(num_nodes)

    print(f"Generating DB file with 3 graphs")
    db_path = f"/Users/simoneavellino/Desktop/CMiner/test/brute_force/db/graphs_{i}.data"
    with open(db_path, 'w') as f:
        for transaction_id in range(3):
            f.write(f"t # {transaction_id}\n{graph_string(G)}")

    cminer_script = f"CMiner {db_path} 2 -l {num_nodes} -u {num_nodes}"
    print(f"Running: {cminer_script}")
    result = subprocess.run(cminer_script, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    output = result.stdout
    # remove line that start with '->'
    output = "\n".join([line for line in output.split("\n") if not line.startswith("->")])
    print(output)
    print("Save result to file")
    result_path = f"/Users/simoneavellino/Desktop/CMiner/test/brute_force/results/graphs_{i}.txt"
    with open(result_path, 'w') as f:
        f.write(output)
    print("Check if there is only one solution and is isomorphic to the input graph")
    parser = CMinerParser(result_path)
    solutions = parser.all_solutions()
    if len(solutions) == 0:
        print("No solution found")
    elif len(solutions) > 1:
        print("More than one solution found")
        for s in solutions:
            print(s)
    else:
        print("Right")
        # solution = nx.DiGraph(solutions[0].graph)
        # print(type(G), type(solution))
        # print(f"Generated graph:\n{graph_string(G)}")
        # print(f"Solution graph:\n{graph_string(solution)}")
        # if nx.is_isomorphic(G, solution, node_match=node_match, edge_match=edge_match):
        #     print("Solution is isomorphic to the input graph")
        # else:
        #     print("Solution is not isomorphic to the input graph")

    print("\n")



