from NetworkX.NetworkConfigurator import NetworkConfigurator
from NetworkX.NetworksLoading import NetworksLoading
from CMiner.MultiGraphMatch import MultiGraphMatch
from Graph.Graph import MultiDiGraph
from CMiner.BitMatrix import TargetBitMatrixOptimized, BitMatrixStrategy2
from ZODB.config import databaseFromFile
from sympy.physics.units import frequency
import os
from parser import Parser
import networkx as nx
from tabulate import tabulate
from grandiso import find_motifs

def node_match(n1, n2):
    labels1 = n1.get('labels', [])
    labels2 = n2.get('labels', [])
    return set(labels1) == set(labels2)


def edge_match(e1, e2):
    return e1.get('type', None) == e2.get('type', None)

class GraphMatchingInfo:

    def __init__(self, frequency):
        self.frequency = frequency

    def get_frequency(self):
        return self.frequency

    def set_frequency(self, frequency):
        self.frequency = frequency

class DBGraph(MultiDiGraph):

    def __init__(self, graph, name, index):
        """
        Represents a graph in the database.
        """
        super().__init__(graph)
        self.name = name
        self.index = index


    def get_name(self):
        """
        Return the name of the graph
        """
        return self.name

    def get_index(self):
        """
        Return the index of the graph
        """
        return self.index

class Checker:

    """
    Class to check the solutions of a mining algorithm.
    """
    def __init__(self, db_path: str, solutions_parser: Parser, matching_algorithm: str = "MultiGraphMatch"):
        self.parser = solutions_parser
        self.db_path = db_path
        self.type_file = db_path.split(".")[-1]
        self.matching_algorithm = matching_algorithm
        self.db = []

    def _read_graphs_from_file(self):
        type_file = self.type_file
        configurator = NetworkConfigurator(self.db_path, type_file)
        for i, (name, network) in enumerate(NetworksLoading(type_file, configurator.config).Networks.items()):
            self.db.append(DBGraph(network, name, i))

    def _match(self, db_graph: DBGraph, solution: MultiDiGraph) -> GraphMatchingInfo:

        if self.matching_algorithm == "VF2":
            matcher = nx.algorithms.isomorphism.GraphMatcher(db_graph, solution, node_match=node_match, edge_match=edge_match)
            frequency = sum(1 for _ in matcher.subgraph_isomorphisms_iter())
        elif self.matching_algorithm == "MultiGraphMatch":
            matcher = MultiGraphMatch(db_graph, TargetBitMatrixOptimized(db_graph, BitMatrixStrategy2()))
            solutions = matcher.match(solution)
            frequency = len(solutions)
        elif self.matching_algorithm == "GrandIso":
            frequency = find_motifs(db_graph, solution.graph, count_only=True)
        else:
            raise ValueError("Invalid matching algorithm")

        return GraphMatchingInfo(frequency)


    def run(self):
        print(f"Checking {self.parser.algorithm()} solutions with {self.matching_algorithm} algorithm.")
        self._read_graphs_from_file()
        solutions = self.parser.all_solutions()

        table_data = []

        for i, solution in enumerate(solutions):
            # variable to store the number graphs containing the solution
            print(f"Checking solution having ID {solution.get_id()}: ", end="", flush=True)
            support = 0

            graph_names_where_not_found = []
            for db_graph in self.db:
                matching_info =  self._match(db_graph, solution.graph)
                if matching_info.frequency > 0:
                    graph_names_where_not_found.append(db_graph.get_name())
                    support += 1

            if support != solution.get_support():
                print(f"WRONG - Expected support: {solution.get_support()}, Found support: {support}")
                print(f"        Graphs where the solution was not found: {graph_names_where_not_found}")
                table_data.append([solution.get_id(), solution.get_support(), support])
                exit()
            else:
                print("CORRECT")

        # clear the screen
        os.system('cls' if os.name == 'nt' else 'clear')

        print(f"Checking {self.parser.algorithm()} solutions with {self.matching_algorithm} algorithm.")
        if table_data:
            headers = [f"Solution ID", "Expected support:", f"Found support:"]
            print(tabulate(table_data, headers=headers, tablefmt="grid"))
        else:
            print("All solutions are correct.")

        print("Accuracy:", (len(solutions) - len(table_data)) / len(solutions) * 100, "%")

    def isomorphic_solutions(self):
        """
        Check if there are isomorphic solutions.
        """
        print(f"Checking isomorphic solutions with {self.matching_algorithm} algorithm.")
        self._read_graphs_from_file()
        solutions = self.parser.all_solutions()
        for i in range(len(solutions)):
            for j in range(i+1, len(solutions)):
                if solutions[i] == solutions[j]:
                    print(f"Solution {solutions[i].get_id()} is isomorphic to solution {solutions[j].get_id()}")


