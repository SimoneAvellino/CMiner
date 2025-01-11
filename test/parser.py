"""
This file contains all the function to read the solutions from a file containing them.
"""
from Graph.Graph import MultiDiGraph
import networkx as nx

def node_match(n1, n2):
    labels1 = n1.get('labels', [])
    labels2 = n2.get('labels', [])
    return set(labels1) == set(labels2)


def edge_match(e1, e2):
    return e1[0].get('type', None) == e2[0].get('type', None)

class FileReader:
        """
        Class to read a file
        """
        def __init__(self, path):
            self.path = path

        def read(self) -> str:
            with open(self.path, "r") as f:
                return f.read()

class Solution:

    def __init__(self, graph: MultiDiGraph, support: int, _id: int):
        self.graph = graph
        self.support = support
        self.id = _id

    def get_support(self):
        return self.support

    def get_id(self):
        return self.id

    def __eq__(self, other):
        return nx.is_isomorphic(self.graph, other.graph, node_match=node_match, edge_match=edge_match)

    def canonical_code(self):
        return self.graph.canonical_code()


class CMinerSolution(Solution):

    def __init__(self, graph: MultiDiGraph, graph_ids: [str], frequencies: [int], _id: int, support: int):
        super().__init__(graph, support, _id)
        self.graph_ids = graph_ids
        self.frequencies = frequencies

    def __str__(self):
        out = ""
        out += f"t # {self.id}\n"
        for n in self.graph.nodes(data=True):
            labels = " ".join(n[1]['labels'])
            out += f"v {n[0]} {labels}\n"
        for e in self.graph.edges(data=True):
            out += f"e {e[0]} {e[1]} {e[2]['type']}\n"
        out += f"s {self.support}\n"
        out += "-\n"
        return out

    def __eq__(self, other):
        return nx.is_isomorphic(self.graph, other.graph, node_match=node_match, edge_match=edge_match)

class gSpanSolution(Solution):

        def __init__(self, graph: MultiDiGraph, support: int, _id: int):
            super().__init__(graph, support, _id)

        def __str__(self):
            out = ""
            out += f"t # {self.id}\n"
            for n in self.graph.nodes(data=True):
                out += f"v {n[0]} {n[1]['labels'][0]}\n"
            for e in self.graph.edges(data=True):
                out += f"e {e[0]} {e[1]} {e[2]['type']}\n"
            out += f"Support: {self.support}\n"
            out += "-----------------\n"
            return out

class Parser(FileReader):

    def __init__(self, solutions_path):
        super().__init__(solutions_path)

    def solutions_str(self, sep) -> [str]:
        """
        Read the solutions from the file and return them as a list of strings
        """
        strings = self.read().split(sep)
        if not strings[-1].__contains__("e"):
            strings.pop()
        return strings

    def all_solutions(self) -> ['Solution']:
        pass

    def algorithm(self) -> str:
        pass

class CMinerParser(Parser):
    """
    Class to read the CMiner solutions
    """
    def __init__(self, solutions_path):
        super().__init__(solutions_path)

    def algorithm(self):
        return "CMiner"

    @staticmethod
    def _parse_solution(solution: str) -> (MultiDiGraph, [int], int, [int]):
        graph = MultiDiGraph()
        nodes, edges, graphs_to_consider, frequencies = [], [], [], []
        solution_id = None
        support = 0

        for line in solution.strip().split("\n"):
            parts = line.split()
            if line.startswith("v"):
                node_id = int(parts[1])
                labels = parts[2:]
                nodes.append((int(node_id), labels))
            elif line.startswith("e"):
                src = int(parts[1])
                tgt = int(parts[2])
                labels = parts[3:]
                edges.append((int(src), int(tgt), labels))
            elif line.startswith("x"):
                # remove last character (")")
                line = line[:-1]
                # Properly split the graphs and extract names
                graphs = line[3:].split(") (")
                for g in graphs:
                    if g:
                        graph_name, frequency = g.split(", ")
                        graphs_to_consider.append(graph_name)
                        frequencies.append(int(frequency))
            elif line.startswith("t"):
                _, _, graph_id = parts
                solution_id = graph_id
            elif line.startswith("f"):
                _, *frequencies = parts
                frequencies = [int(f) for f in frequencies]
            elif line.startswith("s"):
                _, support = parts

        for node_id, labels in nodes:
            graph.add_node(node_id, labels=labels)
        for src, tgt, labels in edges:
            for label in labels:
                graph.add_edge(src, tgt, type=label)

        return graph, graphs_to_consider, frequencies, solution_id, int(support)

    def all_solutions(self) -> [CMinerSolution]:
        """
        Read all the solutions from the file
        """
        solutions_str = super().solutions_str("----------")
        return [CMinerSolution(*self._parse_solution(solution)) for solution in solutions_str]

class gSpanParser(Parser):
    """
    Class to read the gSpan solutions
    """
    def __init__(self, solutions_path):
        super().__init__(solutions_path)

    def algorithm(self):
        return "gSpan"

    @staticmethod
    def _parse_solution(solution: str) -> (MultiDiGraph, int, int):
        graph = MultiDiGraph()
        nodes, edges = [], []
        solution_id = None
        support = None

        for line in solution.strip().split("\n"):
            parts = line.split()
            if line.startswith("v"):
                _, node_id, label = parts
                nodes.append((int(node_id), label))
            elif line.startswith("e"):
                _, src, tgt, label = parts
                edges.append((int(src), int(tgt), label))
            elif line.startswith("t"):
                _, _, graph_id = parts
                solution_id = graph_id
            elif line.startswith("Support"):
                _, support = parts

        for node_id, label in nodes:
            graph.add_node(node_id, labels=[label])
        for src, tgt, label in edges:
            graph.add_edge(src, tgt, type=label)

        return graph, int(support), int(solution_id)

    def all_solutions(self) -> [gSpanSolution]:
        """
        Read all the solutions from the file
        """
        solutions_str = super().solutions_str("-----------------")
        return [gSpanSolution(*self._parse_solution(solution)) for solution in solutions_str]