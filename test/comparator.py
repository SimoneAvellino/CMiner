from parser import *
import networkx as nx
from tabulate import tabulate




class Comparator:

    """
    This class is used to compare the solutions of two different mining algorithms.
    """
    def __init__(self, x_parser: Parser, y_parser: Parser):
        self.x_parser = x_parser
        self.y_parser = y_parser

    def solution_count(self) -> 'Comparator':
        """
        Print the number of solutions for each algorithm
        """
        print("---------------------------------")
        x_solution_num = len(self.x_parser.all_solutions())
        y_solution_num = len(self.y_parser.all_solutions())
        if x_solution_num == y_solution_num:
            print("Both algorithms have the same number of solutions")
            return self

        if x_solution_num > y_solution_num:
            print(self.x_parser.algorithm(), "has more solutions then", self.y_parser.algorithm())
        elif x_solution_num < y_solution_num:
            print(self.y_parser.algorithm(), "has more solutions then", self.x_parser.algorithm())

        print(self.x_parser.algorithm(), "has", x_solution_num, "solutions")
        print(self.y_parser.algorithm(), "has", y_solution_num, "solutions")

        return self

    def different_solutions(self, algorithm_x = True, algorithm_y = True) -> 'Comparator':
        """
        Print the solutions that are different between the two algorithms.

        Parameters:
        algorithm_x: bool
            If True, print the solutions that are in the first algorithm but not in the second one.
        algorithm_y: bool
            If True, print the solutions that are in the second algorithm but not in the first one.
        """
        print("---------------------------------")

        x_solutions = self.x_parser.all_solutions()
        y_solutions = self.y_parser.all_solutions()

        to_remove_x = []
        to_remove_y = []

        for x in x_solutions:
            for y in y_solutions:
                if x == y:
                    to_remove_x.append(x)
                    to_remove_y.append(y)

        for x in to_remove_x:
            if x in x_solutions:
                x_solutions.remove(x)

        for y in to_remove_y:
            if y in y_solutions:
                y_solutions.remove(y)

        if len(to_remove_x) == 0  and len(to_remove_y) == 0:
            # No solutions are different
            print("Both algorithms have the same solutions")
            return self

        def print_solutions(display, solutions, algorithm1, algorithm2):
            if not display:
                return
            print("That are" + ((" " + str(len(solutions))) if len(solutions) > 0 else "n't"), "solutions in", algorithm1, "that are not in", algorithm2, (":\n" if len(solutions) > 0 else ""))
            for solution in solutions:
                print(solution)

        print_solutions(algorithm_x, x_solutions, self.x_parser.algorithm(), self.y_parser.algorithm())
        print_solutions(algorithm_y, y_solutions, self.y_parser.algorithm(), self.x_parser.algorithm())

        return self

    def check_support(self):
        """
        Between same solution of the two algorithms, check if the support is the same.
        """
        print("---------------------------------")
        x_solutions = self.x_parser.all_solutions()
        y_solutions = self.y_parser.all_solutions()

        table_data = []

        for x in x_solutions:
            for y in y_solutions:
                if x == y:
                    if x.get_support() != y.get_support():
                        table_data.append([x.get_id(), x.get_support(), y.get_id(), y.get_support()])

        if table_data:
            print("NOTE: all solutions within the same row are isomorphic.\n")
            headers = [f"{self.x_parser.algorithm()} Solution ID X", "Support X", f"{self.y_parser.algorithm()} Solution ID Y", "Support Y"]
            # Stampa la tabella
            print(tabulate(table_data, headers=headers, tablefmt="grid"))
        else:
            print("No differences in support found.")

        return self