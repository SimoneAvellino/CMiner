from parser import CMinerParser, gSpanParser
from comparator import *
from checker import Checker
#
db_path = "/Users/simoneavellino/Desktop/CMiner/test/Datasets/OntoUML-db/graphs.data"
solutions_path = "/Users/simoneavellino/Desktop/CMiner/test/solution/cminer.data"

checker = Checker(db_path, CMinerParser(solutions_path), matching_algorithm="VF2")

checker.isomorphic_solutions()

#
#
# gpsan_parser = CMinerParser("/Users/simoneavellino/Desktop/CMiner/test/solution/cminer.data")
#
# solutions = gpsan_parser.all_solutions()
# a = set()
# for s in solutions:
#     code = s.canonical_code()
#     print(code)
#     if code in a:
#         print("Duplicate")
#     else:
#         a.add(code)