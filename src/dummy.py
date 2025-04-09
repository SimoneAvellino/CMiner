from CMiner.MultiGraphMatch import MultiGraphMatch
from Graph.Graph import MultiDiGraph

q = MultiDiGraph()

q.add_node(7, labels=["blue"])
q.add_node(8, labels=["yellow"])
q.add_edge(7, 8, type="white")

# q.add_node(1, labels=["blue"])
# q.add_node(2, labels=["yellow"])
# q.add_edge(1, 2, type="white")

print(q.canonical_code())

# t = MultiDiGraph()
# t.add_node(0, labels=["a"])
# t.add_node(1, labels=["b"])
# t.add_node(2, labels=["c"])
# t.add_edge(0, 1, type="e1")
# t.add_edge(1, 2, type="e2")
# t.add_edge(0, 2, type="e3")

# matcher = MultiGraphMatch(t)
# print(matcher.match(q))