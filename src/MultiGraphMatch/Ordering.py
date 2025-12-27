def compare_tuple_score(tuple_1, tuple_2):
    return tuple_1[0] > tuple_2[0] or tuple_1[0] == tuple_2[0] and tuple_1[1] > tuple_2[1]


class Ordering:

    def __init__(self, query_graph, compatibility_domain):
        self.compatibility_domain = compatibility_domain
        self.query_graph = query_graph
        self.order = []
        self.node_count = {}

    def score_cf_0(self, q_i, q_j):
        denominator = self.compatibility_domain.get_domain_cardinality((q_i, q_j))
        return (
                self.query_graph.tot_deg(q_i) *
                self.query_graph.tot_deg(q_j) *
                self.query_graph.jaccard_similarity(q_i, q_j) /
                self.compatibility_domain.get_domain_cardinality((q_i, q_j))
        ) if denominator != 0 else 0

    def score_cf_1(self, q_i, q_j, free_node):
        denominator = self.compatibility_domain.get_domain_cardinality((q_i, q_j))
        return (
                self.query_graph.tot_deg(free_node) *
                self.query_graph.jaccard_similarity(q_i, q_j) /
                denominator
        ) if denominator != 0 else 0

    def score_cf_2(self, q_i, q_j):
        denominator = self.compatibility_domain.get_domain_cardinality((q_i, q_j))
        return 1 / self.compatibility_domain.get_domain_cardinality((q_i, q_j)) if denominator != 0 else 0

    def compute_cf_score(self, q_i, q_j):
        return self.node_count[q_i] + self.node_count[q_j]

    def free_node(self, q_i, q_j):
        return q_i if self.node_count[q_i] == 0 else q_j

    def partial_candidate_score(self, q_i, q_j):
        cf = self.compute_cf_score(q_i, q_j)
        if cf == 0:
            return self.score_cf_0(q_i, q_j)
        elif cf == 1:
            return self.score_cf_1(q_i, q_j, self.free_node(q_i, q_j))
        else:
            return self.score_cf_2(q_i, q_j)

    def compute(self):
        # computing compatibility domain
        self.compatibility_domain.compute()
        # take edges (q_i, q_j) so that id(q_i) < id(q_j)
        # they are the key of the compatibility domain (See CompatibilityDomain/CompatibilityDomain.py)
        edges = list(self.compatibility_domain.get_all_query_edges())
        distinct_nodes = set(q for edge in edges for q in edge)
        # node_count[q] = 0 if q is not in the order, 1 otherwise
        # used to compute cf score for each edge
        # cf(q_i, q_j) = node_count[q_i] + node_count[q_j]
        self.node_count = {q: 0 for q in distinct_nodes}

        # while there are edges to consider
        while len(edges) > 0:
            # assume the first edge is the best candidate
            candidate = edges[0]
            # assume the best candidate has a cf score of 0 and a score of 0
            tuple_score = (0, 0)
            # NOTE: each edge has a score (cf, sc)
            # search for best candidate
            for partial_candidate in edges:
                q_i, q_j = partial_candidate
                # compute cf score and score for the partial candidate
                tuple_score_candidate = (self.compute_cf_score(q_i, q_j), self.partial_candidate_score(q_i, q_j))
                # if the partial candidate has a better score than the candidate
                if compare_tuple_score(tuple_score_candidate, tuple_score):
                    # update the candidate
                    candidate = partial_candidate
                    tuple_score = tuple_score_candidate

            # add all edges between q_i and q_j to the order
            for edge in self.query_graph.get_edges_consider_no_direction(candidate):
                self.order.append(edge)
            # remove the edge from the list of edges to consider
            edges.remove(candidate)
            # update node_count for q_i and q_j
            self.node_count[candidate[0]] = 1
            self.node_count[candidate[1]] = 1



    def length(self):
        return len(self.order)

    def get(self, index):
        return self.order[index]