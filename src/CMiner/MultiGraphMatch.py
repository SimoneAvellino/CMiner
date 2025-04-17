from CMiner.BitMatrix import QueryBitMatrixOptimized, TargetBitMatrixOptimized, BitMatrixStrategy2
from CMiner.BreakingConditions import BreakingConditionsNodes, BreakingConditionsEdges
from CMiner.CompatibilityDomain import CompatibilityDomainWithDictionary
from CMiner.Ordering import Ordering
import ray
from sympy import false
from Graph.Graph import MultiDiGraph


class Mapping:

    def __init__(self, node_mapping=None, edge_mapping=None, extended_mapping: 'Mapping' = None):
        # extended_mapping is used to keep track of the previous mapping
        # when a new mapping that extends the previous one
        # it saves memory instead of copying the entire mapping
        self.extended_mapping = extended_mapping
        self.node_mapping = {} if node_mapping is None else node_mapping
        self.edge_mapping = {} if edge_mapping is None else edge_mapping
        
    def code(self): # FIX - it do not consider
        mapped_node_ids = list(self.nodes_mapping().values())
        return "".join(sorted([str(x) for x in mapped_node_ids]))
        
    def get_mapped_graph(self, target_graph):
        mapped_graph = MultiDiGraph()
        node_mapping = self._retrieve_node_mapping()
        edge_mapping = self._retrieve_edge_mapping()
        for pattern_node, target_node in node_mapping.items():
            mapped_graph.add_node(int(target_node), labels=target_graph.get_node_labels(target_node))
        for pattern_edge, target_edge in edge_mapping.items():
            for label in target_graph.get_edge_labels(target_edge[0], target_edge[1]):
                mapped_graph.add_edge(int(target_edge[0]), int(target_edge[1]), type=label)
        return mapped_graph
            

    def nodes_mapping(self) -> dict:
        node_mapping = {}
        if self.extended_mapping is not None:
            node_mapping.update(self.extended_mapping.nodes_mapping())
        node_mapping.update(self.node_mapping)
        return node_mapping

    def _retrieve_node_mapping(self):
        node_mapping = {}
        if self.extended_mapping is not None:
            node_mapping.update(self.extended_mapping._retrieve_node_mapping())
        node_mapping.update(self.node_mapping)
        return node_mapping

    def _retrieve_edge_mapping(self):
        edge_mapping = {}
        if self.extended_mapping is not None:
            edge_mapping.update(self.extended_mapping._retrieve_edge_mapping())
        edge_mapping.update(self.edge_mapping)
        return edge_mapping

    def node_pairs(self):
        """
        Returns the pairs of pattern nodes and target nodes
        """
        return self._retrieve_node_mapping().items()

    def node(self, pattern_node):
        """
        Returns the target node that corresponds to the pattern node
        """
        return self._retrieve_node_mapping()[pattern_node]

    def edge(self, pattern_edge):
        """
        Returns the target edge that corresponds to the pattern edge
        """
        return self._retrieve_edge_mapping()[pattern_edge]

    def nodes(self):
        """
        Returns the target nodes that correspond to the pattern nodes
        """
        return self._retrieve_node_mapping().values()

    def inverse(self, with_nodes=True, with_edges=True) -> 'Mapping':
        """
        Returns the inverse mapping
        """
        inverse_mapping = Mapping()
        if with_nodes:
            for k, v in self._retrieve_node_mapping().items():
                inverse_mapping.node_mapping[v] = k
        if with_edges:
            for k, v in self._retrieve_edge_mapping().items():
                inverse_mapping.edge_mapping[v] = k
        return inverse_mapping

    def get_target_edges(self) -> list:
        if self.extended_mapping is not None:
            edges = list(self.extended_mapping.get_target_edges())
            edges.extend(self.edge_mapping.values())
            return edges
        return list(self.edge_mapping.values())

    def get_pattern_nodes(self):
        return self.node_mapping.keys()

    def get_pattern_edges(self):
        return self.edge_mapping.keys()

    def __str__(self):
        return f"({{{self.get_node_mapping_str()}}}, {{{self.get_edge_mapping_str()}}})"

    def get_node_mapping_str(self):
        node_mapping_str = ""
        if self.extended_mapping is not None:
            node_mapping_str += self.extended_mapping.get_node_mapping_str() + ", "
        node_mapping_str += " ".join([f"{k}->{v}" for k, v in self.node_mapping.items()])
        return node_mapping_str

    def get_edge_mapping_str(self):
        if len(self.edge_mapping) == 0:
            return ""
        edge_mapping_str = ""
        if self.extended_mapping is not None:
            previous_edge_mapping_str = self.extended_mapping.get_edge_mapping_str()
            edge_mapping_str += previous_edge_mapping_str + (", " if len(previous_edge_mapping_str) > 0 else "")
        edge_mapping_str += " ".join([f"{k}->{v}" for k, v in self.edge_mapping.items()])
        return edge_mapping_str

    def set_edge(self, pattern_edge, target_edge):
        self.edge_mapping[pattern_edge] = target_edge

    def remove_edge(self, pattern_edge):
        if pattern_edge in self.edge_mapping:
            del self.edge_mapping[pattern_edge]
        if self.extended_mapping is not None:
            self.extended_mapping.remove_edge(pattern_edge)
            
    def reset_edge_mapping(self, edge_mapping = None):
        """
        Reset the keys of the edge mapping so 
        that for each edge the keys start from 0
        """
        edge_mapping = edge_mapping if edge_mapping is not None else self._retrieve_edge_mapping()
        for edge in self.edge_mapping.keys():
            src, dst, key = edge

class MultiGraphMatch:

    def __init__(self, target, target_bit_matrix=None):

        self.target = target
        if target_bit_matrix is None:
            self.tbm = TargetBitMatrixOptimized(self.target, BitMatrixStrategy2())
        else:
            self.tbm = target_bit_matrix
        # all of this attributes are initialized in the match method
        # so that the class can be reused
        self.query = None
        self.qbm = None
        self.node_mapping_function = None
        self.edge_mapping_function = None
        self.cand = None
        self.cand_index = None
        self.br_cond_node = None
        self.br_cond_edge = None
        self.domain = None
        self.ordering = None
        self.solutions = []
        self.f = None
        self.g = None

    def _init_matching(self,
                       query,
                       query_bit_matrix=None,
                       compatibility_domain=None,
                       ordering=None,
                       breaking_conditions_nodes=None,
                       breaking_conditions_edges=None
                       ):
        self.solutions = []
        self.f = {node: None for node in query.nodes()}
        self.g = {edge: None for edge in query.get_all_edges()}
        self.cand = {edge: [] for edge in self.query.get_all_edges()}
        self.cand_index = {edge: 0 for edge in self.query.get_all_edges()}
        if query_bit_matrix is None:
            self.qbm = QueryBitMatrixOptimized(query, BitMatrixStrategy2())
        else:
            self.qbm = query_bit_matrix
        if compatibility_domain is None:
            self.domain = CompatibilityDomainWithDictionary(self.qbm, self.tbm)
        else:
            self.domain = compatibility_domain
        if ordering is None:
            self.ordering = Ordering(self.query, self.domain)
        else:
            self.ordering = ordering
        if breaking_conditions_nodes is None:
            self.br_cond_node = BreakingConditionsNodes(self.query, self.f)
        else:
            self.br_cond_node = breaking_conditions_nodes
        if breaking_conditions_edges is None:
            self.br_cond_edge = BreakingConditionsEdges(self.query, self.g)
        else:
            self.br_cond_edge = breaking_conditions_edges

    def _match_1node_query(self) -> list[Mapping]:
        mappings = []
        q_node_id = list(self.query.nodes())[0]
        q_node_labels = self.query.get_node_labels(q_node_id)
        for node in self.target.nodes():
            t_node_labels = self.target.get_node_labels(node)
            if all(label in t_node_labels for label in q_node_labels):
                mappings.append(Mapping(node_mapping={q_node_id: node}))

        return mappings

    def match(self,
              query,
              query_bit_matrix=None,
              compatibility_domain=None,
              ordering=None,
              breaking_conditions_nodes=None,
              breaking_conditions_edges=None):
        
        # check if all the labels that have the query are also inside the target
        query_node_labels = query.get_all_node_labels()
        target_node_labels = self.target.get_all_node_labels()
        query_edge_labels = query.get_all_edge_labels()
        target_edge_labels = self.target.get_all_edge_labels()
        if not all(label in target_node_labels for label in query_node_labels):
            return []
        if not all(label in target_edge_labels for label in query_edge_labels):
            return []

        self.query = query

        # if the query has no edge (so it is a single node) the matching is trivial
        if len(query.get_all_edges()) == 0:
            if len(query.nodes()) != 1:
                raise ValueError("The query has no edges but it has more than one node")
            return self._match_1node_query()

        self._init_matching(query,
                            query_bit_matrix,
                            compatibility_domain,
                            ordering,
                            breaking_conditions_nodes,
                            breaking_conditions_edges)

        self.ordering.compute()
        forceBack = False
        i = 0
        q_i, q_j, q_key = query_edge = self.ordering.get(i)
        self._find_candidates(query_edge)
        while i >= 0:
            if forceBack or self.cand_index[query_edge] >= len(self.cand[query_edge]):
                forceBack = False
                if self.g[query_edge] is None:
                    i -= 1
                if i < 0:
                    # no more solutions
                    break
                q_i, q_j, q_key = query_edge = self.ordering.get(i)

                # backtracking
                # RESET THE MAPPING OF THE QUERY EDGE
                self.g[query_edge] = None
                # RESET THE MAPPING OF THE QUERY NODES
                # reset f[q_i] if there are no mapped edges in the query that have q_i as source/destination
                if all(self.g[q_e] is None for q_e in self.query.get_all_edges() if q_e[0] == q_i or q_e[1] == q_i):
                    self.f[q_i] = None
                # reset f[q_j] if there are no mapped edges in the query that have q_j as source/destination
                if all(self.g[q_e] is None for q_e in self.query.get_all_edges() if q_e[0] == q_j or q_e[1] == q_j):
                    self.f[q_j] = None
                self.cand_index[query_edge] += 1
            else:
                # CHECK IF THE NEXT CANDIDATE TARGET EDGE IS COMPATIBLE WITH THE QUERY EDGE
                # extract the target edge
                t_i, t_j, key = target_edge = self.cand[query_edge][self.cand_index[query_edge]]
                # check if the target edge has not been already mapped to another query edge
                if all(self.g[q_e] != target_edge for q_e in self.query.get_all_edges() if self.g[q_e] is not None):
                    # if the mapping for the source is not already set, set it
                    if self.f[q_i] is None:
                        if t_i in self.f.values():
                            self.cand_index[query_edge] += 1
                            continue
                        self.f[q_i] = t_i
                    # if the mapping for the destination is not already set, set it
                    if self.f[q_j] is None:
                        if t_j in self.f.values():
                            self.cand_index[query_edge] += 1
                            continue
                        self.f[q_j] = t_j
                    # set the mapping for the query edge
                    self.g[query_edge] = target_edge
                    # if the edge is the last in the ordering, the solution is found
                    if i == len(self.query.get_all_edges()) - 1:
                        forceBack = True
                        # SOLUTION FOUND
                        # save the mapping
                        self.solutions.append(Mapping(node_mapping=self.f.copy(), edge_mapping=self.g.copy()))
                    else:
                        # shift the index to the next query edge in the ordering
                        i += 1
                        # get the next query edge
                        q_i, q_j, q_key = query_edge = self.ordering.get(i)
                        # find the candidates for the next query edge
                        self._find_candidates(query_edge)
                        # reset the index for the candidates of the next query edge
                        self.cand_index[query_edge] = 0
                else:
                    # shift the index to the next candidate
                    self.cand_index[query_edge] += 1

        return self.solutions



    def _find_candidates(self, query_edge):
        q_i, q_j, query_key = query_edge
        self.cand[query_edge] = []
        if self.f[q_i] is None and self.f[q_j] is None:
            for t_i, t_j in self.domain.get_domain((q_i, q_j)):
                # each target edge is a tuple (source, target, edge_id)
                for target_key in self.target.edges_keys((t_i, t_j)):
                    target_edge = (t_i, t_j, target_key)
                    if (
                            (
                                    # if query edge label is not specified it means that any edge label is accepted
                                    not self.query.edge_has_label(query_edge) or
                                    # check if the edge labels are the same
                                    self.target.get_edge_label(target_edge) == self.query.get_edge_label(query_edge)
                            ) and
                            # check if the target edge source node contains the same attributes as the query edge source node
                            self.target.node_contains_attributes(t_i, self.query.get_node_attributes(q_i)) and
                            # check if the target edge destination node contains the same attributes as the query edge destination node
                            self.target.node_contains_attributes(t_j, self.query.get_node_attributes(q_j)) and
                            # check if the target edge contains the same attributes as the query edge
                            self.target.edge_contains_attributes(target_edge,
                                                                 self.query.get_edge_attributes(query_edge))
                    ):
                        self.cand[query_edge].append(target_edge)
        elif self.f[q_i] is not None and self.f[q_j] is not None:
            for target_key in self.target.edges_keys((self.f[q_i], self.f[q_j])):
                target_edge = (self.f[q_i], self.f[q_j], target_key)
                if (
                        (
                                # if query edge label is not specified it means that any edge label is accepted
                                not self.query.edge_has_label(query_edge) or
                                # check if the edge labels are the same
                                self.target.get_edge_label(target_edge) == self.query.get_edge_label(query_edge)
                        ) and
                        self.target.edge_contains_attributes(target_edge, self.query.get_edge_attributes(query_edge))
                ):
                    if self.br_cond_edge.check(query_edge, target_edge):
                        self.cand[query_edge].append(target_edge)
        elif self.f[q_i] is not None:
            for t_i, t_j in self.domain.get_domain((q_i, q_j)):
                if t_i == self.f[q_i]:
                    for target_key in self.target.edges_keys((t_i, t_j)):
                        target_edge = (t_i, t_j, target_key)
                        if (
                                (
                                        # if query edge label is not specified it means that any edge label is accepted
                                        not self.query.edge_has_label(query_edge) or
                                        # check if the edge labels are the same
                                        self.target.get_edge_label(target_edge) == self.query.get_edge_label(query_edge)
                                ) and
                                self.target.node_contains_attributes(t_j, self.query.get_node_attributes(q_j)) and
                                self.target.edge_contains_attributes(target_edge,
                                                                     self.query.get_edge_attributes(query_edge))
                        ):
                            if self.br_cond_node.check(q_j, t_j):
                                self.cand[query_edge].append(target_edge)
        else:
            for t_i, t_j in self.domain.get_domain((q_i, q_j)):
                if t_j == self.f[q_j]:
                    for target_key in self.target.edges_keys((t_i, t_j)):
                        target_edge = (t_i, t_j, target_key)

                        if (
                                (
                                    # if query edge label is not specified it means that any edge label is accepted
                                    not self.query.edge_has_label(query_edge) or
                                    self.target.get_edge_label(target_edge) == self.query.get_edge_label(query_edge)
                                ) and
                                self.target.node_contains_attributes(t_i, self.query.get_node_attributes(q_i)) and
                                self.target.edge_contains_attributes(target_edge,
                                                                     self.query.get_edge_attributes(query_edge))
                        ):
                            if self.br_cond_node.check(q_i, t_i):
                                self.cand[query_edge].append(target_edge)

@ray.remote
def match_parallel_worker(target, query, worker_id):

    matcher = MultiGraphMatch(target, query)
    matcher.match(query)
    solutions = matcher.solutions()
    return solutions