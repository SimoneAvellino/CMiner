from abc import ABC, abstractmethod


class CompatibilityDomain(ABC):

    def __init__(self, query_bit_matrix, target_bit_matrix):
        """
        Constructor for the CompatibilityDomain class.

        :param query_bit_matrix:
        :param target_bit_matrix:
        """
        self.qbm = query_bit_matrix
        self.tbm = target_bit_matrix
        self.computed = False

    def _check_conditions(self, query_edge, target_edge):
        """
        Check the conditions for the compatibility domain.

        :param query_edge:
        :param target_edge:
        :return: boolean
        """
        # for each label we check the 4 conditions explained in the paper
        # query graph
        q_graph = self.qbm.get_graph()
        # target graph
        t_graph = self.tbm.get_graph()
        q_i = query_edge[0]
        q_j = query_edge[1]
        t_i = target_edge[0]
        t_j = target_edge[1]
        # the conditions must be satisfied for all labels
        labels = t_graph.get_all_edge_labels()
        # catch and return pattern
        # if one condition is not satisfied we return False
        for label in labels:
            if q_graph.t_out_deg(q_i, label) > t_graph.t_out_deg(t_i, label):
                return False
            if q_graph.t_in_deg(q_i, label) > t_graph.t_in_deg(t_i, label):
                return False
            if q_graph.t_out_deg(q_j, label) > t_graph.t_out_deg(t_j, label):
                return False
            if q_graph.t_in_deg(q_j, label) > t_graph.t_in_deg(t_j, label):
                return False
        # if all conditions are satisfied we return True
        return True

    @abstractmethod
    def compute(self):
        self.computed = True

    @abstractmethod
    def get_all_query_edges(self):
        """
        Return all the edges of the query graph with a domain.
        :return list of tuples
        """
        pass

    @abstractmethod
    def get_domain(self, edge):
        """
        Return the domain of the edge (q_i, q_j).

        :param edge: tuple
        :return list of tuples
        """
        pass

    @abstractmethod
    def get_domain_cardinality(self, edge):
        """
        Return the cardinality of the domain of the edge.

        :param edge: tuple
        :return: int
        """
        pass


class CompatibilityDomainWithDictionary(CompatibilityDomain):

    def __init__(self, query_bit_matrix, target_bit_matrix):
        """
        Constructor for the CompatibilityDomain class.

        :param query_bit_matrix:
        :param target_bit_matrix:
        """
        super().__init__(query_bit_matrix, target_bit_matrix)
        self.domain = {}

    def compute(self):
        if self.computed:
            return
        # set the flag to True to avoid re-computation
        self.computed = True
        # array of tuples [(query_edge_index, target_edge_index), ...]
        candidates = self.qbm.find_candidates(self.tbm)
        # the i-th matrix index corresponds to an edge in the graph
        # these two arrays contains this correspondence
        qbm_indices = self.qbm.get_matrix_indices()
        tbm_indices = self.tbm.get_matrix_indices()
        # construct the domain for each edge (q_i, q_j) in
        # the query graph so that idx(q_i) < idx(q_j)
        # NOTE: the indices of the QueryBitMatrix contains (q_i, q_j) and (q_j, q_i)
        #       but the QueryBitMatrix is constructed so that (q_i, q_j) with id(q_i) < id(q_j)
        #       is placed in even indices and (q_j, q_i) after (q_i, q_j) in odd indices.
        #       So we don't need to check if idx(q_i) < idx(q_j) because it is already guaranteed
        #       by the construction of the QueryBitMatrix.
        # EXAMPLE:
        #       place 0: (q1, q2) -> even index 0
        #       place 1: (q2, q1) -> odd index 1
        #       place 2: (q1, q3) -> even index 2
        #       place 3: (q3, q1) -> odd index 3
        for i in range(0, len(qbm_indices), 2):  # i = 0, i = 2, i = 4, ...
            self.domain[qbm_indices[i]] = []
        # compute the domain of each edge
        for candidate in candidates:
            query_edge_index = candidate[0]
            target_edge_index = candidate[1]

            if query_edge_index & 1 == 0:
                # if query_edge_index is even then the edge is (q_i, q_j)
                # associated with the even index is in the domain
                query_edge = qbm_indices[query_edge_index]
                target_edge = tbm_indices[target_edge_index]
            else:
                # if query_edge_index is odd then the edge is (q_i, q_j)
                # associated with the even index is not in the domain but
                # the edge (q_j, q_i) is in the domain (it is in the previous index that is even).
                query_edge = qbm_indices[query_edge_index - 1]
                # the target edge is swapped because the query edge is swapped
                target_edge = (tbm_indices[target_edge_index][1], tbm_indices[target_edge_index][0])

            # check the conditions for the compatibility domain
            if self._check_conditions(query_edge, target_edge):
                # if the conditions are satisfied we add the target edge to the domain
                self.domain[query_edge].append(target_edge)

    def get_all_query_edges(self):
        return list(self.domain.keys())

    def get_domain(self, edge):
        # domain is defined for (q_i, q_j) with id(q_i) < id(q_j)
        # so we need to check if the edge is (q_i, q_j) or (q_j, q_i)
        # if the edge is (q_j, q_i) we swap the nodes of the edge in the domain
        if edge[0] > edge[1]:
            return [(t_j, t_i) for t_i, t_j in self.domain[(edge[1], edge[0])]]
        # sort in decreasing order
        return self.domain[edge]

    def get_domain_cardinality(self, edge):
        """
        Return the cardinality of the domain of the edge.

        :param edge:
        :return: int
        """
        return len(self.domain[edge])
