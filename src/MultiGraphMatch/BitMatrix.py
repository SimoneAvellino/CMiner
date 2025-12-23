from abc import ABC, abstractmethod
from bitarray import bitarray
from bitstring import BitArray


# TO-DO: Find an optimal way to discard edges already
#        computed in the compute method [O(log n)?]
# TO-DO: Find candidates in a different way.
#        Represent BitMatrix as a BTree?
# TO-DO: Keep lazy computing?
# DONE:  Write better the code for showing the matrix
# DONE:  Optimization on the QueryBitMatrix computation.
#        After computing bitmap for edge (a, b), bitmap
#        For edge (b, a) can the computed with two
#        permutaion on the bitmap for (a, b)


class BitMatrix(ABC):
    """Abstract class for describing a BitMatrix

    :param graph               Graph in which the BitMatrix is computed
    :type  graph               Multigraph
    :param bit_matrix_strategy Datastructure for the rows of the matrix
    :type  bit_matrix_strategy BitMatrixStrategy
    """

    def __init__(self, graph, bit_matrix_strategy):
        bit_matrix_strategy.set_graph(graph)
        self.bit_matrix_strategy = bit_matrix_strategy
        self.graph = graph
        # BitMatrix (list of bitmaps)
        self.matrix = []
        # List that keeps trace of the edge associeted to each row
        self.matrix_indices = []
        # To implement lazy computing on other methods
        self.computed = False

    @abstractmethod
    def compute(self):
        """Compute the BitMatrix

        Method that is defined in the Concrete instance depending on the
        type of the BitMatrix (QueryBitMatrix, TargetBitMatrix)

        :rtype void
        """
        self.computed = True
        self.graph.reset_memoization()
        pass

    def _lazy_computing(self):
        """Method to compute the BitMatrix if it has not been calculated

        :rtype void
        """
        not self.is_computed() and self.compute()
        self.computed = True

    def get_matrix(self):
        """Retrieve the BitMatrix

        It returns only the list of bitmaps, not the edges
        associeted to all edges

        :rtype list of bitmaps

        See also
            :func get_matrix_indices()
        """
        self._lazy_computing()
        return self.matrix

    def get_graph(self):
        return self.graph

    def get_matrix_indices(self):
        """Retrieve the BitMatrix

        :rtype list of edges
        """
        self._lazy_computing()
        return self.matrix_indices

    def is_computed(self):
        return self.computed

    def split_bitmap_row(self, row_num):
        self._lazy_computing()
        row = self.matrix[row_num]
        return self.split_bitmap(row)

    def split_bitmap(self, bitmap):
        """Return each part of the bitmap associated to an edge as a string

        each bitmap is made up from four parts:
            L_first, T_in, T_out, L_second
            L_first and L_second have num_node_labels elements
            T_in    and T_out have num_edge_labels elements

        rtype: list of bitmaps
        """
        num_node_labels = len(self.graph.get_all_node_labels())
        num_edge_labels = len(self.graph.get_all_edge_labels())
        row_parts = [
            bitmap[:num_node_labels],
            bitmap[num_node_labels : num_node_labels + num_edge_labels],
            bitmap[
                num_node_labels
                + num_edge_labels : num_node_labels
                + 2 * num_edge_labels
            ],
            bitmap[num_node_labels + 2 * num_edge_labels :],
        ]
        return row_parts

    @abstractmethod
    def L_first_distinct_values(self):
        pass

    @abstractmethod
    def L_second_distinct_values(self):
        pass

    @abstractmethod
    def T_in_distinct_values(self):
        pass

    @abstractmethod
    def T_out_distinct_values(self):
        pass


class TargetBitMatrix(BitMatrix):
    """Concrete class for describing a BitMatrix for a Target Graph

    A Target Graph BitMatrix is a matrix in which the i-th row if a
    bitmap associeted to the edge (a, b) so that the id(a) < id(b).
    We we impose this constraint to keep memory costs low.

    :param graph               Graph in which the BitMatrix is computed
    :type  graph               Multigraph
    :param bit_matrix_strategy Datastructure for the rows of the matrix
    :type  bit_matrix_strategy BitMatrixStrategy
    """

    def __init__(self, graph, bit_matrix_strategy):
        super().__init__(graph, bit_matrix_strategy)

    def compute(self):
        """Compute the TargetBitMatrix
        :rtype void

        See also:
            :func get()
        """
        super().compute()
        # Extracting edges
        # Using a set to have single occurrence of each edge
        edges = set(self.graph.edges())
        for edge in edges:  # edge = (source_id, destination_id)
            # NOTE: there could be this edge e = (a, b)
            # a -> b : id(b) < id(a)
            # checking only id(a) < id(b) and not id(b) < id(a)
            # would bring to a lost of the edge (b, a) in the matrix
            # so we check both conditions
            edge_to_compute = ()
            if edge[0] < edge[1]:  # condition 1
                edge_to_compute = edge
            elif edge[1] < edge[0]:  # condition 2
                edge_to_compute = (edge[1], edge[0])
            # there could exist an edge (a, b) an (b, a)
            # with the previews if we would consider (b, a) twice
            # so we check if the edges is already been computed
            if edge_to_compute not in self.matrix_indices:  # FIND BETTER METHOD
                # add the bitmap associeted to the edge to the matrix
                self.matrix.append(
                    self.bit_matrix_strategy.compute_row(edge_to_compute)
                )
                # saving the edge associeted to the bitmap
                self.matrix_indices.append(edge_to_compute)


class TargetBitMatrixOptimized(TargetBitMatrix):

    def __init__(self, graph, bit_matrix_strategy):
        super().__init__(graph, bit_matrix_strategy)
        self.matrix = [{} for _ in range(4)]

    def split_bitmap_row(self, row_num):
        """Split the bitmap of the i-th row of the matrix

        :param row_num:
        :return:
        """

        # construct the bitmap of the i-th row.
        # In the optimized version there is no matrix, but a list of dictionaries
        # each dictionary is associated to a part of the bitmap. The key of the dictionary
        # is the bitmap as a string and the value is the list of indices associated to the bitmap.
        # We take all dictionary and we retrieve each bitmap part associated to the i-th row.
        self._lazy_computing()
        bitmap = ""
        for i, part in enumerate(self.matrix):
            for key, value in part.items():
                if row_num in value:
                    bitmap += key
                    break

        return self.split_bitmap(self.bit_matrix_strategy.str_to_bitmap(bitmap))

    def compute(self):
        """Compute the TargetBitMatrix
        :rtype void

        See also:
            :func get()
        """
        self.computed = True
        self.graph.reset_memoization()
        # Extracting edges
        # Using a set to have single occurrence of each edge
        edges = set(self.graph.edges())
        for edge in edges:  # edge = (source_id, destination_id)
            # NOTE: there could be this edge e = (a, b)
            # a -> b : id(b) < id(a)
            # checking only id(a) < id(b) and not id(b) < id(a)
            # would bring to a lost of the edge (b, a) in the matrix
            # so we check both conditions
            edge_to_compute = ()
            if edge[0] < edge[1]:  # condition 1
                edge_to_compute = edge
            elif edge[1] < edge[0]:  # condition 2
                edge_to_compute = (edge[1], edge[0])
            else:
                continue
            # there could exist an edge (a, b) and (b, a)
            # so we check if the edges is already been computed
            if edge_to_compute not in self.matrix_indices:  # FIND BETTER METHOD
                bitmap = self.bit_matrix_strategy.compute_row(edge_to_compute)
                bitmap_parts = self.split_bitmap(bitmap)
                edge_index = len(self.matrix_indices)  # take the index of the edge
                for i, part in enumerate(bitmap_parts):
                    # take the key for the i-th dictionary
                    str_part = "".join("1" if bit else "0" for bit in part)
                    # save the index of the edge in the i-th dictionary associated to the i-th part of the bitmap
                    self.matrix[i][str_part] = self.matrix[i].get(str_part, []) + [
                        edge_index
                    ]
                # saving the edge associated to the bitmap
                self.matrix_indices.append(edge_to_compute)

    def L_first_distinct_values(self):
        return list(self.matrix[0].keys())

    def L_second_distinct_values(self):
        return list(self.matrix[3].keys())

    def T_in_distinct_values(self):
        return list(self.matrix[1].keys())

    def T_out_distinct_values(self):
        return list(self.matrix[2].keys())


class QueryBitMatrix(BitMatrix):
    """Concrete class for describing a BitMatrix for a Query Graph

    A Query Graph BitMatrix is a matrix in which each row associeted to
    the edge (a, b) with id(a) < id(b) also has the row associeted to
    the edge (b, a). We impose this constraint not to loose any solutions.
    We can "afford" this memory cost because the Query BitMatrix is
    usually small.

    :param graph               Graph in which the BitMatrix is computed
    :type  graph               Multigraph
    :param bit_matrix_strategy Datastructure for the rows of the matrix
    :type  bit_matrix_strategy BitMatrixStrategy
    """

    def __init__(self, graph, bit_matrix_strategy):
        super().__init__(graph, bit_matrix_strategy)

    def compute(self):
        """Compute the QueryBitMatrix

        :rtype void

        See also:
            :func get()
        """
        self.computed = True
        super().compute()
        # The comments are the same as the TargetBitMatrix compute
        # method, the small change is at the end.
        edges = set(self.graph.edges())
        for edge in edges:
            if edge[0] < edge[1]:
                edge_to_compute = edge
            elif edge[1] < edge[0]:
                edge_to_compute = (edge[1], edge[0])
            else:
                # This will be executed only if the edge source and destination
                # are the same. To comprehend why check how the query graph is
                # constructed
                continue
            if edge_to_compute not in self.matrix_indices:
                # edge (a, b) bitmap
                self.matrix.append(
                    self.bit_matrix_strategy.compute_row(edge_to_compute)
                )
                self.matrix_indices.append(edge_to_compute)
                # edge (b, a) bitmap
                # do not compute again, we just swap L_first with L_second and T_in with T_out
                i = len(self.matrix) - 1  # index of the bitmap computed before
                row_parts = self.split_bitmap_row(i)  # taking the parts of the bitmap
                self.matrix.append(
                    row_parts[3] + row_parts[2] + row_parts[1] + row_parts[0]
                )
                self.matrix_indices.append((edge_to_compute[1], edge_to_compute[0]))

    def _adapt_query_to_target(self, target_graph):
        """Adding the correct labels to perform the query

        The query is constructed on the query graph but it couldn't
        not have all labels of the target. To handle this situation
        we add a dummy node with all labels that are in the target
        graph. In this way we "force" the query to have all labels
        that have the target. The same thing is done with the edges
        adding loops to the dummy node with all edge labels from the
        target graph.

        Example:    Query  labels: x, y
                    Target labels: x, y, z
                    Add a dummy node with labels x, y, z in the query

        :rtype void
        """
        self.graph.add_node("dummy", labels=target_graph.get_all_node_labels())
        for label in target_graph.get_all_edge_labels():
            self.graph.add_edge("dummy", "dummy", type=label)

    def _undo_adapt_query_to_target(self):
        self.graph.remove_nodes(["dummy"])
        self.graph.remove_edges(self.graph.edges("dummy"))

    def find_candidates(self, target_bitmatrix):
        """Find the candidate target edges

        This method cycle all query bitmaps (bq) associeted to the edge (a, b)
        and execute an AND operation for each target bitmap (bt) of the edge (x, y).
        If bq & bt == bq then (a, b) and (x, y) could be compatible, thus the
        tuple candidate ( (a, b), (x, y) ) is added to the list of candidates.

        NOTE:   The node a can be mapped to x while b can be mapped to y.

        :rtype list of candidates
        """
        self._adapt_query_to_target(target_bitmatrix.get_graph())
        # lazy computing
        self._lazy_computing()
        target_bitmatrix._lazy_computing()
        candidates = []
        bmt = target_bitmatrix.get_matrix()
        bmt_indices = target_bitmatrix.get_matrix_indices()
        # bmq_i is the indices to cycle through the Query  BitMatrix
        # bmt_i is the indices to cycle through the Target BitMatrix
        for bmq_i in range(len(self.matrix)):
            for bmt_i in range(len(bmt)):
                # check if the edge are compatible (read method explanation)
                if self.matrix[bmq_i] & bmt[bmt_i] == self.matrix[bmq_i]:
                    candidates.append((bmq_i, bmt_i))

        self._undo_adapt_query_to_target()
        return candidates


class QueryBitMatrixOptimized(QueryBitMatrix):
    def __init__(self, graph, bit_matrix_strategy):
        super().__init__(graph, bit_matrix_strategy)

    def find_candidates(self, target_bitmatrix):
        self._adapt_query_to_target(target_bitmatrix.get_graph())
        # lazy computing
        self._lazy_computing()
        target_bitmatrix._lazy_computing()
        # declare the set of indices that matches the query
        match = []
        """
        The indexed bitmatrix is a list of dictionaries, each dictionary
        is associated to a part of the bitmap. The key of the dictionary
        is the bitmap as a string and the value is the list of indices
        associated to the bitmap.

        Example:
            indexed_bitmatrix = [
                {
                    '000': [1, 2, 3],
                    '001': [4, 5, 6],
                    ...
                },
                ...
            ]
        """
        indexed_bitmatrix = target_bitmatrix.get_matrix()
        # to prune the search I sort the indices by the length of
        # the keys of the dictionary. The less keys the dictionary
        # has the more candidates are associated to the same bitmap.
        indices = sorted(
            range(len(indexed_bitmatrix)),
            key=lambda i: len(indexed_bitmatrix[i].keys()),
        )

        for bmq_i in range(len(self.matrix)):
            # take the i-th row of the query bitmap and split it in 4 parts
            query_row = self.split_bitmap_row(bmq_i)
            # set of indices candidates
            partial_match = set()

            for i in indices:
                # take the i-th subpart of the query bitmap
                bitmap_query = query_row[i]
                # take the i-th indexing of the target
                sub_indexing = indexed_bitmatrix[i]
                # take each distinct bitmap of the i-th column of the bitmatrix
                target_bitmaps_str = [key for key in sub_indexing.keys()]
                # set of candidates
                candidates = set()
                for target_bitmap_str in target_bitmaps_str:
                    # check if the subpart of the target bitmatrix matches the query
                    if (
                        bitmap_query
                        & self.bit_matrix_strategy.str_to_bitmap(target_bitmap_str)
                        == bitmap_query
                    ):
                        # adding the index of the edges associated to the part of the bitmatrix that matches the query
                        candidates = candidates.union(sub_indexing[target_bitmap_str])
                # if there are no candidates I don't need to check the others bitmaps
                #  because it's impossible that all the bitmap matches the query
                if len(candidates) == 0:
                    partial_match = set()
                    break
                # if I found for the first time the candidates I just add them to the match
                if i == indices[0]:
                    partial_match = candidates
                else:
                    # if I already found some candidates I take the intersection.
                    # The intersection is the set of indices that matches all the subpart of the query bitmap
                    partial_match = partial_match.intersection(candidates)

            # if there are candidates I add them to the match
            match.extend([(bmq_i, edge) for edge in partial_match])
        self._undo_adapt_query_to_target()
        return match

    def L_first_distinct_values(self):
        raise NotImplementedError

    def L_second_distinct_values(self):
        raise NotImplementedError

    def T_in_distinct_values(self):
        raise NotImplementedError

    def T_out_distinct_values(self):
        raise NotImplementedError


class BitMatrixStrategy(ABC):

    def __init__(self):
        self.graph = None

    def set_graph(self, graph):
        self.graph = graph

    @abstractmethod
    def str_to_bitmap(self, str):
        pass

    @abstractmethod
    def compute_row(self, edge):
        """Compute the BitMatrix row

        Method that is defined in the Concrete instance depending on the
        type of the bitmap used

        :rtype bitmap
        """
        pass

    def _get_row_string(self, edge):
        """Compute the bitmap with a string format

        :rtype str
        """
        source = edge[0]
        destination = edge[1]
        # compute the bitmaps strings for each part of the row
        L_first = self._compute_node_string_bitmap(source)
        L_second = self._compute_node_string_bitmap(destination)
        tao_in = self._compute_edge_string_bitmap((destination, source))
        tao_out = self._compute_edge_string_bitmap(edge)
        # concatenate the strings
        return L_first + tao_in + tao_out + L_second

    def _compute_node_string_bitmap(self, node):
        """
        Given a node this method compute the bitmap of the labels of that node

        Example:    all node labels in the graph [x, y, z]
                    node label [x, y]
                    bitmap string [110]

        :rtype string
        """
        all_node_labels = self.graph.get_all_node_labels()
        node_labels = self.graph.get_node_labels(node)
        # if the node has no labels it means that the
        # query graph on the specific node has no constraints
        # about the labels, so we return a string of zeros
        # because during the AND operation with the target
        # every node will match
        if len(node_labels) == 0:
            return "0" * len(all_node_labels)
        return "".join(
            "1" if label in node_labels else "0" for label in all_node_labels
        )

    def _compute_edge_string_bitmap(self, edge):
        """
        Given an edge this method compute the bitmap of the label
        of that edge

        Example:    all edge labels in the graph [a, b, c, d]
                    edge label [a, c]
                    bitmap string [1010]

        NOTE:       the edges are directed so edge[0] is the
                    source and edge[0] the destination

        :rtype string
        """
        all_edge_labels = self.graph.get_all_edge_labels()
        edge_labels = self.graph.get_edge_labels(edge[0], edge[1])
        # if the edge has no labels it means that the
        # query graph on the specific edge has no constraints
        # about the labels, so we return a string of zeros
        # because during the AND operation with the target
        # every edge will match
        if len(edge_labels) == 0:
            return "0" * len(all_edge_labels)
        return "".join(
            "1" if label in edge_labels else "0" for label in all_edge_labels
        )


class BitMatrixStrategy1(BitMatrixStrategy):

    def __init__(self):
        super().__init__()

    def compute_row(self, edge):
        """Convert the string bitmap in a bitmap

        It uses bitarray library

        :rtype bitmap
        """
        return bitarray(super()._get_row_string(edge))

    def str_to_bitmap(self, str):
        return bitarray(str)


class BitMatrixStrategy2(BitMatrixStrategy):

    def __init__(self):
        super().__init__()

    def compute_row(self, edge):
        """Convert the string bitmap in a bitmap

        It uses bitarray library

        :rtype bitmap
        """
        return BitArray(bin=super()._get_row_string(edge))

    def str_to_bitmap(self, str):
        return BitArray(bin=str)
