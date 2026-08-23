from .Extension import DirectedExtension, Extension, UndirectedExtension
import bisect


class EdgeExtension:

    def __init__(
        self, pattern_node_src, pattern_node_dst, extension_strategy: Extension
    ):
        """
        Initialize the EdgeExtension with a specific extension strategy.

        Parameters:
            extension_strategy (Extension): The strategy to use for extending the edge.
        """
        self.extension_strategy = extension_strategy
        self.pattern_node_src = pattern_node_src
        self.pattern_node_dst = pattern_node_dst

    def __str__(self):
        return f"EdgeExtension({self.pattern_node_src}, {self.pattern_node_dst}, {self.extension_strategy})"

    def __copy__(self):
        return EdgeExtension(
            self.pattern_node_src,
            self.pattern_node_dst,
            self.extension_strategy.__copy__(),
        )

    def graphs(self):
        """
        Return the graphs of the extension strategy.
        """
        return self.extension_strategy.graphs()


class EdgeExtensionManager:

    def __init__(self, support):
        self.min_support = support
        self.extensions = {}
        # Two-pass discovery state (see compute_needed_codes):
        # count_only=True  -> add() records only ext_code -> set(graphs)
        # code_whitelist   -> add() stores tuples only for these codes
        self.count_only = False
        self.code_whitelist = None
        self._graph_sets = {}

    def needed_codes(self) -> set:
        """
        Pass-1 result: ext_codes whose occurrence tuples must be collected
        in pass 2 to reproduce the exact frequent_extensions output.
        """
        needed = compute_needed_codes(self._graph_sets, self.min_support)
        self._graph_sets = {}
        return needed

    def add(self, pattern_node_src, pattern_node_dest, labels, db_graph, _map):
        """
        Add an extension to the manager

        Parameters:
        pattern_node_src (int): The source node of the pattern.
        pattern_node_dest (int): The destination node of the pattern.
        labels (list[str]): The labels of the edges.
        db_graph (DBGraph): The database graph.
        _map (Mapping): The mapping in the database graph.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def frequent_extensions(self) -> list[EdgeExtension]:
        """
        Return a list of EdgeExtensions that if applied to the pattern, it still remains frequent.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")


class DirectedEdgeExtensionManager(EdgeExtensionManager):

    def __init__(self, support):
        """
        Initialize the DirectedEdgeExtensionManager with a specific support value.
        Parameters:
            support (float): The minimum support value for the extensions.
        """
        super().__init__(support)

    def add(self, pattern_node_src, pattern_node_dest, labels, db_graph, _map):
        """
        Add an extension to the manager.
        """
        from .NodeExtension import (
            DirectedNodeExtensionManager,
        )  # avoid circular import, is there a better way?

        target_edge_labels_code = " ".join(
            sorted(
                DirectedNodeExtensionManager.orientation_code(label, True)
                for label in labels
            )
        )
        extension_code = (pattern_node_src, pattern_node_dest, target_edge_labels_code)

        if self.count_only:
            self._graph_sets.setdefault(extension_code, set()).add(db_graph)
            return
        if self.code_whitelist is not None and extension_code not in self.code_whitelist:
            return

        if extension_code not in self.extensions:
            self.extensions[extension_code] = {}
        if db_graph not in self.extensions[extension_code]:
            self.extensions[extension_code][db_graph] = []
        self.extensions[extension_code][db_graph].append(_map)

    def frequent_extensions(self) -> list["EdgeExtension"]:
        """
        Return a list of Extensions that are frequent
        """
        frequent_extensions = []

        edge_group_finders = {}

        # Drain self.extensions as we go
        while self.extensions:
            (
                pattern_node_src,
                pattern_node_dest,
                target_edge_labels_code,
            ), db_graphs = self.extensions.popitem()

            # use the finder code to identify the finder
            finder_code = (pattern_node_src, pattern_node_dest)
            # instantiate the finder if it is not present
            if finder_code not in edge_group_finders:
                edge_group_finders[finder_code] = EdgeGroupsFinder(self.min_support)

            # create the location dictionary
            location = {}
            for g in db_graphs:
                location[g] = set(db_graphs[g])
            del db_graphs

            # select the correct finder and add the edge extension
            edge_group_finder = edge_group_finders[finder_code]
            edge_group_finder.add(target_edge_labels_code.split(" "), location)

        # save all frequent extensions
        for (
            pattern_node_src,
            pattern_node_dest,
        ), edge_group_finder in edge_group_finders.items():
            ext = edge_group_finder.find()
            for e in ext:
                frequent_extensions.append(
                    EdgeExtension(
                        pattern_node_src,
                        pattern_node_dest,
                        DirectedExtension(
                            e.out_edge_labels, e.in_edge_labels, e.location
                        ),
                    )
                )

        return frequent_extensions


class UndirectedEdgeExtensionManager(EdgeExtensionManager):

    def __init__(self, support):
        """
        Initialize the DirectedEdgeExtensionManager with a specific support value.
        Parameters:
            support (float): The minimum support value for the extensions.
        """
        super().__init__(support)

    def add(self, pattern_node_src, pattern_node_dest, labels, db_graph, _map):
        """
        Add an extension to the manager.
        """
        from .NodeExtension import (
            DirectedNodeExtensionManager,
        )  # avoid circular import, is there a better way?

        target_edge_labels_code = " ".join(
            sorted(
                DirectedNodeExtensionManager.orientation_code(label, True)
                for label in labels
            )
        )
        extension_code = (pattern_node_src, pattern_node_dest, target_edge_labels_code)

        if self.count_only:
            self._graph_sets.setdefault(extension_code, set()).add(db_graph)
            return
        if self.code_whitelist is not None and extension_code not in self.code_whitelist:
            return

        if extension_code not in self.extensions:
            self.extensions[extension_code] = {}
        if db_graph not in self.extensions[extension_code]:
            self.extensions[extension_code][db_graph] = []
        self.extensions[extension_code][db_graph].append(_map)

    def frequent_extensions(self) -> list["EdgeExtension"]:
        """
        Return a list of Extensions that are frequent
        """
        frequent_extensions = []

        edge_group_finders = {}

        while self.extensions:
            (
                pattern_node_src,
                pattern_node_dest,
                target_edge_labels_code,
            ), db_graphs = self.extensions.popitem()

            # use the finder code to identify the finder
            finder_code = (pattern_node_src, pattern_node_dest)
            # instantiate the finder if it is not present
            if finder_code not in edge_group_finders:
                edge_group_finders[finder_code] = EdgeGroupsFinder(self.min_support)

            # create the location dictionary
            location = {}
            for g in db_graphs:
                location[g] = set(db_graphs[g])
            del db_graphs

            # select the correct finder and add the edge extension
            edge_group_finder = edge_group_finders[finder_code]
            edge_group_finder.add(target_edge_labels_code.split(" "), location)

        # save all frequent extensions
        for (
            pattern_node_src,
            pattern_node_dest,
        ), edge_group_finder in edge_group_finders.items():
            ext = edge_group_finder.find()
            if pattern_node_src > pattern_node_dest:
                pattern_node_src, pattern_node_dest = (
                    pattern_node_dest,
                    pattern_node_src,
                )
            for e in ext:
                frequent_extensions.append(
                    EdgeExtension(
                        pattern_node_src,
                        pattern_node_dest,
                        UndirectedExtension(
                            e.out_edge_labels, e.location
                        ),  # FIXME: in_edge_labels is not used because it is undirected so it is not needed
                    )
                )

        return frequent_extensions


class EdgeGroupsFinder:
    """
    Class to find all the edge extensions that are frequent.

    How it works:
        Given a list tuples:  list((edge_labels : list(str), location : dict[DBGraph, list[Mapping]]))
        - Construct a table where the columns are the edge labels and each row contains 0 or 1.
        - The first column contains the location of the edges in the graphs.
        - The table is constructed in such a way that the rows are ordered by the number of 1 in the row.

    Rows are plain Python lists ([location, 0/1, ...])
    with a cached int bitmask + popcount per row, kept sorted with bisect.
    """

    def __init__(self, min_support):
        self.min_support = min_support
        # column order: "location" first, then labels in first-appearance order
        self._columns = ["location"]
        self._col_index = {"location": 0}
        # row layout: [location, 0/1, 0/1, ...] aligned with self._columns
        self._rows: list[list] = []
        self._masks: list[int] = []  # bit (col_idx - 1) set => row[col_idx] == 1
        self._neg_popcounts: list[int] = []  # ascending negated popcounts (bisect)

    def columns(self):
        return list(self._columns)

    @staticmethod
    def column_name(label, i):
        """
        Return the column name.
        """
        return label + "_" + str(i)

    @staticmethod
    def label_from_column_name(column_name):
        """
        Return the label from the column name.
        """
        return column_name.rsplit("_", 1)[0]

    @staticmethod
    def parse_edge_labels(edge_labels):
        """
        Parse the edge_labels array. For each edge label add _0. If there are duplicates add _1, _2, ...
        """
        edge_labels_dict = {}
        for i, edge_label in enumerate(edge_labels):
            if edge_label not in edge_labels_dict:
                edge_labels_dict[edge_label] = 1
            else:
                edge_labels_dict[edge_label] += 1
        new_labels = []
        for edge_label, i in edge_labels_dict.items():
            new_labels.extend(
                [EdgeGroupsFinder.column_name(edge_label, n) for n in range(i)]
            )
        return new_labels

    def check_columns(self, edge_labels):
        # CHECK IF THE EDGE LABELS ARE ALREADY PRESENT IN THE COLUMNS
        # NOTE: edge_labels can contain duplicates
        #       e.g.
        #       edge_labels = ['a', 'a', 'b']
        #       columns = ['a_0', 'b_0', 'c_0']
        #       in this case we want to add only 'a_1' because 'a_0','b_0' and 'c_0' is already present in the columns
        for edge_label in edge_labels:
            if edge_label not in self._col_index:
                self._col_index[edge_label] = len(self._columns)
                self._columns.append(edge_label)
                # fill the new column with 0 on all existing rows
                for row in self._rows:
                    row.append(0)

    def compute_new_row(self, edge_labels, location):
        """
        Given a set of edge labels and a location, it returns the new row to add to the dataframe.
        """
        new_row = [0] * len(self._columns)
        new_row[0] = location
        for l in edge_labels:
            new_row[self._col_index[l]] = 1
        return new_row

    def add_in_order(self, row):
        """
        Add the row in the table in the correct position.

        The position is determined by the number of 1s in the row.
        The row is added above all the rows which have a number of 1 less
        than the new row (stable for equal counts: after existing equals).
        """
        mask = 0
        for idx in range(1, len(row)):
            if row[idx]:
                mask |= 1 << (idx - 1)
        popcount = mask.bit_count()
        ins = bisect.bisect_right(self._neg_popcounts, -popcount)
        self._rows.insert(ins, row)
        self._masks.insert(ins, mask)
        self._neg_popcounts.insert(ins, -popcount)

    def add(self, edge_labels, location):
        """
        Given a set of edge labels, graphs and mappings, it adds the edge extension to the table.

        Parameters:
        edge_labels (list[str]): edge labels
        location (dict): location of the extension in the graphs
        """
        edge_labels = EdgeGroupsFinder.parse_edge_labels(edge_labels)
        self.check_columns(edge_labels)
        new_row = self.compute_new_row(edge_labels, location)
        self.add_in_order(new_row)

    @staticmethod
    def support(row):
        """
        Return the support of the row.
        """
        return len(row[0].keys())

    @staticmethod
    def bitmap(row):
        """
        Return the bitmap of the row.
        """
        # each row contains the location of the edges in the graphs, this method returns the bitmap of the row
        # e.g.
        #   row = [{g1: [m1, m2]}, 1, 0, 1]
        #   bitmap = [1, 0, 1]
        return row[1:]

    def is_subset(self, row1, row2):
        """
        Return True if row1 is a subset of row2.
        """
        bitmap1 = EdgeGroupsFinder.bitmap(row1)
        bitmap2 = EdgeGroupsFinder.bitmap(row2)
        for i in range(len(bitmap1)):
            if bitmap1[i] > bitmap2[i]:
                return False
        return True

    @staticmethod
    def extend_location(location1, location2):
        """
        Extend the location of the two rows.
        """
        for g, mappings in location2.items():
            if g in location1:
                location1[g].update(mappings)  # FIX deepcopy?
            else:
                location1[g] = mappings

    @staticmethod
    def split_into_in_and_out_array(array):
        in_array = []
        out_array = []

        for str in array:
            if str.startswith("in_"):
                in_array.append(str[3:])
            elif str.startswith("out_"):
                out_array.append(str[4:])

        return in_array, out_array

    @staticmethod
    def transform_row_in_extension(row, columns):
        """
        Transform a row in an extension.
        """
        edge_labels = []
        location = row[0]
        for i in range(1, len(columns)):
            if row[i] == 1:
                edge_labels.append(EdgeGroupsFinder.label_from_column_name(columns[i]))
        in_edge_labels, out_edge_labels = EdgeGroupsFinder.split_into_in_and_out_array(
            edge_labels
        )
        return DirectedExtension(out_edge_labels, in_edge_labels, location)

    def common_columns(self, row1, row2):
        """
        Return the common columns between row1 and row2.
        """
        common = []
        for i, col in enumerate(self._columns):
            if row1[i] == 1 and row2[i] == 1:
                common.append(col)
        return common

    def find(self):
        """
        Find all the frequent edge extensions.

        Rows are visited in descending-popcount order. Each row absorbs the
        locations of every previous row that is a superset of it (the merge
        is in-place and order-dependent), then a
        row is emitted when its support reaches min_support.
        """
        extensions = []

        # i := index row to check
        # j := index row to compare with i-th row
        for i in range(len(self._rows)):
            row = self._rows[i]
            mask_i = self._masks[i]

            for j in range(i - 1, -1, -1):
                # row_i subset of row_j  <=>  mask_i & ~mask_j == 0
                if mask_i & ~self._masks[j] == 0:
                    # merge the location of the two rows
                    EdgeGroupsFinder.extend_location(row[0], self._rows[j][0])

            if EdgeGroupsFinder.support(row) >= self.min_support:
                extensions.append(
                    EdgeGroupsFinder.transform_row_in_extension(row, self._columns)
                )

        return extensions


class _LightEdgeGroupsFinder:
    """
    Graph-set replica of EdgeGroupsFinder, used by the two-pass extension
    discovery (pass 1) to decide which extension codes survive without
    materializing any occurrence tuple.

    Rows carry a frozenset of graphs instead of the full location dict.
    Row ordering (descending popcount, stable), the transitive top-down
    merge (row_i absorbs row_j when row_i ⊆ row_j) and the emission rule
    (|merged graphs| >= min_support) mirror EdgeGroupsFinder exactly, so the
    survival decisions are identical to the full finder's: the full finder
    computes support as the number of graph keys in the merged location,
    which is exactly the merged graph-set union computed here.
    """

    def __init__(self, min_support):
        self.min_support = min_support
        self._columns = []
        self._col_index = {}
        self._masks: list[int] = []
        self._graphs: list[set] = []  # own graph set per row
        self._codes: list = []  # ext_code per row
        self._neg_popcounts: list[int] = []

    def add(self, edge_labels, graphs, code):
        edge_labels = EdgeGroupsFinder.parse_edge_labels(edge_labels)
        mask = 0
        for label in edge_labels:
            if label not in self._col_index:
                self._col_index[label] = len(self._columns)
                self._columns.append(label)
            mask |= 1 << self._col_index[label]
        popcount = mask.bit_count()
        ins = bisect.bisect_right(self._neg_popcounts, -popcount)
        self._masks.insert(ins, mask)
        self._graphs.insert(ins, graphs)
        self._codes.insert(ins, code)
        self._neg_popcounts.insert(ins, -popcount)

    def find(self):
        """
        Return (emitted_codes, needed_codes): emitted rows (in row order)
        plus every row that is a superset of an emitted row (its merge
        contributors, transitively closed by the ⊆ relation).
        """
        merged = [set(g) for g in self._graphs]
        emitted = []
        for i in range(len(self._masks)):
            mask_i = self._masks[i]
            for j in range(i - 1, -1, -1):
                if mask_i & ~self._masks[j] == 0:
                    merged[i] |= merged[j]
            if len(merged[i]) >= self.min_support:
                emitted.append(i)
        needed = set()
        for i in emitted:
            needed.add(self._codes[i])
            mask_i = self._masks[i]
            for j in range(len(self._masks)):
                if j != i and mask_i & ~self._masks[j] == 0:
                    needed.add(self._codes[j])
        return [self._codes[i] for i in emitted], needed


_TWO_PASS = True


def set_two_pass(enabled: bool):
    """Enable/disable the two-pass extension discovery (see Pattern.find_*)."""
    global _TWO_PASS
    _TWO_PASS = bool(enabled)


def two_pass_enabled() -> bool:
    return _TWO_PASS


def compute_needed_codes(graph_sets: dict, min_support) -> set:
    """
    Two-pass extension discovery, pass-1 survivor computation.

    Parameters:
        graph_sets: {ext_code: set(DBGraph)} collected by a manager in
            count_only mode. ext_code is (finder_a, finder_b, labels_code);
            the finder group key is ext_code[:2] for both node extensions
            ((pattern_node_id, node_labels_code)) and edge extensions
            ((pattern_node_src, pattern_node_dst)).
        min_support: minimum number of graphs for an extension to survive.

    Returns:
        The set of ext_codes whose full occurrence tuples must be
        materialized in pass 2 to reproduce the exact frequent_extensions
        output (emitted rows plus all their merge contributors).

    Codes are consumed in REVERSE insertion order, matching the popitem()
    drain order of the full managers, so the light finders see rows in the
    same order as the full ones.
    """
    finders: dict = {}
    for code in reversed(list(graph_sets.keys())):
        finder_code = code[:2]
        finder = finders.get(finder_code)
        if finder is None:
            finder = finders[finder_code] = _LightEdgeGroupsFinder(min_support)
        finder.add(code[2].split(" "), graph_sets[code], code)
    needed = set()
    for finder in finders.values():
        _, finder_needed = finder.find()
        needed |= finder_needed
    return needed
