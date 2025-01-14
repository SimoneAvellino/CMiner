from NetworkX.NetworkConfigurator import NetworkConfigurator
from NetworkX.NetworksLoading import NetworksLoading
from CMiner.BreakingConditions import BreakingConditionsNodes
from CMiner.BitMatrix import TargetBitMatrixOptimized, BitMatrixStrategy2
from CMiner.MultiGraphMatch import MultiGraphMatch, Mapping
from Graph.Graph import MultiDiGraph
import pandas as pd
import itertools

def print_red(s):
    print("\033[91m{}\033[00m".format(s))

pattern_count = 0

class EdgeGroupsFinder:
    """
    Class to find all the edge extensions that are frequent.

    How it works:
        Given a list tuples:  list((edge_labels : list(str), location : dict[DBGraph, list[Mapping]]))
        - Construct a table where the columns are the edge labels and each row contains 0 or 1.
        - The last column contains the location of the edges in the graphs.
        - The table is constructed in such a way that the rows are ordered by the number of 1 in the row.
    """
    def __init__(self, min_support):
        # set the column 'location' as the last column
        self.min_support = min_support
        self.df = pd.DataFrame(columns=['location'])

    def columns(self):
        return list(self.df.columns)

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
        return column_name.rsplit('_', 1)[0]

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
            new_labels.extend([EdgeGroupsFinder.column_name(edge_label, n) for n in range(i)])
        return new_labels

    def check_columns(self, edge_labels):
        # CHECK IF THE EDGE LABELS ARE ALREADY PRESENT IN THE COLUMNS
        # NOTE: edge_labels can contain duplicates
        #       e.g.
        #       edge_labels = ['a', 'a', 'b']
        #       columns = ['a_0', 'b_0', 'c_0']
        #       in this case we want to add only 'a_1' because 'a_0','b_0' and 'c_0' is already present in the columns
        columns = self.columns()
        for edge_label in edge_labels:
            if edge_label not in columns:
                self.df[edge_label] = 0

    def compute_new_row(self, edge_labels, location):
        """
        Given a set of edge labels and a location, it returns the new row to add to the dataframe.
        """
        new_row = [0] * len(self.columns())
        cols = self.columns()
        new_row[0] = location
        for l in edge_labels:
            new_row[cols.index(l)] = 1

        return pd.Series(new_row, index=self.df.columns)

    def add_in_order(self, row):
        """
        Add the row in the DataFrame in the correct position.

        The position is determined by the number of 1s in the row.
        The row is added above all the rows which have a number of 1 less than the new row.
        """
        if len(self.df) == 0:
            self.df.loc[0] = row
            return

        new_row_size = sum(row[1:])  # Number of 1s in the new row

        # Find the correct position based on the number of 1s
        for i in range(len(self.df)):
            row_size = sum(self.df.iloc[i][1:])  # number of 1s in the row
            if row_size < new_row_size:
                # add a row at the end to avoid conflicts
                self.df.loc[len(self.df)] = self.df.iloc[len(self.df) - 1]
                # Shift rows down
                self.df.iloc[i + 1:] = self.df.iloc[i:-1]
                # add new_row
                self.df.loc[i] = row
                return

        # If no row with less 1s is found, add at the end
        self.df.loc[len(self.df)] = row

    def add(self, edge_labels, location):
        """
        Given a set of edge labels, graphs and mappings, it adds the edge extension to the dataframe.

        Parameters:
        edge_labels (list[str]): edge labels
        graphs (list[DBGraph]): graphs
        mappings (list[Mapping]): mappings
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
        return len(row['location'].keys())

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
            if bitmap1.iloc[i] > bitmap2.iloc[i]:
                return False
        return True

    @staticmethod
    def extend_location(location1, location2):
        """
        Extend the location of the two rows.
        """
        for g, mappings in location2.items():
            if g in location1:
                location1[g].update(mappings) # FIX deepcopy?
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
    def transform_row_in_extension(row):
        """
        Transform a row in an extension.
        """
        edge_labels = []
        location = {}
        for i, col in enumerate(row.index):
            if i == 0:
                location = row[col]
            elif row[col] == 1:
                edge_labels.append(EdgeGroupsFinder.label_from_column_name(col))
        in_edge_labels, out_edge_labels = EdgeGroupsFinder.split_into_in_and_out_array(edge_labels)
        return Extension(out_edge_labels, in_edge_labels, location)

    def common_columns(self, row1, row2):
        """
        Return the common columns between row1 and row2.
        """
        common = []
        for col in self.columns():
            if row1[col] == 1 and row2[col] == 1:
                common.append(col)
        return common

    def find(self):
        """
        Find all the frequent edge extensions.
        """
        extensions = []

        # i := index row to check
        # j := index row to compare with i-th row
        for i in range(len(self.df)):
            row = self.df.iloc[i]

            j = i - 1

            while j >= 0:
                row_to_compare = self.df.iloc[j]

                if self.is_subset(row, row_to_compare):
                    location = row['location']
                    # merge the location of the two rows
                    location_row_to_compare = row_to_compare['location']
                    EdgeGroupsFinder.extend_location(location, location_row_to_compare)
                j -= 1

            if EdgeGroupsFinder.support(row) >= self.min_support:
                extensions.append(EdgeGroupsFinder.transform_row_in_extension(row))

        return extensions

class Extension:

    def __init__(self, out_edge_labels, in_edge_labels, location):
        """
        Represents an extension of the pattern in the database.
        """
        self.out_edge_labels = out_edge_labels
        self.in_edge_labels = in_edge_labels
        self.location = location

    def graphs(self):
        """
        Return the graphs where the extension is found.
        """
        return list(self.location.keys())

    def mapping(self, graph):
        """
        Return the mapping of the pattern in the graph.
        """
        return self.location[graph]

    def set_graphs(self, graphs):
        """
        Set the graphs where the extension is found.
        """
        self.location = {g: {} for g in graphs}

class NodeExtension(Extension):

    def __init__(self, pattern_node_id, node_labels, out_edge_labels, in_edge_labels, location):
        """
        pattern_node_id -- [out_edge_labels] --> neigh_target_node_id
        pattern_node_id <- [in_edge_labels]  --- neigh_target_node_id
        location (dict[DBGraph, list[Mapping]]): location of the extension in the graphs
        """
        super().__init__(out_edge_labels, in_edge_labels, location)
        self.pattern_node_id = pattern_node_id
        self.node_labels = node_labels

    def target_node_ids(self, graph, _map):
        """
        Return the target node ids from which the extension is found.
        """
        if _map not in self.location[graph]:
            return []
        return self.location[graph][_map]

class EdgeExtension(Extension):

    def __init__(self, pattern_node_id_src, pattern_node_id_dst, out_edge_labels, in_edge_labels, location):
        """

        pattern_node_id_src -- [out_edge_labels] --> pattern_node_id_dst
        pattern_node_id_src <- [in_edge_labels]  --- pattern_node_id_dst
        location (dict[DBGraph, list[Mapping]]): location of the extension in the graphs
        """
        super().__init__(out_edge_labels, in_edge_labels, location)
        self.pattern_node_id_src = pattern_node_id_src
        self.pattern_node_id_dst = pattern_node_id_dst

    def __copy__(self):
        return EdgeExtension(
            self.pattern_node_id_src,
            self.pattern_node_id_dst,
            self.out_edge_labels,
            self.in_edge_labels,
            self.location
        )

class DBGraph(MultiDiGraph):

    def __init__(self, graph, name):
        """
        Represents a graph in the database.
        """
        super().__init__(graph)
        self.name = name
        self.matcher = None

    def _init_matcher(self):
        """
        Initialize the matching algorithm.
        """
        bit_matrix = TargetBitMatrixOptimized(self, BitMatrixStrategy2())
        bit_matrix.compute()
        self.matcher = MultiGraphMatch(self, target_bit_matrix=bit_matrix)

    def localize(self, pattern) -> list['Mapping']:
        """
        Find all the mappings of the pattern in the graph.
        """
        if self.matcher is None:
            self._init_matcher()
        return self.matcher.match(pattern)

    def get_name(self):
        """
        Return the name of the graph
        """
        return self.name

    def __str__(self):
        return self.name

    def all_edges_of_subgraph(self, nodes):
        """
        Return all edges of the subgraph induced by the given nodes.
        The edges are returned as a set of tuples (u, v, key).
        """
        nodes_set = set(nodes)
        edges = set()

        for src, dst in itertools.combinations(nodes, 2):
            for u, v, key in self.edges([src, dst], keys=True):
                if u in nodes_set and v in nodes_set:
                    edges.add((u, v, key))

        return edges

class PatternMappings:

    def __init__(self):
        """
        Keep track of each mapping for each graph of a specific pattern.
        """
        self.patterns_mappings = {}

    def graphs(self) -> list[DBGraph]:
        """
        Return the graphs that contains the pattern
        """
        return list(self.patterns_mappings.keys())

    def mappings(self, graph) -> list[Mapping]:
        """
        Return the mappings of the pattern in the graph.
        """
        return self.patterns_mappings[graph]

    def set_mapping(self, graph, mappings: [Mapping]):
        """
        Set the mappings of the pattern in the graph.
        """
        self.patterns_mappings[graph] = mappings

class NodeExtensionManager:

    def __init__(self, support):
        self.min_support = support
        self.extensions = {}

        self.memoization = {}


    def add(self, pattern_node_id, target_node_id, neigh_target_node_id, db_graph, _map):
        """
        Add an extension to the manager

        Parameters:
            pattern_node_id (int): the id of the node in the pattern that is extended
            target_node_id (int): the id of the node mapped with pattern_node_id
            neigh_target_node_id (int): neighbor of target_node_id
            db_graph (DBGraph): the graph where the extension is found
            _map (Mapping): the mapping of the pattern in the db_graph
        """
        edge_labels = []
        if (pattern_node_id, neigh_target_node_id) not in self.memoization:
            self.memoization[(target_node_id, neigh_target_node_id)] = db_graph.get_edge_labels_with_duplicate(target_node_id, neigh_target_node_id)
            self.memoization[(neigh_target_node_id, target_node_id)] = db_graph.get_edge_labels_with_duplicate(neigh_target_node_id, target_node_id)
        for label in self.memoization[(target_node_id, neigh_target_node_id)]:
            edge_labels.append(NodeExtensionManager.orientation_code(label, True))
        for label in self.memoization[(neigh_target_node_id, target_node_id)]:
            edge_labels.append(NodeExtensionManager.orientation_code(label, False))

        neigh_target_node_labels = db_graph.get_node_labels(neigh_target_node_id)
        edge_labels = sorted(edge_labels)

        neigh_target_node_labels_code = " ".join(neigh_target_node_labels)
        target_edge_labels_code = " ".join(edge_labels)

        ext_code = (pattern_node_id, neigh_target_node_labels_code, target_edge_labels_code)

        if ext_code not in self.extensions:
            self.extensions[ext_code] = {}
        if db_graph not in self.extensions[ext_code]:
            self.extensions[ext_code][db_graph] = []
        self.extensions[ext_code][db_graph].append((_map, neigh_target_node_id))

    def frequent_extensions(self) -> list['NodeExtension']:
        """
        Return a list of NodeExtensions that if applied to the pattern, it still remains frequent.
        """

        frequent_extensions = []

        edge_group_finders = {}

        for (pattern_node_id, node_labels_code, target_edge_labels_code), db_graphs in self.extensions.items():

            # use the finder code to identify the finder
            finder_code = (pattern_node_id, node_labels_code)
            # instantiate the finder if it is not present
            if finder_code not in edge_group_finders:
                edge_group_finders[finder_code] = EdgeGroupsFinder(self.min_support)

            # create the location dictionary
            location = {}
            for g in db_graphs:
                aa = {}
                for mapping, node_id in db_graphs[g]:
                    if mapping not in aa:
                        aa[mapping] = []
                    aa[mapping].append(node_id)
                location[g] = aa

            # select the correct finder and add the edge extension
            edge_group_finder = edge_group_finders[finder_code]
            edge_group_finder.add(target_edge_labels_code.split(" "), location)

        # save all frequent extensions
        for (pattern_node_id, node_labels_code), edge_group_finder in edge_group_finders.items():
            ext = edge_group_finder.find()
            for e in ext:
                frequent_extensions.append(
                    NodeExtension(pattern_node_id, node_labels_code.split(" "), e.out_edge_labels, e.in_edge_labels,
                                  e.location))

        return frequent_extensions

    @staticmethod
    def orientation_code(label, outgoing):
        """
        Return the code of the orientation of the edge.

        E.g.
            src_node -- edge_label --> dst_node

            edge_label became 'out_edge_label' if outgoing is True
            edge_label became 'in_edge_label' if outgoing is False
        """
        return ("out_" if outgoing else "in_") + label

class EdgeExtensionManager:

    def __init__(self, support):
        self.min_support = support
        self.extensions = {}

    def add(self, pattern_node_src, pattern_node_dest, labels, db_graph, _map):
        """
        Add an extension to the manager.
        """

        target_edge_labels_code = " ".join(
            sorted([NodeExtensionManager.orientation_code(label, True) for label in labels]))
        extension_code = (pattern_node_src, pattern_node_dest, target_edge_labels_code)

        if extension_code not in self.extensions:
            self.extensions[extension_code] = {}
        if db_graph not in self.extensions[extension_code]:
            self.extensions[extension_code][db_graph] = []
        self.extensions[extension_code][db_graph].append(_map)

    def frequent_extensions(self) -> list['EdgeExtension']:
        """
        Return a list of Extensions that are frequent
        """
        frequent_extensions = []

        edge_group_finders = {}

        for (pattern_node_src, pattern_node_dest, target_edge_labels_code), db_graphs in self.extensions.items():

            # use the finder code to identify the finder
            finder_code = (pattern_node_src, pattern_node_dest)
            # instantiate the finder if it is not present
            if finder_code not in edge_group_finders:
                edge_group_finders[finder_code] = EdgeGroupsFinder(self.min_support)

            # create the location dictionary
            location = {}
            for g in db_graphs:
                location[g] = set(db_graphs[g])

            # select the correct finder and add the edge extension
            edge_group_finder = edge_group_finders[finder_code]
            edge_group_finder.add(target_edge_labels_code.split(" "), location)

        # save all frequent extensions
        for (pattern_node_src, pattern_node_dest), edge_group_finder in edge_group_finders.items():
            ext = edge_group_finder.find()
            for e in ext:
                frequent_extensions.append(
                    EdgeExtension(pattern_node_src, pattern_node_dest, e.out_edge_labels, e.in_edge_labels, e.location))

        return frequent_extensions

class Pattern(MultiDiGraph):

    def __init__(self, pattern_mappings, extended_pattern: 'Pattern' = None,
                 **attr):
        """
        Represents a pattern in the database.
        """
        if extended_pattern is not None:
            super().__init__(extended_pattern, **attr)
        else:
            super().__init__(**attr)
        self.extended_pattern = extended_pattern
        self.pattern_mappings = pattern_mappings

    def to_graph(self):
        """
        Return the pattern as a networkx graph
        """
        return MultiDiGraph(self)

    def graphs(self) -> list[DBGraph]:
        """
        Return the graphs that contains the pattern
        """
        return self.pattern_mappings.graphs()

    def support(self):
        """
        Return the support of the pattern
        """
        return len(self.graphs())

    def find_node_extensions(self, min_support) -> list[NodeExtension]:
        """
        Generate all possible node extension that if applied to the pattern, it still remains frequent.
        """
        extension_manager = NodeExtensionManager(min_support)
        # for all graph in the database that contains the current extension
        for g in self.graphs():
            # obtain where the current extension is located in the graph
            mappings = self.pattern_mappings.mappings(g)
            # For each map we know one place where the extension is located in the graph.
            # We search all nodes that are neighbors of the current pattern and create a new extension.
            for _map in mappings:
                # retrieve nodes mapped in the DB graph
                mapped_target_nodes = _map.nodes()
                # node_p  := node pattern
                # node_db := node in the DB graph mapped to node_p
                for node_p, node_db in _map.node_pairs():
                    # for each node of the pattern search a possible extension
                    for neigh in g.all_neighbors(node_db).difference(mapped_target_nodes):
                        extension_manager.add(node_p, node_db, neigh, g, _map)

        extensions = extension_manager.frequent_extensions()
        del extension_manager
        return extensions

    def find_edge_extensions(self, min_support) -> list[list[EdgeExtension]]:

        if len(self.nodes()) < 3:
            # if the pattern has less than 3 nodes, it is not possible to find edge extensions
            return []

        extension_manager = EdgeExtensionManager(min_support)

        for g in self.graphs():
            for _map in self.pattern_mappings.mappings(g):
                mapped_pattern_complete_graph_edges = g.all_edges_of_subgraph(_map.nodes())
                mapped_pattern_edges = set(_map.get_target_edges())
                candidate_edges = set()

                for src, dst, key in mapped_pattern_complete_graph_edges:
                    skip = False
                    for s, d, k in mapped_pattern_edges:
                        if src == s and dst == d:
                            # remove i-th element from the list
                            ss, dd, kk = s, d, k
                            mapped_pattern_edges.remove((ss, dd, kk))
                            skip = True
                            break
                    if skip:
                        continue
                    candidate_edges.add((src, dst, key, g.get_edge_label((src, dst, key))))


                groups = {}

                inverse_map = _map.inverse()
                for src, dst, key, lab in candidate_edges:
                    pattern_node_src = inverse_map.node(src)
                    pattern_node_dest = inverse_map.node(dst)
                    code = (pattern_node_src, pattern_node_dest)
                    if code not in groups:
                        groups[code] = []
                    groups[code].append(lab)

                for (src, dst), labels in groups.items():
                    extension_manager.add(src, dst, labels, g, _map)

        extensions = extension_manager.frequent_extensions()

        if len(extensions) == 0:
            return []

        graphs = sorted(self.graphs(), key=lambda x: x.get_name())
        extension_matrix = [[0 for _ in range(len(graphs))] for _ in range(len(extensions))]
        for i, ext in enumerate(extensions):
            for j, g in enumerate(graphs):
                if g in ext.graphs():
                    extension_matrix[i][j] = 1

        # group row by row
        matrix_indices_grouped = {}
        for i, row in enumerate(extension_matrix):
            row_code = "".join(map(str, row))
            if row_code not in matrix_indices_grouped:
                matrix_indices_grouped[row_code] = []
            matrix_indices_grouped[row_code].append(i)

        groups = []
        for row_code, indices in matrix_indices_grouped.items():
            columns_to_select = [i for i, v in enumerate(row_code) if v == "1"]
            group = []
            for i, ext in enumerate(extensions):
                skip = False
                if all(extension_matrix[i][j] == 1 for j in columns_to_select):
                    for e in group:
                        if ext.pattern_node_id_src == e.pattern_node_id_src and ext.pattern_node_id_dst == e.pattern_node_id_dst:
                            skip = True
                            break
                    if skip:
                        continue
                    ext_copy = ext.__copy__()
                    new_location = {v: k for v, k in ext.location.items() if
                                    any(v == graphs[j] for j in columns_to_select)}
                    ext_copy.location = new_location
                    group.append(ext_copy)
            groups.append(group)

        return groups

    def apply_node_extension(self, extension: NodeExtension) -> 'Pattern':
        """
        Apply the node extension to the pattern.
        """

        # Object to keep track of the new pattern mappings
        new_pattern_mappings = PatternMappings()
        # The id of the previous pattern node that is extended
        pattern_node_id = extension.pattern_node_id

        # Apply extension to the pattern (add node and edges)
        new_pattern = Pattern(extended_pattern=self, pattern_mappings=new_pattern_mappings)
        new_pattern_new_node_id = len(new_pattern.nodes())
        new_pattern.add_node(new_pattern_new_node_id, labels=extension.node_labels)

        # in_edge_labels: labels that enter the node from which the extension start
        for lab in extension.in_edge_labels:
            new_pattern.add_edge(new_pattern_new_node_id, pattern_node_id, type=lab)
        # out_edge_labels: labels that exit the node from which the extension start
        for lab in extension.out_edge_labels:
            new_pattern.add_edge(pattern_node_id, new_pattern_new_node_id, type=lab)

        # Update the pattern mappings
        for target in extension.graphs():
            new_mappings = []
            for target_map in self.pattern_mappings.mappings(target):
                target_node_ids = extension.target_node_ids(target, target_map)
                # when trying to extend the pattern Pn (pattern with n nodes), there can be some mappings of Pn
                # that are not extended because the extension is not applicable.
                if len(target_node_ids) == 0:
                    continue

                br_cond_node = BreakingConditionsNodes(new_pattern, target_map._retrieve_node_mapping())

                for target_node_id in target_node_ids:

                    if not br_cond_node.check(new_pattern_new_node_id, target_node_id):
                        continue

                    # node mapping
                    node_mapping = {new_pattern_new_node_id: target_node_id}
                    # edge mapping
                    edge_mapping = {}

                    new_key = 0
                    prev_lab = None
                    prev_keys = []
                    for lab in sorted(extension.in_edge_labels):
                        new_pattern_edge = (new_pattern_new_node_id, pattern_node_id, new_key)
                        target_edge_src = target_node_id
                        target_edge_dest = target_map.nodes_mapping()[extension.pattern_node_id]
                        if prev_lab != lab:
                            prev_keys = target.edge_keys_by_type(target_edge_src, target_edge_dest, lab)
                            prev_lab = lab
                        target_edge = (target_edge_src, target_edge_dest, prev_keys.pop(0))
                        edge_mapping[new_pattern_edge] = target_edge
                        new_key += 1
                    new_key = 0
                    prev_lab = None
                    prev_keys = []
                    for lab in sorted(extension.out_edge_labels):
                        new_pattern_edge = (pattern_node_id, new_pattern_new_node_id, new_key)
                        target_edge_src = target_map.nodes_mapping()[extension.pattern_node_id]
                        target_edge_dest = target_node_id
                        if prev_lab != lab:
                            prev_keys = target.edge_keys_by_type(target_edge_src, target_edge_dest, lab)
                            prev_lab = lab
                        target_edge = (target_edge_src, target_edge_dest, prev_keys.pop(0))
                        edge_mapping[new_pattern_edge] = target_edge
                        new_key += 1

                    new_mapping = Mapping(node_mapping=node_mapping, edge_mapping=edge_mapping,
                                          extended_mapping=target_map)
                    new_mappings.append(new_mapping)
            if len(new_mappings) > 0:
                new_pattern_mappings.set_mapping(target, new_mappings)
        return new_pattern

    def apply_edge_extension(self, extensions: list[EdgeExtension]) -> 'Pattern':
        """
        Apply the edge extension to the pattern.
        """

        db_graphs = extensions[0].graphs()
        new_pattern_mappings = PatternMappings()

        # Apply extension to the pattern (add edges)
        new_pattern = Pattern(extended_pattern=self, pattern_mappings=new_pattern_mappings)

        for ext in extensions:
            for lab in ext.in_edge_labels:
                new_pattern.add_edge(ext.pattern_node_id_dst, ext.pattern_node_id_src, type=lab)
            for lab in ext.out_edge_labels:
                new_pattern.add_edge(ext.pattern_node_id_src, ext.pattern_node_id_dst, type=lab)

        # Update the pattern mappings
        for target in db_graphs:

            new_mappings = []

            for target_map in self.pattern_mappings.mappings(target):

                try:
                    if any(target_map not in ext.mapping(target) for ext in extensions):
                        continue
                except:
                    # target could not be associated to any mapping in the extension
                    continue

                new_mapping = Mapping(extended_mapping=target_map)

                for extension in extensions:

                    target_edge_src = target_map.nodes_mapping()[extension.pattern_node_id_src]
                    target_edge_dest = target_map.nodes_mapping()[extension.pattern_node_id_dst]
                    new_key = 0
                    prev_lab = None
                    prev_keys = []
                    for lab in sorted(extension.out_edge_labels):
                        if prev_lab != lab:
                            prev_keys = target.edge_keys_by_type(target_edge_src, target_edge_dest, lab)
                            prev_lab = lab
                        if len(prev_keys) == 0:
                            continue
                        target_edge = (target_edge_src, target_edge_dest, prev_keys.pop(0))
                        new_mapping.set_edge((extension.pattern_node_id_dst, extension.pattern_node_id_src, new_key),
                                             target_edge)
                        new_key += 1
                    new_key = 0
                    prev_lab = None
                    prev_keys = []
                    for lab in sorted(extension.in_edge_labels):
                        if prev_lab != lab:
                            prev_keys = target.edge_keys_by_type(target_edge_src, target_edge_dest, lab)
                            prev_lab = lab
                        if len(prev_keys) == 0:
                            continue
                        target_edge = (target_edge_src, target_edge_dest, prev_keys.pop(0))
                        new_mapping.set_edge((extension.pattern_node_id_src, extension.pattern_node_id_dst, new_key),
                                             target_edge)
                        new_key += 1

                new_mappings.append(new_mapping)
            new_pattern_mappings.set_mapping(target, new_mappings)

        return new_pattern

    def __str__(self, show_mappings=False, is_directed=False, with_frequencies=False):

        global pattern_count
        output = ""

        output += f"t # {pattern_count}\n"

        to_remove = None

        # graph info
        if is_directed:
            output += self.directed_pattern_str()
        else:
            output += self.undirected_pattern_str()

        output += f"s {self.support()}\n"
        output += f"f {sum(len(self.pattern_mappings.mappings(g)) for g in self.graphs())}\n"

        if with_frequencies:
            frequencies = ["(" + g.get_name() + ", " + str(len(self.pattern_mappings.mappings(g))) + ")" for g in
                           self.graphs()]
            output += "x " + " ".join(frequencies) + "\n"
        if show_mappings:
            if is_directed:
                output += self.directed_mapping_str()
            else:
                output += self.undirected_mapping_str(to_remove)

        output += "----------"

        pattern_count += 1
        return output

    def edge_to_remove_when_undirected(self):
        """
        Return the edges to remove when the pattern is undirected.
        """
        to_remove = {}
        for e in self.edges(keys=True):
            s, d, k = e
            src_code = "".join(sorted(self.get_node_labels(s)))
            dst_code = "".join(sorted(self.get_node_labels(d)))
            if src_code != dst_code:
                continue

            if (d, s) in to_remove:
                continue
            if (s, d) not in to_remove:
                to_remove[(s, d)] = []
            to_remove[(s, d)].append(k)
        return to_remove

    def directed_mapping_str(self):
        """
        Print the mappings in a directed way.
        """
        output = ""
        for g in self.graphs():
            output += g.get_name() + " " + str(len(self.pattern_mappings.mappings(g))) + " "
            # output += " ".join([str(_map) for _map in self.pattern_mappings.mappings(g)])
            output += "\n"
        return output

    def undirected_mapping_str(self, to_remove):
        """
        Print the corret mappings for the undirected pattern.
        """
        to_remove = self.edge_to_remove_when_undirected()
        output = ""
        for g in self.graphs():
            output += g.get_name() + " " + str(len(self.pattern_mappings.mappings(g))) + " "
            for _map in self.pattern_mappings.mappings(g):
                m = _map._retrieve_edge_mapping()
                for (s, d) in to_remove:
                    for k in to_remove[(s, d)]:
                        if (s, d, k) in m:
                            del m[(s, d, k)]
                new_map = Mapping(edge_mapping=m, node_mapping=_map._retrieve_node_mapping())
                output += str(new_map) + " "
                del new_map
            output += "\n"

        return output

    def directed_pattern_str(self):
        """
        Print the pattern in a directed way.
        """
        graph_str = ""
        for node in self.nodes(data=True):
            graph_str += f"v {node[0]} {' '.join(node[1]['labels'])}\n"
        for edge in self.edges(data=True):
            graph_str += f"e {edge[0]} {edge[1]} {edge[2]['type']}\n"

        # edge_labels = {}
        # for edge in self.edges(data=True):
        #     e = (edge[0], edge[1])
        #     if e not in edge_labels:
        #         edge_labels[e] = []
        #     edge_labels[e].append(edge[2]['type'])
        # for edge, labels in edge_labels.items():
        #     labels = " ".join(labels)
        #     graph_str += f"e {edge[0]} {edge[1]} {labels}\n"

        return graph_str

    def undirected_pattern_str(self):
        """
        Print the pattern in an undirected way.
        """

        to_remove = {}
        for e in self.edges(keys=True):
            s, d, k = e
            src_code = "".join(sorted(self.get_node_labels(s)))
            dst_code = "".join(sorted(self.get_node_labels(d)))
            if src_code != dst_code:
                continue

            if (d, s) in to_remove:
                continue
            if (s, d) not in to_remove:
                to_remove[(s, d)] = []
            to_remove[(s, d)].append(k)

        graph_str = ""
        for node in self.nodes(data=True):
            graph_str += f"v {node[0]} {' '.join(node[1]['labels'])}\n"

        for s, d, data in self.edges(data=True):
            if (s, d) not in to_remove:
                graph_str += f"e {s} {d} {data['type']}\n"

        return graph_str

class DFSStack(list):

    def __init__(self, min_nodes, max_nodes, output_options):
        """
        Stack that only keeps patterns that can be extended.
        """
        super().__init__()
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        # Dictionary that keeps track of already computed pattern.
        # key   := pattern_code
        # value := list of patterns
        self.found_patterns = set()
        self.output_options = output_options
        # In case of closed pattern mining, keep track of the last popped pattern
        # in case of backtracking it is printed.
        self.last_popped_pattern = None
        # Memoize BitMatrices
        self.bit_matrices = {}

    def pop(self, __index = -1, backtracking = False) -> Pattern:
        """
        Pop the last element from the stack.
        """
        if self.output_options["closed_patterns"] and backtracking:
            self.output(self.last_popped_pattern)

        pattern = super().pop(__index)
        self.last_popped_pattern = pattern
        return pattern

    def push(self, pattern: Pattern):
        """
        Push the pattern into the stack.
        """
        if len(pattern.nodes()) <= self.max_nodes and not self.was_stacked(pattern):
                super().append(pattern)
                code = pattern.canonical_code()
                if code not in self.found_patterns:
                    self.found_patterns.add(code)
                if not self.output_options["closed_patterns"]:
                    self.output(pattern)
        else:
            del pattern

    def was_stacked(self, pattern: Pattern):
        """
        Check if the pattern is already computed.

        NOTE: If two pattern are isomorphic, they have the same code.
              There could be two different patterns with the same code,
              so the code is used to prune the search space, but the
              isomorphism check is still needed.
        """
        code = pattern.canonical_code()
        return code in self.found_patterns

    def output(self, pattern: Pattern):
        if len(pattern.nodes()) < self.min_nodes:
            return
        print(pattern.__str__(self.output_options["show_mappings"], self.output_options["is_directed"], self.output_options["with_frequencies"]))

class CMiner:

    def __init__(self,
                 db_file,
                 support,
                 min_nodes=1,
                 max_nodes=float('inf'),
                 start_patterns=None,  # implement
                 show_mappings=False,
                 output_path=None,
                 is_directed=False,
                 with_frequencies=False,
                 only_closed_patterns=False
                 ):
        self.db_file = db_file
        self.stack = DFSStack(min_nodes, max_nodes, {
            "is_directed": is_directed,
            "show_mappings": show_mappings,
            "with_frequencies": with_frequencies,
            "closed_patterns": only_closed_patterns
        })
        self.min_support = support
        self._start_patterns = start_patterns
        self.db = []
        self.show_mappings = show_mappings
        self.output_path = output_path
        self.is_directed = is_directed
        self.with_frequencies = with_frequencies
        self.only_closed_patterns = only_closed_patterns

    def mine(self):
        self._read_graphs_from_file()
        self._parse_support()
        self._find_start_patterns()

        backtracking = False

        while self.stack:

            pattern_to_extend = self.stack.pop(backtracking=backtracking)

            backtracking = False

            if len(pattern_to_extend.nodes()) >= self.stack.max_nodes:
                backtracking = True
                continue

            node_extensions = pattern_to_extend.find_node_extensions(self.min_support)

            if len(node_extensions) == 0:
                # Backtracking occurs when no more extensions are found
                backtracking = True
                del pattern_to_extend
                continue

            for node_ext in node_extensions:

                node_extended_pattern = pattern_to_extend.apply_node_extension(node_ext)

                self.stack.push(node_extended_pattern)

                edge_extensions = node_extended_pattern.find_edge_extensions(self.min_support)

                for edge_ext in edge_extensions:

                    edge_extended_pattern = node_extended_pattern.apply_edge_extension(edge_ext)

                    self.stack.push(edge_extended_pattern)

    # def mine(self):
    #     self._read_graphs_from_file()
    #     self._parse_support()
    #     self._find_start_patterns()
    #
    #     backtracking = False
    #
    #     while self.stack:
    #
    #         pattern_to_extend = self.stack.pop(backtracking=backtracking)
    #
    #         backtracking = False
    #
    #         if len(pattern_to_extend.nodes()) >= self.stack.max_nodes:
    #             backtracking = True
    #             continue
    #
    #         node_extensions = pattern_to_extend.find_node_extensions(self.min_support)
    #
    #         if len(node_extensions) == 0:
    #             # Backtracking occurs when no more extensions are found
    #             backtracking = True
    #             del pattern_to_extend
    #             continue
    #
    #         for node_ext in node_extensions:
    #
    #             node_extended_pattern = pattern_to_extend.apply_node_extension(node_ext)
    #             # Ensure no duplicate patterns are processed
    #             if self.stack.was_stacked(node_extended_pattern) or node_extended_pattern.support() == 0:
    #                 continue
    #
    #             tree_pattern_added = False
    #
    #             edge_extensions = node_extended_pattern.find_edge_extensions(self.min_support)
    #
    #             # If no edge extensions are found, add the pattern
    #             # to the stack, it could be extended adding a node
    #             if len(edge_extensions) == 0:
    #                 self.stack.push(node_extended_pattern)
    #                 continue
    #
    #             graphs_covered_by_edge_extensions = {g for edge_ext in edge_extensions for g in edge_ext[0].graphs()}
    #
    #             for edge_ext in edge_extensions:
    #
    #                 edge_extended_pattern = node_extended_pattern.apply_edge_extension(edge_ext)
    #
    #                 # If the support of the tree pattern is greater than the cycle pattern
    #                 # it means that the tree cannot be closed in a cycle for all of his
    #                 # occurrence in each graph, so it's considered the tree pattern and added to the stack.
    #                 # Also check if the pattern is not already in the stack, because the same tree can be
    #                 # considered with more than one edge extension.
    #                 if (not tree_pattern_added) and (
    #                         node_extended_pattern.support() > len(graphs_covered_by_edge_extensions)) and (
    #                         node_extended_pattern.support() > edge_extended_pattern.support()):
    #                 # if (not tree_pattern_added) and (
    #                 #             node_extended_pattern.support() > edge_extended_pattern.support()):
    #                     self.stack.push(node_extended_pattern)
    #                     tree_pattern_added = True
    #
    #                 self.stack.push(edge_extended_pattern)

    def _find_start_patterns(self) -> [Pattern]:

        if self._start_patterns is None:
            for p in self._mine_1node_patterns():
                self.stack.push(p)
            return

        for p in self._start_patterns:
            pattern_mappings = PatternMappings()
            for g in self.db:
                matching = g.localize(p)
                if len(matching) > 0:
                    pattern_mappings.set_mapping(g, matching)
            self.stack.push(Pattern(extended_pattern=p, pattern_mappings=pattern_mappings))
                
    def _mine_1node_patterns(self) -> list[Pattern]:
        counter = {}
        for g in self.db:
            for node in g.nodes():
                sorted_labels = g.get_node_labels(node)
                sorted_labels_str = " ".join(sorted_labels)
                if sorted_labels_str in counter:
                    counter[sorted_labels_str].add(g)
                else:
                    counter[sorted_labels_str] = {g}

        # update the mappings
        patterns = []
        for sorted_labels_str, graphs in counter.items():
            if len(graphs) >= self.min_support:
                pattern_mappings = PatternMappings()
                p = Pattern(pattern_mappings)
                p.add_node(0, labels=sorted_labels_str.split(" "))
                for g in graphs:
                    p.pattern_mappings.set_mapping(g, [Mapping(node_mapping={0: node}) for node in g.nodes() if
                                                       g.get_node_labels(node) == sorted_labels_str.split(" ")])
                patterns.append(p)
        return patterns

    def _read_graphs_from_file(self):
        type_file = self.db_file.split('.')[-1]
        configurator = NetworkConfigurator(self.db_file, type_file)
        for name, network in NetworksLoading(type_file, configurator.config).Networks.items():
            self.db.append(DBGraph(network, name))
        self._parse_graphs_direction()

    def _parse_graphs_direction(self):
        """
        Handle the case when the mining is not directed.

        If the mining is not directed, convert all the graphs to undirected using these rules:
        Let (x, y) be an edge in the directed graph:
            if x and y nodes have the same label, then add the edge (y, x) with the same label
            if x and y nodes have different labels, revert the edge (x, y) to (y, x)
        """
        if self.is_directed:
            return

        for g in self.db:
            edges = list(g.edges(keys=True))
            for edge in edges:
                src, dst, key = edge
                src_labels_code = "".join(sorted(g.get_node_labels(src)))
                dst_labels_code = "".join(sorted(g.get_node_labels(dst)))
                if src_labels_code == dst_labels_code:
                    all_keys = g.edge_keys(dst, src) if g.has_edge(dst, src) else [-1]
                    g.add_edge(dst, src, key=max(all_keys) + 1, type=g.get_edge_label(edge))
                elif src_labels_code > dst_labels_code:
                    edge_label = g.get_edge_label(edge)
                    g.remove_edge(src, dst, key=key)
                    g.add_edge(dst, src, key=key, type=edge_label)

    def _parse_support(self):
        """
        If the support is > 1, then the user want common
        graphs that are present in a certain amount of
        db graphs.
        If the support is <= 1 then the user want common
        graphs that are present in a certain percentage of
        df graphs.
        """
        if self.min_support <= 1:
            db_len = len(self.db)
            self.min_support = int(self.min_support * db_len)