from .Extension import DirectedExtension, Extension, UndirectedExtension
import pandas as pd


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

        for (
            pattern_node_src,
            pattern_node_dest,
            target_edge_labels_code,
        ), db_graphs in self.extensions.items():

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

        for (
            pattern_node_src,
            pattern_node_dest,
            target_edge_labels_code,
        ), db_graphs in self.extensions.items():

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
        - The last column contains the location of the edges in the graphs.
        - The table is constructed in such a way that the rows are ordered by the number of 1 in the row.
    """

    def __init__(self, min_support):
        # set the column 'location' as the last column
        self.min_support = min_support
        self.df = pd.DataFrame(columns=["location"])

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
                self.df.iloc[i + 1 :] = self.df.iloc[i:-1]
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
        return len(row["location"].keys())

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
        in_edge_labels, out_edge_labels = EdgeGroupsFinder.split_into_in_and_out_array(
            edge_labels
        )
        return DirectedExtension(out_edge_labels, in_edge_labels, location)

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
                    location = row["location"]
                    # merge the location of the two rows
                    location_row_to_compare = row_to_compare["location"]
                    EdgeGroupsFinder.extend_location(location, location_row_to_compare)
                j -= 1

            if EdgeGroupsFinder.support(row) >= self.min_support:
                extensions.append(EdgeGroupsFinder.transform_row_in_extension(row))

        return extensions
