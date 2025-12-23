import json
import pandas as pd
from os import listdir
from os.path import join
from networkx import Graph, DiGraph, MultiGraph, MultiDiGraph
from networkx.readwrite import json_graph


# CONSTANTS
FP = "file_path"
FN = "file_name"
MR = "is_mr"
ND = "networks_path"
NP = "nodes_path"
EP = "edges_path"
LC = "label_col_nm"
ID = "id_col_nm"
SF = "sep_feat"
SL = "sep_label"
IDR = "is_directed"
IMG = "is_multigraph"


class NetworksLoading:
    def __init__(self, file_type, metadata, is_directed):
        self.Networks = {}
        self.is_directed = is_directed
        self.load_network(metadata, file_type)

    @staticmethod
    def _graph_type(is_directed, is_multigraph):
        """
        selects the type of NetworkX graph based on is_directed and is_multigraph parameters

        Graph        : undirected, self-loops allowed, parallel edges not allowed
        DiGraph      : directed,   self-loops allowed, parallel edges not allowed
        MultiGraph   : undirected, self-loops allowed, parallel edges allowed
        MultiDiGraph : directed,   self-loops allowed, parallel edges allowed

        :param is_directed: if true will be a directed graph
        :param is_multigraph: if true will be a multigraph

        :return: NetworkX graph type (MultiDiGraph, MultiGraph, DiGraph, Graph)
        """
        if is_directed and is_multigraph:
            return MultiDiGraph()
        if is_directed and not is_multigraph:
            return DiGraph()
        if not is_directed and is_multigraph:
            return MultiGraph()
        return Graph()

    @staticmethod
    def _multiple_rows(components, nodes_dict, sep):
        """
        THE FILE CONTAINS ONE OR MULTIPLE ROWS FOR EACH NODE (ONE PER LABEL)
        EXAMPLE:
           V 0 C
           V 0 A
           V 1 B

        :param components: node data (v, node_id, label)
        :param nodes_dict: nodes to load into the graph

        :return: None
        """
        try:
            node_id = int(components[1])
        except ValueError as exc:
            raise ValueError(
                f"Vertex id must be an integer, got '{components[1]}'"
            ) from exc
        if node_id not in nodes_dict:
            nodes_dict[node_id] = []
        label = components[2].strip()
        if label:
            nodes_dict[node_id].append(label)

    @staticmethod
    def _single_row(components, nodes_dict, sep):
        """
        THE FILE CONTAINS ONE ROW PER EACH NODE, SO THE LABELS ARE SEPARATED BY A USER-DESIRED SEPARATOR
        EXAMPLE
          V 0 A,B
          V 1 C

        :param components: components: node data (v, node_id, label)
        :param nodes_dict: nodes to load into the graph
        :return:
        """
        try:
            node_id = int(components[1])
        except ValueError as exc:
            raise ValueError(
                f"Vertex id must be an integer, got '{components[1]}'"
            ) from exc
        nodes_dict[node_id] = [lab for lab in components[2:] if lab.strip()]

    # CASE 1: .data LOADING
    def _load_data_file(self, metadata):
        """
        Load a network from .data file.

        :param metadata: this parameter contains:
           - file_path     : path to the networks file
           - file_name     : network file name
           - sep_label     : separator for the features (default is ","; used only labels into the same row)
           - is_mr         : if 1 the file contains multiple rows for each node
           - is_directed   : if true will be a directed graph
           - is_multigraph : if true will be a multigraph

        :return: None
        """
        nodes_dict = None
        edges_vect = None
        last_net_id = None
        node_func = self._multiple_rows if metadata[MR] else self._single_row
        for line in open(join(metadata[FP], metadata[FN])):
            components = line.replace("\n", "").split(" ")
            if components[0] == "t":
                if nodes_dict is not None:
                    self.Networks[last_net_id].add_nodes_from(
                        [(k, {"labels": v}) for k, v in nodes_dict.items()]
                    )
                    self.Networks[last_net_id].add_edges_from(edges_vect)
                try:
                    last_net_id = components[3]
                except:
                    # if the name is not set use the index of the graph as the name
                    last_net_id = components[2]
                # print(last_net_id + " LOADING")
                self.Networks[last_net_id] = self._graph_type(
                    self.is_directed, metadata[IMG]
                )
                nodes_dict = {}
                edges_vect = []
            # VERTICES SECTION
            elif components[0] == "v":
                node_func(components, nodes_dict, metadata[SL])
            # EDGES SECTION
            elif components[0] == "e":
                # Handle unlabeled edges: if no extra components, create one edge with empty type
                try:
                    src = int(components[1])
                    dst = int(components[2])
                except ValueError as exc:
                    raise ValueError(
                        f"Edge vertices must be integers, got '{components[1]}' and '{components[2]}'"
                    ) from exc

                edge_labels = [t for t in components[3:] if t.strip()]

                if len(edge_labels) == 0:
                    edges_vect.append((src, dst, {}))
                else:
                    for type in edge_labels:
                        edges_vect.append((src, dst, {"type": type}))
        # ADD NODES TO THE LAST GRAPH
        self.Networks[last_net_id].add_nodes_from(
            [(k, {"labels": v}) for k, v in nodes_dict.items()]
        )
        self.Networks[last_net_id].add_edges_from(edges_vect)
        return

    # CASE 2: .csv DATA LOADING
    def _load_csv_file(self, metadata):
        """
        Load a networks from .csv file.

        :param metadata: this parameter contains:
           - networks_path: path to the networks directory
           - nodes_path   : path to the nodes directory
           - edges_path   : path to the edges directory
           - sep_feat     : separator for the features
           - sep_label    : separator for the labels
           - id_col_nm    : column name for the node id
           - label_col_nm : column name for the node label
           - is_directed  : if true will be a directed graph
           - is_multigraph: if true will be a multigraph

        :return: None
        """
        # LOAD NODES
        for network in listdir(metadata[ND]):
            graph = self._graph_type(metadata[IDR], metadata[IMG])
            nodes_path = str(join(metadata[ND], network, metadata[NP]))
            for file in listdir(nodes_path):
                file_nodes = pd.read_csv(join(nodes_path, file), sep=metadata[SF])
                file_nodes[metadata[LC]] = file_nodes[metadata[LC]].str.split(
                    metadata[SL]
                )
                file_nodes = file_nodes.to_dict(orient="records")
                file_nodes = [(row.pop(metadata[ID]), row) for row in file_nodes]
                graph.add_nodes_from(file_nodes)
            # LOAD EDGES
            edges_path = str(join(metadata[ND], network, metadata[EP]))
            for file in listdir(edges_path):
                file_edges = pd.read_csv(join(edges_path, file), sep=metadata[SF])
                src_dst_cols = file_edges.columns[:2]
                file_edges = file_edges.to_dict(orient="records")
                file_edges = [
                    (row.pop(src_dst_cols[0]), row.pop(src_dst_cols[1]), row)
                    for row in file_edges
                ]
                graph.add_edges_from(file_edges)
            self.Networks[network] = graph
        return

    def load_network(self, metadata, file_type):
        if file_type == "csv":
            # CSV ELABORATION
            self._load_csv_file(metadata)
        elif file_type == "data":
            # DATA ELABORATION
            self._load_data_file(metadata)
        elif file_type == "json":
            # JSON ELABORATION
            for network_file in listdir(metadata[ND]):
                net_name = network_file.split(".")[0]
                js_graph = json.load(open(join(metadata[ND], network_file)))
                self.Networks[net_name] = json_graph.node_link_graph(js_graph)
        else:
            print("FILE TYPE NOT SUPPORTED")
