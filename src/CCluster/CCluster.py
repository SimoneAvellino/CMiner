class CCluster:
    def __init__(
        self,
        db_file,
        num_clusters,
        directed_graph=0,
        output_path=None,
        init_method="random",
        max_iter=100,
        tolerance=1e-4,
    ):
        self.db_file = db_file
        self.num_clusters = num_clusters
        self.directed_graph = directed_graph
        self.output_path = output_path
        self.init_method = init_method
        self.max_iter = max_iter
        self.tolerance = tolerance

    def cluster(self):
        raise NotImplementedError(
            "Graph clustering is in boilerplate stage and has not been implemented yet."
        )
