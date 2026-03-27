import time
import argparse
import sys
from CCluster import CCluster


def _build_base_parser():
    parser = argparse.ArgumentParser(description="CMiner algorithm")
    parser.add_argument("db_file", type=str, help="Path to graph db")
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "-s",
        "--support",
        type=float,
        help="Minimum support for mining",
        default=None,
    )
    mode_group.add_argument(
        "-c",
        "--num_clusters",
        type=int,
        help="Number of clusters for graph clustering",
        default=None,
    )
    return parser


def _build_mining_parser():
    parser = _build_base_parser()
    parser.add_argument(
        "-l", "--min_nodes", type=int, help="Minimum number of nodes", default=1
    )
    parser.add_argument(
        "-u",
        "--max_nodes",
        type=int,
        help="Maximum number of nodes",
        default=float("inf"),
    )
    parser.add_argument(
        "-n", "--num_nodes", type=int, help="Number of nodes", default=None
    )
    parser.add_argument(
        "-m", "--show_mappings", type=int, help="Show pattern mappings", default=0
    )
    parser.add_argument(
        "-o", "--output_path", type=str, help="Output file", default=None
    )
    parser.add_argument(
        "-t", "--templates_file", type=str, help="Starting template file", default=None
    )
    parser.add_argument(
        "-d",
        "--is_directed",
        type=int,
        help="Specify if the graph is directed",
        default=0,
    )
    parser.add_argument(
        "-f",
        "--with_frequencies",
        type=int,
        help="Show the relative frequencies of the pattern",
        default=0,
    )
    parser.add_argument(
        "-x", "--pattern_type", type=str, help="[all | maximum]", default="all"
    )
    parser.add_argument(
        "-w", "--worker", type=int, help="Number of parallel workers", default=1
    )
    return parser


def _build_clustering_parser():
    parser = _build_base_parser()
    parser.add_argument(
        "-d",
        "--is_directed",
        type=int,
        help="Specify if the graph is directed",
        default=0,
    )
    parser.add_argument(
        "-o", "--output_path", type=str, help="Output file", default=None
    )
    parser.add_argument(
        "--strategy",
        type=str,
        help="Distance-matrix strategy for clustering",
        default="simple_structural",
    )
    parser.add_argument(
        "--init_method",
        type=str,
        choices=["random", "kmeans++"],
        help="Cluster initialization method",
        default="random",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        help="Maximum iterations for the clustering algorithm",
        default=100,
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        help="Convergence tolerance for clustering",
        default=1e-4,
    )
    return parser


def main_function():
    argv = sys.argv[1:]
    has_support_mode = any(
        arg == "-s" or arg.startswith("--support") for arg in argv
    )
    has_cluster_mode = any(
        arg == "-c" or arg.startswith("--num_clusters") for arg in argv
    )

    if has_support_mode:
        args = _build_mining_parser().parse_args()
        from CMiner import CMiner

        if args.num_nodes is not None:
            args.min_nodes = args.num_nodes
            args.max_nodes = args.num_nodes

        miner = CMiner(
            args.db_file,
            support=args.support,
            min_nodes=args.min_nodes,
            max_nodes=args.max_nodes,
            show_mappings=args.show_mappings,
            output_path=args.output_path,
            templates_file=args.templates_file,
            directed_graph=args.is_directed,
            with_frequencies=args.with_frequencies,
            pattern_type=args.pattern_type,
            workers=args.worker,
        )

        start_time = time.time()

        try:
            miner.mine()
        except KeyboardInterrupt:
            print("\n-> Ctrl+C detected. Closing miner...")
            miner.close()
        finally:
            end_time = time.time()
            print(f"\n-> Execution time: {end_time - start_time} seconds")
        return

    if not has_cluster_mode:
        _build_base_parser().parse_args()
        return

    args = _build_clustering_parser().parse_args()
    clusterer = CCluster(
        db_file=args.db_file,
        num_clusters=args.num_clusters,
        directed_graph=args.is_directed,
        output_path=args.output_path,
        strategy=args.strategy,
        init_method=args.init_method,
        max_iter=args.max_iter,
        tolerance=args.tolerance,
    )

    start_time = time.time()

    try:
        clusterer.cluster()
    except NotImplementedError as exc:
        print(f"\n-> Clustering not available yet: {exc}")
    except KeyboardInterrupt:
        print("\n-> Ctrl+C detected. Closing clustering...")
    finally:
        end_time = time.time()
        print(f"\n-> Execution time: {end_time - start_time} seconds")
